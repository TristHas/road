import math
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from config import LOG_DIR

flatten = lambda nest: [x for y in nest for x in y ]

class Monitor():
    def __init__(self, logname, subdir=""):
        if subdir:
            logdir = f"{LOG_DIR}/{subdir}/{logname}"
            print(logdir)
        else:
            logdir = f"{LOG_DIR}/{logname}"
        self.writer = SummaryWriter(logdir)
        self.steps = {}
        
    def update(self, data):
        for k,v in data.items():
            i = self.step(k)
            self.writer.add_scalar(k, v, i)
            
    def step(self, key):
        if key not in self.steps:
            self.steps[key] = 0
        self.steps[key] = self.steps[key] + 1
        return self.steps[key]
    
def model_device(model):
    return next(model.parameters()).device

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_epoch(model, optimizer, data_loader, monitor, epoch):
    device = model_device(model)
    model.train()
    lr_scheduler = None
    
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i, (images, targets) in enumerate(tqdm(data_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        if not math.isfinite(losses):
            print("Loss is {}, stopping training".format(losses))
            print(loss_dict)
            return model, images, targets

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        monitor.update({"Training/"+k:v for k,v in loss_dict.items()})
        monitor.update({"Training/loss": loss_value, 
                        "Parameters/lr":optimizer.param_groups[0]["lr"]})

@torch.no_grad()
def evaluate_coco(model, data_loader):
    """
    """
    import torchvision
    def _get_iou_types(model):
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types

    device = model_device(model)
    cpu_device = torch.device("cpu")

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    model.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in tqdm(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    # accumulate predictions from all images
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def get_coco_stat(tester):
    stats = tester.coco_eval["bbox"].stats
    return {
     "Validation/P (IOU=50:95)":stats[1],
     "Validation/P (IOU=50)":stats[1],
     "Validation/P (IOU=75)":stats[2],
     "Validation/R (IOU)=50:95":stats[8],
    }

def lr_schedule(data_loader, optimizer, nepochup=1, nepochdown=10, maxlr=0.001):
    lenepoch = len(data_loader) + 1
    warmup = np.linspace(0, maxlr, nepochup*lenepoch)
    phase_down = np.linspace(maxlr, 0, nepochdown*lenepoch)
    sched = np.concatenate([warmup, phase_down])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:sched[x])

def train_epoch_linear(model, optimizer, data_loader, monitor, lr_scheduler=None):
    device = model_device(model)
    model.train()

    for i, (images, targets) in enumerate(tqdm(data_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        if not math.isfinite(losses):
            print("Loss is {}, stopping training".format(losses))
            print(loss_dict)
            return model, images, targets

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        monitor.update({"Training/"+k:v for k,v in loss_dict.items()})
        monitor.update({"Training/loss": loss_value, 
                        "Parameters/lr":optimizer.param_groups[0]["lr"]})
