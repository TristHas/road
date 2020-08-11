import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import resnet_fpn_backbone, FastRCNNPredictor, MultiScaleRoIAlign, TwoMLPHead, RegionProposalNetwork, TwoMLPHead, RPNHead, AnchorGenerator, RoIHeads, GeneralizedRCNNTransform

from torchvision.models.detection.faster_rcnn import * #resnet_fpn_backbone, FastRCNNPredictor
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN, List, Tuple
from torchvision.models.detection import FasterRCNN
from collections import OrderedDict

def extract_embeds(targets):
    return torch.cat([t["country"] for t in targets]).long()

class Embed(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("embeds", torch.eye(n, dtype=torch.float))

    def forward(self, features, tgts):
        assert len(tgts)==features["0"].shape[0]
        t = extract_embeds(tgts)
        embeddings = self.embeds[t][:,:,None,None]
        ret = OrderedDict()
        for k,feat in features.items():
            h,w  = feat.shape[-2:]
            b,c  = embeddings.shape[:2]
            feat = torch.cat([embeddings.expand(b,c,h,w), feat], dim=1)
            ret[k]=feat
        return ret

class FasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))


        out_channels = backbone.out_channels + num_classes
        
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
        self.embed = Embed(num_classes)


    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        features = self.embed(features, targets)
        
        #if isinstance(features, torch.Tensor):
        #    features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)


def get_anchors(
                anchor_sizes  = ((32,), (64,), (128,), (256,), (512,)),
                aspect_ratios = (0.5, 1.0, 2.0)):
    """
    """
    aspect_ratios = (aspect_ratios,) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)
    
def load_backbone(model_name="resnet101", chkpt=None):
    """
        Current best backbone chkpt:
            "../../Checkpoints/models/class/resnet101_lr_0.005_annealing_6/epoch_23"
    """
    backbone = resnet_fpn_backbone(model_name, True)
    if chkpt:
        state = torch.load(chkpt, map_location="cpu")
        state = {k[9:]:v for k,v in state.items() if k.startswith("backbone.")}
        backbone.load_state_dict(state)
    return backbone
    
def get_model_detection(num_classes, 
                        model_name="resnet101", 
                        aspect_ratios = (0.5, 1.0, 2.0),
                        anchor_sizes  = ((32,), (64,), (128,), (256,), (512,)), 
                        backbone_chkpt=None,
                        model_chkpt=None,
                        train_backbone=True,
                        **kwargs):
    """
        
    """
    backbone = load_backbone(model_name, backbone_chkpt)
    anchors = get_anchors(aspect_ratios=aspect_ratios,
                          anchor_sizes=anchor_sizes)
    model = FasterRCNN(backbone, num_classes,
                       rpn_anchor_generator=anchors,
                       **kwargs) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if model_chkpt:
        backbone.load_state_dict(torch.load(model_chkpt, map_location="cpu"))
    
    if not train_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    
    return model