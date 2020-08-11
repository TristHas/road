import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import resnet_fpn_backbone, FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

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
                        aspect_ratios=(0.5, 1.0, 2.0),
                        anchor_sizes=((32,), (64,), (128,), (256,), (512,)), 
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