import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_detection(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_model_detection(num_classes, model_name="resnet101", device="cuda",
                        backbone_path=False):
    """
    """
    #"../../Checkpoints/models/class/resnet101_lr_0.005_annealing_6/epoch_23"
    if backbone_path:
        model,_ = get_model_detection(num_classes, model_name=model_name, backbone_path=None)
        model.load_state_dict(torch.load(backbone_path, map_location=device));
        backbone = model.backbone
    else:
        backbone = torchvision.models.detection.faster_rcnn.resnet_fpn_backbone(model_name, True)

    model = torchvision.models.detection.FasterRCNN(backbone, num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    if backbone_path:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    
    #params = list(chain(model.rpn.parameters(), model.roi_heads.parameters()))
    return model#, params
