import collections
import PIL
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan'
]

def tohwc(img):
    ndim = len(img.shape)
    if ndim == 2:
        return img
    elif ndim == 3:
        if (img.shape[0] in {1,3}) and (not (img.shape[-1] in {1,3})):
            return np.transpose(img, (1,2,0))
        else:
            return img
    else:
        raise NotImplementedError("Wrong image dimension")
        
def touint8(img):
    if img.max() <= 1:
        img = img * 255
    assert 0 < img.max() < 256
    return img.astype("uint8")
    
def topil(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if isinstance(img, np.ndarray):
        img = tohwc(img)
        img = touint8(img)
        img = PIL.Image.fromarray(img)
    if isinstance(img, PIL.Image.Image):
        return img
    else:
        raise NotImplementedError
        
def plot_sample(img, anno, categories=[], thr=.3, 
                color=None, thickness=4):
    boxes = anno["boxes"].float()
    classes = anno["labels"]
    scores = anno["scores"] if "scores" in anno else torch.ones_like(classes)
    img = draw_anno(img, boxes, classes, scores, categories, 
                    thr=thr, color=color, 
                    thickness=thickness)
    return img

def plot_result(img, gt, res, categories=[], thr=.3):
    img = plot_sample(img, gt, categories, color="blue")
    img = plot_sample(img, res, categories, color="red", 
                      thr=thr, thickness=2)
    return img

def plot_results(model, data_loader, N=9):
    model.eval()
    device = next(model.parameters()).device
    cpu_device="cpu"
    n = math.ceil(math.sqrt(N))
    fig, ax = plt.subplots(n, n, figsize=(20, 20))

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i == N:
                break
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            img = plot_result(images[0].cpu(), targets[0], outputs[0], 
                              categories=data_loader.dataset.category_names)
            ax[i//n, i%n].imshow(img)

def draw_anno(image, boxes, classes, scores,
              categories=[], color=None,
              thr=.5,
              norm_coord=False,
              max_boxes_to_draw=20,
              agnostic_mode=False,
              thickness=4,
              ):
    image = topil(image)
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
                
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > thr:
            box = tuple(boxes[i].tolist())
            if classes[i] < len(categories):
                class_name = categories[classes[i]] 
            else:
                class_name = classes[i]
            display_str = '{}: {}%'.format( class_name, int(100*scores[i]))
            box_to_display_str_map[box].append(display_str)
            box_to_color_map[box] = COLORS[classes[i] % len(COLORS)]

    # Draw all boxes onto image.
    for box, color_ in box_to_color_map.items():
        xmin, ymin, xmax, ymax = box
        
        draw_bbox( image, ymin, xmin, ymax, xmax,
            color=color or color_,
            thickness=thickness,
            display_str_list=box_to_display_str_map[box],
            norm_coord=norm_coord)
    return image

def draw_bbox(image, ymin, xmin, ymax, xmax,
              color='red', thickness=4, 
              display_str_list=(),
              norm_coord=True):

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if norm_coord:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
        
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin