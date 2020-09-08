import random

from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional as F
import torchvision


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    
class Pad(object):
    def __init__(self, size, fill=0, padding_mode="constant"):
        """
        """
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, target, size=None):
        """
        """
        size = self.size if size is None else size
        pad = self.compute_pad(img, size)
        padder = torchvision.transforms.Pad(pad, fill=self.fill, 
                                            padding_mode=self.padding_mode)
        return padder(img), target
    
    def compute_pad(self, img, size):
        #print("Etner pad")
        xs, ys = img.size
        #print(xs)
        padleft = (size - xs) // 2
        padright  = size - xs - padleft 
        #print(padleft, padright)
        
        #print(ys)
        padtop  = (size - ys) // 2
        padbottom = size - ys - padtop 
        #print(padtop, padbottom)
        
        pad = (padleft, padtop, padright, padbottom)
        return pad
    
def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    w, h = img.size
    if (w <= size and h <= size):
        return img
    if w > h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), interpolation)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), interpolation)
    
class Resize(object):
    def __init__(self, size=512):
        """
        """
        self.size = size
        
    def __call__(self, img, target, size=None):
        """
        """
        size = self.size if size is None else size
        return resize(img, self.size), target
        
class PadResize(object):
    def __init__(self, size=512):
        """
        """
        self.size = size
        self.pad  = Pad(size)
        self.resize = Resize(size)
        
    def __call__(self, img, target, size=None):
        #print("enter")
        #print(img.size)
        img, target = self.resize(img, target, size=size)
        #print(img.size)
        img, target = self.pad(img, target, size=size)
        #print(img.size)
        return img, target
    
from bbaug.policies import policies
from bbaug.policies.policies import POLICY_TUPLE

def default_policy():
    """
    Version of the policies used in the paper
    :rtype: List[List[POLICY_TUPLE_TYPE]]
    :return: List of policies
    """
    policy = [
      [
          POLICY_TUPLE('Shear_Y',    .4, 4),
          POLICY_TUPLE('Color',      .4, 1),
          POLICY_TUPLE('Sharpness',  .4, 1),
          POLICY_TUPLE('Rotate',     .2, 2),
          POLICY_TUPLE('Equalize',   .2, 1),
      ],
      [
          POLICY_TUPLE('Shear_Y',    .4, 4),
          POLICY_TUPLE('Color',      .4, 1),
          POLICY_TUPLE('Sharpness',  .4, 1),
          POLICY_TUPLE('Rotate',     .4, 1),
          POLICY_TUPLE('Brightness', .3, 3),
      ],
    ]
    return policy

class AugPolicy():
    def __init__(self, policy_id=default_policy):
        """
        """
        if isinstance(policy_id, int):
            policy = getattr(policies, policies.list_policies()[policy_id])
            self.container = policies.PolicyContainer(policy())
        elif callable(policy_id):
            self.container = policies.PolicyContainer(policy_id())
        else:
            raise Exception
            
    def __call__(self, img, tgt):
        """
        """
        tgt_aug = {k:v for k,v in tgt.items()}
        pol = self.container.select_random_policy()
        img = np.array(img)
        img_aug, bb = self.container.apply_augmentation(pol, img, tgt["boxes"].tolist(), tgt["labels"].tolist())
        if len(bb.shape) > 1:
            tgt_aug["boxes"] = torch.from_numpy(bb[:,1:])
            tgt_aug["labels"] = torch.from_numpy(bb[:, 0]).long()
            boxes = tgt_aug["boxes"]
            tgt_aug["area"] = (boxes[:,0] - boxes[:,2]) * (boxes[:,1] - boxes[:,3])
            if (tgt_aug["area"] > 10).all().item():
                tgt = tgt_aug
                img = img_aug
        return F.to_tensor(img), tgt
