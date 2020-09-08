import os, pickle
from xml.etree import ElementTree

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

import transforms as T
from config import data_path as base_path

# Base helpers
def read_root(root):
    return [read_object(obj) for obj in root.iter('object')]

def read_object(obj):
    cs = ["xmin", "ymin", "xmax", "ymax"]
    cls_name = obj.find('name').text
    bbox = obj.find('bndbox')
    coord = [int(bbox.find(c).text) for c in cs]
    return cls_name, coord

def read_anno_file(file_path):
    return read_root(ElementTree.parse(open(file_path)).getroot())

def check_coords(coords):
    return (coords[0] != coords[2]) and (coords[1] != coords[3])

def filter_anno_coord(annos):
    failed = []
    for k,v in annos.items():
        newv = [v_ for v_ in v if check_coords(v_[1])]
        if len(v)!=len(newv):
            failed.append(k)
            annos[k] = newv
    return failed

def read_label_map(mode="class"):
    """
    """
    if mode == "noclass":
        lblmap = {
            'D00' : 1, 'D10' : 1,
            'D20' : 1, 'D40' : 1
        }
        names = ['BG', 'FG']
    elif mode == "class":
        lblmap = {'D00' : 1, 'D10' : 2,
                  'D20' : 3, 'D40' : 4}
        names = ['BG', 'D00', 'D10', 'D20', 'D40']
    elif mode == "class_all":
        lblmap = {
             'D00' : 1, 'D10' : 2,
             'D20' : 3, 'D40' : 4,
             'D01' : 5, 'D11' : 6,
             'D44' : 7, 'D43' : 8, 
             'D50' : 9, 'D0w0': 10
            }
        names = ['BG', 'D00', 'D10', 'D20', 'D40', 'D01', 'D11',
                 'D44', 'D43', 'D50', 'D0w0']
    elif mode == "class_bg":
        lblmap = {
             'D00' : 1, 'D10' : 2,
             'D20' : 3, 'D40' : 4,
             'D01' : 0, 'D11' : 0,
             'D44' : 0, 'D43' : 0, 
             'D50' : 0, 'D0w0': 0
            }
        names = ['BG', 'D00', 'D10', 'D20', 'D40']
    elif mode == "category":
        lblmap = {
             'D00': 1, 'D01': 1, 'D0w0':1,
             'D10': 2, 'D11': 2,
             'D20': 3, 
             'D40': 4, 'D44': 4, 'D43': 4, 
             'D50': 5
            }
        names = ["BG", 'D0', 'D1', 'D2', 'D4', 'D5']
    else:
        raise NotImplementedError
    return lblmap, names

def check_classes(cl_list, classes):
    """
    """
    return cl_list in classes

def filter_anno_classes(annos, classes):
    """
    """
    removed = []
    for k,v in annos.items():
        newv = [v_ for v_ in v if check_classes(v_[0], classes)]
        if len(v)!=len(newv):
            removed.append(k)
            annos[k] = newv
    return removed

def non_empties(annos):
    """
    """
    return [k for k,(v) in annos.items() if non_empty(anno)]

def non_empty(anno):
    return len(anno) > 0

def read_dataset_dict(classes=None):
    govs = ["Japan/", "India/", "Czech/"]
    anndir = 'annotations/xmls/'
    imgdir = "images/"
    annos = {}
    
    for gov in govs:
        file_list = [filename for filename in os.listdir(base_path + gov + anndir) if not filename.startswith('.')]
        for file_name in file_list:
            annid = file_name[:-4]
            file_path = base_path + gov + anndir + file_name
            anno = read_anno_file(file_path)
            annos[annid] = anno

    img_paths = {}
    for gov in govs:
        file_list = [filename for filename in os.listdir(base_path + gov + imgdir) if not filename.startswith('.')]
        for file_name in file_list:
            imgid = file_name[:-4]
            file_path = base_path + gov + imgdir + file_name
            img_paths[imgid] = file_path
            
    # Check no duplicate id
    assert len(set(img_paths.keys())) == len(img_paths.keys())
    # Check ids match for anno and imgs
    assert set(img_paths.keys()) == set(annos.keys())  
    # filter annotations
    if classes is not None:
        filter_anno_classes(annos, classes)
    filter_anno_coord(annos)
    #keys = non_empty(annos)
    #annos = {k:annos[k] for k in keys}
    #img_paths = {k:img_paths[k] for k in keys}
    return annos, img_paths

# DETECTION
def dump_random_split(keys, name="default", ratio=.15):
    valname = base_path + "../splits/" + name + "_val.npy"
    trainname = base_path + "../splits/" + name + "_train.npy"
    assert (not os.path.isfile(valname)) \
            and (not os.path.isfile(trainname))
    val = np.random.choice(keys, int(len(keys)*.15))
    train = np.array(list(set(keys)-set(val)))
    np.save(valname, val)
    np.save(trainname, train)
    
def load_split(name="default"):
    valname = base_path + "../splits/" + name + "_val.npy"
    trainname = base_path + "../splits/" + name + "_train.npy"
    return np.load(trainname), np.load(valname)

def get_transform(train):
    # Resize is done inside the model.
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class RoadDamageDataset(object):
    def __init__(self, mode="class", split="train", split_name="base", transforms=None):
        """
        """
        # data
        self.countries = {"Japan": torch.tensor([0]), "India":torch.tensor([1]), "Czech":torch.tensor([2])}
        self.category_map, self.category_names = read_label_map(mode)
        self.annos, self.imgs = read_dataset_dict(self.category_map)
        # keys
        keys = load_split(split_name)
        keys = keys[0] if split=="train" else keys[1]
        self.keys = [key for key in keys if key in self.annos]
        #transforms
        if transforms is None:
            self.transforms = get_transform(split=="train")
        else:
            self.transforms = transforms

    def __getitem__(self, idx):
        """
        """
        img = self._image(idx)
        target = self._target(idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def _image(self, idx):
        return Image.open(self.imgs[self.keys[idx]]).convert("RGB")

    def _target(self, idx):
        anno = self.annos[self.keys[idx]]
        if non_empty(anno):
            classes, boxes = zip(*anno)
            classes = [self.category_map[x] for x in classes]
            classes = torch.tensor(classes)
            boxes = torch.tensor(boxes)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0, 1), dtype=torch.int64)
        
        area = self._area(boxes)
        country = self._country(idx)
        iscrowd = torch.zeros(classes.shape).long()
        imid = torch.tensor([idx])
        return {
            "labels":classes, 
            "boxes":boxes,
            "area":area, 
            "image_id":imid, 
            "iscrowd": iscrowd,
            "country":country
               }
        
    def _area(self, boxes):
        return (boxes[:,0] - boxes[:,2]) * (boxes[:,1] - boxes[:,3])
        
    def _country(self, idx):
        return self.countries[self.keys[idx][:5]]
        
    def __len__(self):
        """
        """
        return len(self.keys)
    
    def nclass(self):
        return len(set(self.category_names))
    
def collate_fn(batch):
    return tuple(zip(*batch))
    
def load_det_dataset(mode="class", split_name="base",
                     transforms=None, 
                     batch_size=8, num_workers=8):
    """
        
    """
    # datasets
    dataset = RoadDamageDataset(split="train", split_name=split_name, mode=mode, 
                                transforms=transforms)
    dataset_test = RoadDamageDataset(split="val", split_name=split_name, mode=mode)

    # data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    return data_loader, data_loader_test