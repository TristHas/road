from data import *
from helpers import model_device
from config import submission_data_path as base_path

class RoadDamageDatasetTest(RoadDamageDataset):
    def __init__(self):
        """
        """
        govs = ["Japan/", "India/", "Czech/"]
        
        imgdir = "images/"
        self.imgs = {}
        for gov in govs:
            file_list = [filename for filename in os.listdir(base_path + gov + imgdir) if not filename.startswith('.')]
            for file_name in file_list:
                imgid = file_name[:-4]
                file_path = base_path + gov + imgdir + file_name
                self.imgs[imgid] = file_path
        self.keys = list(self.imgs.keys())
        self.transforms = T.ToTensor()

    def __getitem__(self, idx):
        """
        """
        img = self._image(idx)
        return self.transforms(img, None)[0], self.keys[idx]

    def _image(self, idx):
        return Image.open(self.imgs[self.keys[idx]]).convert("RGB")
        
    def __len__(self):
        """
        """
        return len(self.keys)

def get_test_outputs(model, thr=.5, maxbbox=5):
    device = model_device(model)
    data_loader = torch.utils.data.DataLoader(
        RoadDamageDatasetTest(), batch_size=1, shuffle=False, 
        num_workers=1, collate_fn=utils.collate_fn
    )
    model.eval()
    with torch.no_grad():
        results = {}
        for imgs, keys in tqdm(data_loader):
            annos = model.forward([img.to(device) for img in imgs])
            #annos = [select_bboxes(anno, thr, maxbbox) for anno in annos]
            for key, anno in zip(keys, annos):
                results[key] = anno
    return results

def select_bboxes(anno, thr=.5, maxbbox=5, no_empty=False):
    scores = anno["scores"]
    # First filter by thr
    msk = torch.where(scores > thr)[0]
    boxes  = anno["boxes"][msk]
    labels = anno["labels"][msk]
    scores = scores[msk]
    # Then sort by thr and filter by maxbbox
    msk = torch.argsort(scores, descending=True)[:maxbbox]
    return { 
        "boxes" : anno["boxes"][msk],
        "labels" : anno["labels"][msk]
    }

def format_anno(anno, sep=" "):
    N = anno["labels"].shape[0]
    results = []
    for i in range(N):
        lbl = anno["labels"][i]
        xmin, ymin, xmax, ymax = anno["boxes"][i]
        results.extend([str(int(x.item())) for x in [lbl, xmin, ymin, xmax, ymax]])
    return sep.join(results)

def format_results(annos, sep=",", det_sep=" "):
    res = []
    print(sep)
    for k, anno in annos.items():
        line = k + ".jpg" + sep +  format_anno(anno, det_sep)
        res.append(line)
    return "\n".join(res)

def model_submission(model, name="default"):
    results = get_test_outputs(model)
    results = format_results(results, sep=" ")
    path = f"/home/tristan/road_project/workspace/Damage Detection/submissions/{name}"
    assert not os.path.isfile(path)
    with open(path, "w") as f:
        f.write(results)