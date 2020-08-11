import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from helpers import model_device
from visualize import plot_result

def get_dt(output):
    dt_score = output["scores"].detach().cpu().numpy()
    dt_bbox  = output["boxes"].detach().cpu().numpy()
    dt_lbl   = output["labels"].cpu().numpy()
    return dt_bbox, dt_lbl, dt_score

def get_gt(target):
    gt_bbox = target["boxes"].detach().cpu().numpy()
    gt_lbl = target["labels"].cpu().numpy()
    return gt_bbox, gt_lbl

def intersect_boxes(A, B):
    """
    """
    mins = np.maximum(A[:,None, :2], B[None, :, :2])
    maxs = np.minimum(A[:,None, 2:], B[None, :, 2:])
    return np.concatenate([mins, maxs], axis=-1)

def area(bboxes):
    """
    """
    if bboxes.ndim==2:
        return np.maximum(0, (bboxes[:,2:] - bboxes[:,:2])).prod(-1)
    elif bboxes.ndim==3:
        return np.maximum(0, (bboxes[:,:,2:] - bboxes[:,:,:2])).prod(-1)
    else:
        raise NotImplemented
        
def union(A, B):
    """
    """
    inter = intersection(A, B)
    return area(A) + area(B) - inter
    
def intersection(A,B):
    """
    """
    return area(intersect_boxes(A, B))

def iou_fn(A, B):
    """
    """
    if B.shape[0]:
        i = intersection(A, B)
        return i / (area(A)[:,None] + area(B)[None,:] - i)
    else:
        #Handle empty labels
        # ok
        return np.zeros(A.shape[0])

def acc_fn(A, B):
    if B.shape[0]:
        return A[:,None] == B[None,:]
    else:
        #Handle empty labels
        return np.zeros_like(A, dtype=bool)
    
def histogram(df, x, y, log=False, nbins=10, fill_empty=-.1):

    s = df.set_index(x)[y]

    base = s.index.values
    if log:
        base = np.log10(base)
    edges  = np.histogram_bin_edges(base, bins=nbins)
    start, end = edges[None, :-1], edges[None, 1:] 
    msks = np.logical_and(base[:,None] >= start, base[:,None] <= end)

    x,y = np.where(msks)
    xy = pd.Series(x, index=y).drop_duplicates()
    x,y = xy.values, xy.index.values

    bins = y
    result = s.groupby(bins).mean()
    n = s.groupby(bins).sum() / result

    result = result.reindex(np.arange(start.shape[1])).fillna(fill_empty)
    n = n.reindex(np.arange(start.shape[1])).fillna(0)
    index = start.flatten()
    if log:
        index=10**index
    result.index = index.astype(int)
    n.index = index.astype(int)
    return result, n

def get_country_key(keys, country="India"):
    return list(filter(lambda x:x.startswith(country), keys))

def get_country_key_index(keys, country="India"):
    return [i for i,x in enumerate(keys) if x.startswith(country)]

def filter_dt(dets, thr=0, keys=None, maxbb=None):
    """
        Filter detections by confidence threshold and maximum allowed 
        detections per image.
    """
    if keys is None:
        keys = ['iou', 'acc', 'dt_bbox', 'dt_lbl', 'dt_score']
    msk  = dets["dt_score"] > thr
    gt_lbl, gt_bbox = dets["gt_lbl"], dets["gt_bbox"]
    dets = {k:dets[k][msk] for k in keys}
    msk  = np.argsort(dets["dt_score"])[::-1][:maxbb]
    dets = {k:dets[k][msk] for k in keys}
    # Needed for empty check
    dets["gt_lbl"] = gt_lbl
    dets["gt_bbox"] = gt_bbox
    return dets

def is_empty(dets):
    """
        Checck wether the image contains no annotations
    """
    return dets["gt_lbl"].shape[0] == 0

def mark_duplicates_fp(tps):
    """
        In case of multiple detections matching a same groundtruth,
        only leave one a tp and turn the other ones to fp.
    """
    X,Y = np.where(tps)
    msk = np.unique(Y, return_index=True)[1]
    X,Y = X[msk], Y[msk]
    tps = np.zeros(tps.shape, dtype=bool)
    tps[X,Y] = True
    return tps

def tp(dets, thr=.5, acc=True, remove_duplicates=True):
    """
        Return tp annotations give thres, iou and (optionnally) acc
    """
    if acc:
        tps = np.logical_and(dets["acc"], (dets["iou"] > thr))
    else:
        tps = dets["iou"] > thr
    # Remove duplicates
    if remove_duplicates:
        tps = mark_duplicates_fp(tps)
    return tps

def recall(dets, thr, acc=True, remove_duplicates=True):
    """
    """
    if is_empty(dets):
        return np.array([])
    else:
        return tp(dets, thr, acc).any(0)

def precision(dets, thr, acc=True, remove_duplicates=True):
    """
    """
    if is_empty(dets):
        return np.zeros_like(dets["dt_lbl"])
    else:
        return tp(dets, thr, acc).any(1)
    
def fscore(p, r):
    return 2 * p * r / (p + r)
    
def bbox_features(df):
    """
    """
    df["h"] = df["ymax"] - df["ymin"]
    df["w"] = df["xmax"] - df["xmin"]
    df["area"] = df["w"] * df["h"]
    df["ratio"] = df["h"] / df["w"]
    df["minside"] = np.minimum(df["h"], df["w"])
    
def summarize_stats(results, sc_thr=.3, iou_thr=.5, maxbb=None):
    """
    """
    r_stats = []
    p_stats = []

    for k, res in results.items():
        dt_res = filter_dt(res, thr=sc_thr, maxbb=maxbb)
        p  = precision(dt_res, thr=iou_thr, acc=True)
        pa = precision(dt_res, thr=iou_thr, acc=False)
        r  = recall(dt_res, thr=iou_thr, acc=True)
        ra = recall(dt_res, thr=iou_thr, acc=False)

        if is_empty(res):
            gt_lbl = np.array([])
        else:
            gt_lbl = res["gt_lbl"]

        p_stats.append((p, pa, dt_res["dt_bbox"], dt_res["dt_lbl"], dt_res["dt_score"], [k]*len(p)))
        r_stats.append((r, ra, dt_res["gt_bbox"], gt_lbl, [k]*len(r)))

    p, pa, bb, lbl, s, k = map(np.concatenate, zip(*p_stats))
    df_p = pd.DataFrame(np.concatenate([p[:,None], pa[:,None], bb, lbl[:,None], s[:,None], k[:,None]], axis=1),
                                     columns = ["precision", "precision_agno", "xmin", "ymin", "xmax", "ymax", "lbl", "score", "img"])
    bbox_features(df_p)

    r, ra, bb, lbl, k = map(np.concatenate, zip(*r_stats))
    df_r = pd.DataFrame(np.concatenate([r[:,None], ra[:,None], bb, lbl[:,None], k[:,None]], axis=1),
                      columns = ["recall", "recall_agno", "xmin", "ymin", "xmax", "ymax", "lbl", "img"])
    bbox_features(df_r)
    return df_p, df_r

@torch.no_grad()
def extract_results(model, data_loader):
    """
    """
    model.eval()
    device = model_device(model)
    cpu_device = "cpu"
    results = {} 
    
    for images, targets in tqdm(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        for output, target in zip(outputs, targets):
            key = target["image_id"].item()
            dt_bbox, dt_lbl, dt_score = get_dt(output)
            gt_bbox, gt_lbl = get_gt(target)
            acc = acc_fn(dt_lbl, gt_lbl)
            iou = iou_fn(dt_bbox, gt_bbox)
            results[key] = {
                "iou":iou, "acc":acc, 
                "gt_bbox":gt_bbox, "gt_lbl":gt_lbl, 
                "dt_bbox":dt_bbox, "dt_lbl":dt_lbl, 
                "dt_score":dt_score,
            }
    return results

class DetectionEvaluator():
    def __init__(self, model, dl):
        """
        """
        self.dl = dl
        self.model = model
        self.update_results(model, dl)
        self.cache = {}
        self.class_only = [1, 2, 3, 4]
        
    def update_results(self, model=None, dl=None):
        """
        """
        model = self.model if model is None else model
        dl = self.dl if dl is None else dl
        self.idx = sorted(set(self.dl.dataset.category_map.values()))
        self.names = self.dl.dataset.category_names
        self.results = extract_results(model, dl)
        self.cache = {}
        
    def filter_country(self, x, country=None):
        if country is None:
            return x
        else:
            assert country in ["India", "Japan", "Czech"]
            return x.loc[get_country_key_index(self.dl.dataset.keys, country)]

    def summarize(self, score_thr, iou_thr=.5, country=None, maxbb=None):
        """
        """
        if not ((score_thr, iou_thr) in self.cache):
            self.cache[(score_thr, iou_thr, maxbb)] = summarize_stats(self.results, score_thr, iou_thr, maxbb=maxbb) 
        p,r = self.cache[(score_thr, iou_thr, maxbb)]
        p,r = self.filter_country(p, country), self.filter_country(r, country)
        return p,r
    
    def precision(self, score_thr, iou_thr=.5, agnostic=False, 
                  class_only=False, country=None, maxbb=None):
        """
        """
        p, r = self.summarize(score_thr, iou_thr, country=country, maxbb=maxbb)
        key = "precision_agno" if agnostic else "precision"
        if class_only:
            p = p[p["lbl"].isin(self.class_only)]
        return p[key].mean()
        
    def recall(self, score_thr, iou_thr=.5, agnostic=False, 
               class_only=False, country=None, maxbb=None):
        """
        """
        p, r = self.summarize(score_thr, iou_thr, country=country)
        key = "recall_agno" if agnostic else "recall"
        if class_only:
            r = r[r["lbl"].isin(self.class_only)]
        return r[key].mean()
        
    def fscore(self, score_thr, iou_thr=.5, agnostic=False, 
               class_only=False, country=None, maxbb=None):
        """
        """
        p = self.precision(score_thr, iou_thr, agnostic, class_only, 
                           country=country, maxbb=maxbb)
        r = self.recall(score_thr, iou_thr, agnostic, class_only,
                        country=country, maxbb=maxbb)
        return fscore(p, r)
    
    def precision_per_country(self, score_thr, iou_thr=.5, agnostic=False, class_only=False):
        return [self.precision(score_thr, iou_thr, agnostic, class_only, country) \
               for country in [["India", "Japan", "Czech"]]]
    
    def precision_per_size(self, score_thr, size="area", iou_thr=.5, agnostic=False, 
                           log=False, nbins=10):
        p, r = self.summarize(score_thr, iou_thr)
        key = "precision_agno" if agnostic else "precision"
        return histogram(p, size, key, log=log, nbins=nbins)
    
    def precision_per_class(self, score_thr, iou_thr=.5, agnostic=False):
        """
        """
        p, r = self.summarize(score_thr, iou_thr)
        s = p.groupby("lbl").mean()["precision"]
        s = s.reindex(self.idx).fillna(0)#.set_index(names)
        s = s.set_axis(self.names[1:])
        return s#p.groupby("lbl").mean()["precision"].values
    
    def recall_per_class(self, score_thr, iou_thr=.5, agnostic=False):
        """
        """
        p, r = self.summarize(score_thr, iou_thr)
        s = r.groupby("lbl").mean()["recall"]
        s = s.reindex(self.idx).fillna(0)#.set_index(names)
        s = s.set_axis(self.names[1:])
        return s#r.groupby("lbl").mean()["recall"].values
    
    def recall_per_size(self, score_thr, size="area", iou_thr=.5, agnostic=False,
                        log=False, nbins=10):
        p, r = self.summarize(score_thr, iou_thr)
        key = "recall_agno" if agnostic else "recall"
        return histogram(r, size, key, log=log, nbins=nbins)
    
    def fscore_per_class(self, score_thr, iou_thr=.5, agnostic=False):
        """
        """
        raise NotImplemented
        
    def plt_precision_per_class(self, score_thr, iou_thr=.5, agnostic=False, ax=None):
        """
        """
        accs = self.precision_per_class(score_thr, iou_thr, agnostic)
        accs.plot.bar(ax=ax, title="Precision per class")
        
    def plt_recall_per_class(self, score_thr, iou_thr=.5, agnostic=False, ax=None):
        """
        """
        accs = self.recall_per_class(score_thr, iou_thr, agnostic)
        accs.plot.bar(ax=ax, title="Recall per class")
        
    def plt_precision_per_size(self, score_thr, size="area", iou_thr=.5, agnostic=False,
                                     log=False, nbins=10, ax=None):
        """
        """
        if ax is None:
            _,ax = plt.subplots(1,2, figsize=(10, 5))
        ax = plt if ax is None else ax
        hist, count = self.precision_per_size(score_thr, size, iou_thr, agnostic,
                                              log=log, nbins=nbins)
        hist.plot.bar(ax=ax[0])
        count.plot.bar(ax=ax[1])   
        ax[0].set_title(f"Precision per {size}")
        ax[1].set_title(f"Population per {size}")
        
    def plt_recall_per_size(self, score_thr, size="area", 
                            iou_thr=.5, agnostic=False,
                            log=False, nbins=10, ax=None):
        """
        """
        if ax is None:
            _,ax = plt.subplots(1,2, figsize=(10, 5))
        ax = plt if ax is None else ax
        hist, count = self.recall_per_size(score_thr, size, iou_thr, agnostic,
                                           log=log, nbins=nbins)
        hist.plot.bar(ax=ax[0])
        count.plot.bar(ax=ax[1])
        ax[0].set_title(f"Recall per {size}")
        ax[1].set_title(f"Population per {size}")

    def plt_fscores(self, agnostic=False, iou_thr=.5, scores=None, ax=None, 
                    class_only=False, maxbb=None, **kwargs):
        ax = plt if ax is None else ax
        scores = [i/10-.01 for i in range(1,11)] if scores is None else scores
        ys = [self.fscore(score_thr=x, iou_thr=iou_thr, agnostic=agnostic, 
                          class_only=class_only, maxbb=maxbb) for x in scores]
        #return scores, ys
        ax.plot(scores, ys, **kwargs)
        ax.set_title("F-scores")

    def plt_prec_rec(self, agnostic=False, iou_thr=.5, scores=None, ax=None, 
                     class_only=False, maxbb=None, **kwargs):
        ax = plt if ax is None else ax
        scores = [i/10-.01 for i in range(1,11)] if scores is None else scores
        ys = [self.precision(score_thr=x, iou_thr=iou_thr, agnostic=agnostic, 
                             maxbb=maxbb, class_only=class_only) for x in scores]
        xs = [self.recall(score_thr=x, iou_thr=iou_thr, agnostic=agnostic, 
                          maxbb=maxbb, class_only=class_only) for x in scores]
        ax.plot(xs, ys, **kwargs)
        ax.set_title("Precision / recall")

    def plt_summary(self, score_thr, iou_thr=.5, size="area", 
                    log=True, class_only=False, maxbb=None):
        fig, ax = plt.subplots(2,4, figsize=(20,10))
        
        self.plt_prec_rec(agnostic=True, iou_thr=iou_thr, 
                          class_only=class_only, ax=ax[0,0], maxbb=maxbb)
        self.plt_prec_rec(agnostic=False, iou_thr=iou_thr, 
                          class_only=class_only, ax=ax[0,0], maxbb=maxbb)

        self.plt_fscores(agnostic=True, iou_thr=iou_thr, 
                         class_only=class_only, ax=ax[1,0], maxbb=maxbb)
        self.plt_fscores(agnostic=False, iou_thr=iou_thr, 
                         class_only=class_only, ax=ax[1,0], maxbb=maxbb)

        self.plt_precision_per_class(score_thr, iou_thr, ax=ax[0,1])
        self.plt_recall_per_class(score_thr, iou_thr, ax=ax[1,1])

        self.plt_recall_per_size(score_thr, iou_thr=iou_thr, ax=ax[:,2], size=size, log=log)
        self.plt_precision_per_size(score_thr, iou_thr=iou_thr, ax=ax[:,3], size=size, log=log)
        
    def draw_result(self, key=0, sc_thr=.5):
        img, gt = self.dl.dataset[key]
        res = filter_dt(self.results[key], sc_thr)
        res = {"boxes":torch.from_numpy(res['dt_bbox']), "labels":res['dt_lbl'], "scores":res["dt_score"]}
        return plot_result(img, gt, res, categories=self.names)

    def plt_sorted_results(self, sc_thr=.5, offset=1, sorting="precision", ax=None):
        """
        """
        if ax is None:
            fig, ax = plt.subplots(1,2, figsize=(20, 10))
        p,r = summarize_stats(self.results, sc_thr)
        if sorting=="precision":
            imprec = p.groupby("img").mean()["precision"].sort_values()
        elif sorting=="recall":
            imprec = r.groupby("img").mean()["recall"].sort_values()
        idxs = imprec.index.values.astype(int)
        ax[0].imshow(self.draw_result(idxs[offset], sc_thr=sc_thr))
        ax[0].set_title(f"Img {idxs[offset]}: {sorting} {imprec[idxs[offset]]}")
        ax[1].imshow(self.draw_result(idxs[-offset], sc_thr=sc_thr))
        ax[1].set_title(f"Img {idxs[-offset]}: {sorting} {imprec[idxs[-offset]]}")
        return imprec

    
def evaluate(model, data_loader):
    dt_eval = DetectionEvaluator(model, data_loader)
    eval_data = {
        "Validation/precision" : dt_eval.precision(.5),
        "Validation/recall"    : dt_eval.recall(.5),
        "Validation/fscore"    : np.nan_to_num(dt_eval.fscore(.5)),
        "Validation/sub_precision" : dt_eval.precision(.5, class_only=True),
        "Validation/sub_recall"    : dt_eval.recall(.5, class_only=True),
        "Validation/sub_fscore"    : np.nan_to_num(dt_eval.fscore(.5, class_only=True)),
    }
    return dt_eval, eval_data