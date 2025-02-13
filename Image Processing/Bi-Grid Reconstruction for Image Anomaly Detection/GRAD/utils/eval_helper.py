import glob
import logging
import os

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import average_precision_score

def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  
    masks = outputs["mask"].cpu().numpy()  
    heights = outputs["height"].cpu().numpy()
    widths = outputs["width"].cpu().numpy()
    clsnames = outputs["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred"])
        masks.append(npz["mask"])
    preds = np.concatenate(np.asarray(preds), axis=0)
    masks = np.concatenate(np.asarray(masks), axis=0)
    return fileinfos, preds, masks


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        preds = F.avg_pool2d(torch.tensor(preds), 21, 1, 21//2).numpy()
        self.preds = preds
        self.masks = masks


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.uint8)
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc
    
    def eval_pr_auc(self):
        precision, recall, thresholds = metrics.precision_recall_curve(self.masks, self.preds)
        pr_auc = metrics.auc(recall, precision)
        return pr_auc

    def eval_thresholds(self):
            # Compute ROC curve and AUROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(
            self.masks, self.preds
        )
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        auroc = metrics.roc_auc_score(
            self.masks, self.preds
        )
        
        # Compute Precision-Recall curve
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            self.masks, self.preds
        )

        # Get the best precision, recall, and threshold
        closest_index = np.abs(pr_thresholds - optimal_threshold).argmin()
        best_precision = precision[closest_index]
        best_recall = recall[closest_index]
        best_threshold = pr_thresholds[closest_index]

        # Compute predictions based on the best threshold
        preds_binary = (self.preds >= best_threshold).astype(int)
        
        # Compute confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(self.masks, preds_binary).ravel()

        print("True Negative:", tn)
        print("False Positive:", fp)
        print("False Negative:", fn)
        print("True Positive:", tp)

        return best_threshold, best_recall, best_precision, self.preds


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1) 


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1) 


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size): 
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        ) 
        return preds.reshape(N, -1).max(axis=1)


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        # Default
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc
    
    def eval_pr_auc(self):
        precision, recall, thresholds = metrics.precision_recall_curve(self.masks, self.preds)
        pr_auc = metrics.auc(recall, precision)
        return pr_auc

    def eval_thresholds(self):
            # Compute ROC curve and AUROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(
            self.masks, self.preds
        )
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        auroc = metrics.roc_auc_score(
            self.masks, self.preds
        )
        
        # Compute Precision-Recall curve
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            self.masks, self.preds
        )

        # Get the best precision, recall, and threshold
        closest_index = np.abs(pr_thresholds - optimal_threshold).argmin()
        best_precision = precision[closest_index]
        best_recall = recall[closest_index]
        best_threshold = pr_thresholds[closest_index]
        return best_threshold, best_recall, best_precision
    
eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "imageAuroc": EvalImageMax,
    "pixelAuroc": EvalPerPixelAUC,
}


def performances(fileinfos, preds, masks, config):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    image_preds=np.empty(shape=(0,))
    best_threshold_list=[]
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)
        data_meta = EvalDataMeta(preds_cls, masks_cls)

        # auc
        if config.evaluator.metrics.get("auc", None):
            for metric in config.evaluator.metrics.auc:
                # metric: {'name': 'imageAuroc', 'kwargs': {'avgpool_size': [32, 32]}}
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                auc = eval_method.eval_auc()
                pr_auc = eval_method.eval_pr_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc
                ret_metrics["{}_{}_prauc".format(clsname, evalname)] = pr_auc

                if evalname == "imageAuroc":
                    best_threshold, best_recall, best_precision, image_preds_t = eval_lookup_table[evalname](data_meta, **kwargs).eval_thresholds()
                    image_preds = np.append(image_preds, image_preds_t)
                    ret_metrics["{}_{}_threshold".format(clsname, evalname)] = best_threshold
                    ret_metrics["{}_{}_recall".format(clsname, evalname)] = best_recall
                    ret_metrics["{}_{}_precision".format(clsname, evalname)] = best_precision


    if config.evaluator.metrics.get("auc", None):
        for metric in config.evaluator.metrics.auc:
            evalname = metric["name"]
            evalvalues_auc = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            evalvalues_prauc = [
                ret_metrics["{}_{}_prauc".format(clsname, evalname)]
                for clsname in clsnames
            ]
   
            mean_auc = np.mean(np.array(evalvalues_auc))
            mean_prauc = np.mean(np.array(evalvalues_prauc))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc
            ret_metrics["{}_{}_prauc".format("mean", evalname)] = mean_prauc
            ret_metrics["{}_{}_threshold".format("mean", evalname)] = best_threshold
            ret_metrics["{}_{}_recall".format("mean", evalname)] = best_recall
            ret_metrics["{}_{}_precision".format("mean", evalname)] = best_precision
    return ret_metrics,image_preds

def sample_auroc(pred, mask):
    preds_cls = [pred[None, ...]]
    masks_cls = [mask[None, ...]]

    preds_cls = np.concatenate(np.asarray(preds_cls), axis=0) 
    masks_cls = np.concatenate(np.asarray(masks_cls), axis=0) 
    data_meta = EvalDataMeta(preds_cls, masks_cls)

    auc = {}
    for metric in [{'name': 'imageAuroc', 'kwargs': {'avgpool_size': [16, 16]}},
                   {'name': 'pixelAuroc'}]:
        evalname = metric["name"]
        kwargs = metric.get("kwargs", {})
        eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
        auc[f'{evalname}'] = eval_method.eval_auc()

    return auc


def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = sorted(list(clsnames - set(["mean"]))) + ["mean"]
    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = sorted(list(set([k.rsplit("_", 2)[1] for k in auc_keys])))
        evalnames.append('imageAupr')
        evalnames.append('pixelAupr')
        evalnames.append('recall')
        evalnames.append('precision')
        evalnames.append('threshold')
        record = Report(["clsname"] + evalnames)
        for clsname in clsnames:
            clsvalues = [ret_metrics["{}_{}_auc".format(clsname, evalname)] for evalname in evalnames[:2]] + [ret_metrics["{}_{}_prauc".format(clsname, evalname)] for evalname in evalnames[:2]] + [ret_metrics["{}_{}_recall".format(clsname, evalname)] for evalname in evalnames[:1]] + [ret_metrics["{}_{}_precision".format(clsname, evalname)] for evalname in evalnames[:1]] + [ret_metrics["{}_{}_threshold".format(clsname, evalname)] for evalname in evalnames[:1]]
            record.add_one_record([clsname] + clsvalues)
        logger.info(f"\n{record}")
    return clsvalues[-1]
