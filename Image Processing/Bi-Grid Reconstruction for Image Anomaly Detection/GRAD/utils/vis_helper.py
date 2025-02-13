import os

import cv2
import numpy as np
from datasets.image_reader import build_image_reader
from tqdm import tqdm
from utils.eval_helper import sample_auroc
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import torch
import torch.nn.functional as F
# 忽略 UndefinedMetricWarning 警告
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float64)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def visualize_compound(fileinfos, preds, masks, cfg_vis, cfg_reader, threshold, image_preds):
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score
    image_reader = build_image_reader(cfg_reader)
    for i, fileinfo in tqdm(enumerate(fileinfos), total=len(fileinfos), desc="Visualize:"):
        auc = sample_auroc(preds[i], masks[i])
        clsname = fileinfo["clsname"]
        filename = fileinfo["filename"]
        filedir, filename = os.path.split(filename)
        _, defename = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)
        
        # Read image
        h, w = 224, 224
        image = image_reader(fileinfo["filename"])
        image = cv2.resize(image, (h, w))
        pred = preds[i][:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))
        
        # Self normalize just for analysis
        scoremap_self = apply_ad_scoremap(image, normalize(pred))  # Normalize with min/max of single sample
        
        # Global normalize
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)  # Normalize with min/max of whole mini batch
        scoremap_global = apply_ad_scoremap(image, pred)
        
        if masks is not None:
            mask = (masks[i] * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Determine if the image is anomalous based on the threshold
        is_anomalous = (image_preds[i] >= threshold).any()
        
        # Add text to the image to indicate anomaly status
        status_text = "Anomalous" if is_anomalous else "Normal"
        text_color = (255, 0, 0) if is_anomalous else (0, 255, 0)  # Red for Anomalous, Green for Normal
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        
        save_path = os.path.join(save_dir, filename)
        if masks is not None:
            if mask.sum() == 0:
                scoremap = np.vstack([image, scoremap_global])
            else:
                scoremap = np.vstack([image, mask, scoremap_global, scoremap_self])
        else:
            scoremap = np.vstack([image, scoremap_global, scoremap_self])
        
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap)


def visualize_single(fileinfos, preds, cfg_vis, cfg_reader):
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score

    image_reader = build_image_reader(cfg_reader)
    for i, fileinfo in enumerate(fileinfos):
        clsname = fileinfo["clsname"]
        filename = fileinfo["filename"]
        filedir, filename = os.path.split(filename)
        _, defename = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)

        # read image
        h, w = int(fileinfo["height"]), int(fileinfo["width"])
        image = image_reader(fileinfo["filename"])
        image = cv2.resize(image, (224, 224))
        pred = preds[i][:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))

        # write global normalize image
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred)

        save_path = os.path.join(save_dir, filename)
        scoremap_global = cv2.cvtColor(scoremap_global, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap_global)
