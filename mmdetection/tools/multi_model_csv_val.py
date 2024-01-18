# Copyright (c) OpenMMLab. All rights reserved.
"""Support for multi-model fusion, and currently only the Weighted Box Fusion
(WBF) fusion method is supported.

References: https://github.com/ZFTurbo/Weighted-Boxes-Fusion

Example:

     python demo/demo_multi_model.py demo/demo.jpg \
         ./configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_1x_coco.py \
         ./configs/retinanet/retinanet_r50-caffe_fpn_1x_coco.py \
         --checkpoints \
         https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth \  # noqa
         https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth \
         --weights 1 2
"""

import argparse
import os.path as osp
import os
from datetime import datetime

import mmcv
from mmengine.structures import InstanceData

from mmdet.apis import DetInferencer

# from mmdet.models.utils import weighted_boxes_fusion

import torch
from pycocotools.coco import COCO
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from ensemble_boxes import *
import pandas as pd
import numpy as np


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMDetection multi-model inference demo"
    )
    parser.add_argument(
        "type",
        type=str,
        default="wbf",  # avg, max, box_and_model_avg, absent_model_aware_avg
        help="ensemble type",
    )
    parser.add_argument(
        "csv",
        type=str,
        nargs="*",
        help="CSV file(s), support receive multiple files",
    )
    parser.add_argument(
        "--conf-type",
        type=str,
        default="avg",  # avg, max, box_and_model_avg, absent_model_aware_avg
        help="how to calculate confidence in weighted boxes in wbf",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="weights for each model, remember to " "correspond to the above config",
    )
    parser.add_argument(
        "--fusion-iou-thr",
        type=float,
        default=0.55,
        help="IoU value for boxes to be a match in wbf",
    )
    parser.add_argument(
        "--skip-box-thr",
        type=float,
        default=0.0,
        help="exclude boxes with score lower than this variable in wbf",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Output directory of images or prediction results.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    print(f"Ensemble Type : {args.type}")
    print(f"Ensemble weight : {args.weights}")
    print(f"Ensemble iou thr : {args.fusion_iou_thr}")
    print(f"Ensemble skip box thr : {args.skip_box_thr}")
    print(f"Ensemble wbf type : {args.conf_type}")

    now = datetime.now()
    out_path = osp.join(args.out_dir, "result", now.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(out_path, exist_ok=True)

    image_size = 1024
    val_json = "/home/hojun/Documents/code/boostcamp/project2/version1/dataset/val_eye_eda.json"
    results = []
    pred_base_list = []

    submission_df = [pd.read_csv(file) for file in args.csv]
    image_ids = submission_df[0]["image_id"].tolist()

    for i, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        prediction_string = ""
        boxes_list = []
        scores_list = []
        labels_list = []

        for df in submission_df:
            predict_string = df[df["image_id"] == image_id][
                "PredictionString"
            ].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list) == 0 or len(predict_list) == 1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                box[0] = float(box[0]) / image_size
                box[1] = float(box[1]) / image_size
                box[2] = float(box[2]) / image_size
                box[3] = float(box[3]) / image_size
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))
        results.append([boxes_list, scores_list, labels_list])

    for i in tqdm(range(len(results))):
        if args.type == "nms":
            bboxes, scores, labels = nms(
                results[i][0],
                results[i][1],
                results[i][2],
                weights=args.weights,
                iou_thr=args.fusion_iou_thr,
                # skip_box_thr=args.skip_box_thr,
                # conf_type=args.conf_type,
            )
        if args.type == "soft_nms":
            bboxes, scores, labels = soft_nms(
                results[i][0],
                results[i][1],
                results[i][2],
                weights=args.weights,
                iou_thr=args.fusion_iou_thr,
                mode=2,
                sigma=0.5
                # skip_box_thr=args.skip_box_thr,
                # conf_type=args.conf_type,
            )
        elif args.type == "nmw":
            bboxes, scores, labels = non_maximum_weighted(
                results[i][0],
                results[i][1],
                results[i][2],
                weights=args.weights,
                iou_thr=args.fusion_iou_thr,
                skip_box_thr=args.skip_box_thr,
            )
        elif args.type == "wbf":
            bboxes, scores, labels = weighted_boxes_fusion(
                results[i][0],
                results[i][1],
                results[i][2],
                weights=args.weights,
                iou_thr=args.fusion_iou_thr,
                skip_box_thr=args.skip_box_thr,
                conf_type=args.conf_type,
            )

        pred_instances = InstanceData()
        pred_instances.bboxes = bboxes
        pred_instances.scores = scores
        pred_instances.labels = labels

        pred_instances_dict = {"boxes": [], "scores": [], "labels": []}
        pred_instances_dict["boxes"] = torch.tensor(bboxes).to(args.device)
        pred_instances_dict["scores"] = torch.tensor(scores).to(args.device)
        pred_instances_dict["labels"] = torch.tensor(labels).to(args.device)

        pred_base_list.append(pred_instances_dict)

    coco = COCO(val_json)

    gt_base_list = []
    for imgs in tqdm(coco.imgs.values()):
        gt_instances = {"boxes": [], "labels": []}
        file_name = imgs["file_name"]

        image_info = coco.loadImgs(
            coco.getImgIds(imgIds=[int(file_name.split("/")[-1].split(".")[0])])
        )[0]
        annotation_ids = coco.getAnnIds(imgIds=image_info["id"])
        annotations = coco.loadAnns(annotation_ids)
        for annotation in annotations:
            bbox = annotation["bbox"]
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            class_id = annotation["category_id"]
            gt_instances["boxes"].append(bbox)
            gt_instances["labels"].append(class_id)

        gt_instances["boxes"] = torch.tensor(gt_instances["boxes"]).to(args.device)
        gt_instances["labels"] = torch.tensor(gt_instances["labels"]).to(args.device)
        gt_base_list.append(gt_instances)

    base_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    base_metric50 = MeanAveragePrecision(
        iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
    )
    for idx in tqdm(range(len(gt_base_list))):
        pred_base_list[idx]["boxes"] = torch.round(
            pred_base_list[idx]["boxes"] * image_size, decimals=1
        )
        pred_base_list[idx]["labels"] = pred_base_list[idx]["labels"].type(torch.int)
        base_metric.update([pred_base_list[idx]], [gt_base_list[idx]])
        base_metric50.update([pred_base_list[idx]], [gt_base_list[idx]])

    base_metric_score = base_metric.compute()
    base_metric50_score = base_metric50.compute()

    score_dict = {}
    base_score_names = ["mAP", "mAP50", "mAP75", "mAR_1", "mAR_10", "mAR_100"]
    base_socre_indexs = ["map", "map_50", "map_75", "mar_1", "mar_10", "mar_100"]
    labels = [
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    ]
    for score_name, score_index in zip(base_score_names, base_socre_indexs):
        score_dict[f"val_{score_name}"] = base_metric_score[score_index]
    for index, label in enumerate(labels):
        score_dict[f"val_{label}_mAP50"] = base_metric50_score["map_per_class"][index]
        score_dict[f"val_{label}_mAR100"] = base_metric_score["mar_100_per_class"][
            index
        ]

    with open(os.path.join(out_path, "log.txt"), "w") as f:
        f.write("Config & Weight List\n")
        for csv_name in args.csv:
            slice_config_name = csv_name.split("/")[-1]
            f.write(f"{slice_config_name}\n")
        f.write("Scores\n")
        for key, value in score_dict.items():
            f.write(f"{key} : {value}\n")
    print(score_dict)


if __name__ == "__main__":
    main()
