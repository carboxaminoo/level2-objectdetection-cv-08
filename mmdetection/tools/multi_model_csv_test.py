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
    results = []

    submission_df = [pd.read_csv(file) for file in args.csv]
    image_ids = submission_df[0]["image_id"].tolist()

    prediction_strings = []
    file_names = []

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

            for bbox in predict_list[:, 2:6].tolist():
                bbox[0] = float(bbox[0]) / image_size
                bbox[1] = float(bbox[1]) / image_size
                bbox[2] = float(bbox[2]) / image_size
                bbox[3] = float(bbox[3]) / image_size
                box_list.append(bbox)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))
        results.append([boxes_list, scores_list, labels_list])

        if len(boxes_list):
            if args.type == "nms":
                bboxes, scores, labels = nms(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=args.weights,
                    iou_thr=args.fusion_iou_thr,
                )
            if args.type == "soft_nms":
                bboxes, scores, labels = soft_nms(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=args.weights,
                    iou_thr=args.fusion_iou_thr,
                    mode=2,
                    sigma=0.5,
                )
            elif args.type == "nmw":
                bboxes, scores, labels = non_maximum_weighted(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=args.weights,
                    iou_thr=args.fusion_iou_thr,
                    skip_box_thr=args.skip_box_thr,
                )
            elif args.type == "wbf":
                bboxes, scores, labels = weighted_boxes_fusion(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=args.weights,
                    iou_thr=args.fusion_iou_thr,
                    skip_box_thr=args.skip_box_thr,
                    conf_type=args.conf_type,
                )

            for bbox, score, label in zip(bboxes, scores, labels):
                prediction_string += (
                    str(int(label))
                    + " "
                    + str(score)
                    + " "
                    + str(bbox[0] * image_size)
                    + " "
                    + str(bbox[1] * image_size)
                    + " "
                    + str(bbox[2] * image_size)
                    + " "
                    + str(bbox[3] * image_size)
                    + " "
                )
        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    submission.to_csv(osp.join(out_path, "ouput.csv"), index=False)

    with open(os.path.join(out_path, "log.txt"), "w") as f:
        f.write("Config & Weight List\n")
        for csv_name in args.csv:
            slice_config_name = csv_name.split("/")[-1]
            f.write(f"{slice_config_name}\n")
        f.write("Ensemble Config\n")
        f.write(f"Ensemble Type : {args.type}\n")
        f.write(f"Ensemble weight : {args.weights}\n")
        f.write(f"Ensemble iou thr : {args.fusion_iou_thr}\n")
        f.write(f"Ensemble skip box thr : {args.skip_box_thr}\n")
        f.write(f"Ensemble wbf type : {args.conf_type}\n")


if __name__ == "__main__":
    main()
