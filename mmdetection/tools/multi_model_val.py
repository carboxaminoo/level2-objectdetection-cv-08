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
    parser.add_argument("inputs", type=str, help="Input image file or folder path.")
    parser.add_argument(
        "config",
        type=str,
        nargs="*",
        help="Config file(s), support receive multiple files",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        help="Checkpoint file(s), support receive multiple files, "
        "remember to correspond to the above config",
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
        "--conf-type",
        type=str,
        default="avg",  # avg, max, box_and_model_avg, absent_model_aware_avg
        help="how to calculate confidence in weighted boxes in wbf",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Output directory of images or prediction results.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--pred-score-thr", type=float, default=0.3, help="bbox score threshold"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Inference batch size."
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the image in a popup window."
    )
    parser.add_argument(
        "--no-save-vis", action="store_true", help="Do not save detection vis results"
    )
    parser.add_argument(
        "--no-save-pred", action="store_true", help="Do not save detection json results"
    )
    parser.add_argument(
        "--palette",
        default="none",
        choices=["coco", "voc", "citys", "random", "none"],
        help="Color palette used for visualization",
    )

    args = parser.parse_args()

    if args.no_save_vis and args.no_save_pred:
        args.out_dir = ""

    return args


def main():
    args = parse_args()

    results = []

    inputs = []
    filename_list = []

    coco = COCO(args.inputs)

    gt_base_list = []
    for imgs in coco.imgs.values():
        gt_instances = {"boxes": [], "labels": []}
        file_name = imgs["file_name"]
        image_path = "/".join(args.inputs.split("/")[:-1]) + "/" + file_name
        img = mmcv.imread(image_path)
        inputs.append(img)

        filename_list.append(file_name)
        image_info = coco.loadImgs(
            coco.getImgIds(imgIds=[int(file_name.split("/")[-1].split(".")[0])])
        )[0]
        annotation_ids = coco.getAnnIds(imgIds=image_info["id"])
        annotations = coco.loadAnns(annotation_ids)
        for annotation in annotations:
            bbox = annotation["bbox"]
            class_id = annotation["category_id"]
            gt_instances["boxes"].append(bbox)
            gt_instances["labels"].append(class_id)

        gt_instances["boxes"] = torch.tensor(gt_instances["boxes"]).to(args.device)
        gt_instances["labels"] = torch.tensor(gt_instances["labels"]).to(args.device)
        gt_base_list.append(gt_instances)

    pred_base_list = []
    for i, (config, checkpoint) in enumerate(tqdm(zip(args.config, args.checkpoints))):
        inferencer = DetInferencer(
            config, checkpoint, device=args.device, palette=args.palette
        )

        result_raw = inferencer(
            inputs=inputs,
            batch_size=args.batch_size,
            no_save_vis=True,
            pred_score_thr=args.pred_score_thr,
        )

        if i == 0:
            results = [
                {"bboxes_list": [], "scores_list": [], "labels_list": []}
                for _ in range(len(result_raw["predictions"]))
            ]

        for res, raw in zip(results, result_raw["predictions"]):
            res["bboxes_list"].append(raw["bboxes"])
            res["scores_list"].append(raw["scores"])
            res["labels_list"].append(raw["labels"])

    # visualizer = VISUALIZERS.build(cfg_visualizer)
    # visualizer.dataset_meta = dataset_meta

    for i in range(len(results)):
        # bboxes, scores, labels = weighted_boxes_fusion(
        #     results[i]["bboxes_list"],
        #     results[i]["scores_list"],
        #     results[i]["labels_list"],
        #     weights=args.weights,
        #     iou_thr=args.fusion_iou_thr,
        #     skip_box_thr=args.skip_box_thr,
        #     conf_type=args.conf_type,
        # )
        bboxes, scores, labels = weighted_boxes_fusion(
            results[i]["bboxes_list"],
            results[i]["scores_list"],
            results[i]["labels_list"],
            # weights=args.weights,
            # iou_thr=args.fusion_iou_thr,
            # skip_box_thr=args.skip_box_thr,
            # conf_type=args.conf_type,
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

    base_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    base_metric50 = MeanAveragePrecision(
        iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
    )
    for pred, gt in zip(pred_base_list, gt_base_list):
        base_metric.update([pred], [gt])
        base_metric50.update([pred], [gt])

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

    now = datetime.now()
    out_path = osp.join(args.out_dir, "result", now.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "log.txt"), "w") as f:
        f.write("Config & Weight List\n")
        for config_name, weight_name in zip(args.config, args.checkpoints):
            slice_config_name = config_name.split("/")[-1]
            slice_weight_name = weight_name.split("/")[-1]
            f.write(f"{slice_config_name} {slice_weight_name}\n")
        f.write("Scores")
        for key, value in score_dict.items():
            f.write(f"{key} : {value}\n")
    print(score_dict)


if __name__ == "__main__":
    main()
