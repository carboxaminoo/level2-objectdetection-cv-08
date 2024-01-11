import torch
import numpy as np
import torchvision
from torchmetrics.detection import MeanAveragePrecision


class RecycleMetric:
    def __init__(self) -> None:
        self.class_num = 10
        # self.ap50_class_list = [[] for _ in range(self.class_num)]

        # self.ap50_bbox_class_list = [[] for _ in range(self.bbox_class_num)]
        # self.epsil = np.spacing(1)

        # min, boundary ~~~ , max
        self.bbox_size_boundary = [0, 64, 128, 256, 512, 1048577]
        self.bbox_size_class_num = len(self.bbox_size_boundary) - 2

        self.bbox_count_boundary = [0, 10, 20, 30, 50, 99]
        self.bbox_count_class_num = len(self.bbox_count_boundary) - 2

        self.gt_dict = []
        self.predict_base_list = []
        self.predict_bbox_size_dict = [[] for _ in range(self.bbox_size_class_num)]
        self.predict_bbox_count_dict = [[] for _ in range(self.bbox_count_class_num)]

    def clear_init(self) -> None:
        self.ap50_class_list = [[] for _ in range(self.class_num)]
        self.ap50_bbox_class_list = [[] for _ in range(self.bbox_size_class_num)]

        self.gt_dict = []
        self.predict_base_list = []
        self.predict_bbox_size_dict = [[] for _ in range(self.bbox_size_class_num)]
        self.predict_bbox_count_dict = [[] for _ in range(self.bbox_count_class_num)]

    def save_coco_class_data(self, outputs, data_batch):
        """
        outputs과 gt를 bbox와 label를 분리하는 함수
        args :
            outputs : model에 나오는 값들과 각종 실험들에 중요한 값
            data_batch : input에 나오는 값

        return :
            pred_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
            gt_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
        """

        gt_batch_bboxs = outputs[0].gt_instances["bboxes"]
        gt_batch_labels = outputs[0].gt_instances["labels"]

        pred_batch_bboxs = outputs[0]._pred_instances["bboxes"]
        pred_batch_labels = outputs[0]._pred_instances["labels"]
        pred_batch_scores = outputs[0]._pred_instances["scores"]

        # self.gt_dict['boxes'].append(gt_batch_bboxs)
        # self.gt_dict['labels'].append(torch.tensor(gt_batch_labels).to(gt_batch_bboxs.device))

        # self.predict_base_dict['boxes'].append(pred_batch_bboxs)
        # self.predict_base_dict['labels'].append(torch.tensor(pred_batch_labels).to(gt_batch_bboxs.device))
        # self.predict_base_dict['scores'].append(torch.tensor(pred_batch_scores).to(gt_batch_bboxs.device))

        gt_batch_dict = {
            "boxes": gt_batch_bboxs,
            "labels": torch.tensor(gt_batch_labels).to(gt_batch_bboxs.device),
        }
        pred_batch_dict = {
            "boxes": pred_batch_bboxs,
            "labels": torch.tensor(pred_batch_labels).to(gt_batch_bboxs.device),
            "scores": torch.tensor(pred_batch_scores).to(gt_batch_bboxs.device),
        }
        self.predict_base_list.append([gt_batch_dict, pred_batch_dict])

    def save_bbox_size_class_data(self, outputs, data_batch):
        """
        outputs과 gt를 bbox와 label를 분리하는 함수
        args :
            outputs : model에 나오는 값들과 각종 실험들에 중요한 값
            data_batch : input에 나오는 값

        return :
            pred_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
            gt_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
        """

        size_gt_batch_bboxs = [[] for _ in range(self.bbox_size_class_num)]
        size_gt_batch_labels = [[] for _ in range(self.bbox_size_class_num)]

        gt_batch_bboxs = outputs[0].gt_instances["bboxes"]
        gt_batch_labels = outputs[0].gt_instances["labels"]

        pred_batch_bboxs = outputs[0]._pred_instances["bboxes"]
        pred_batch_labels = outputs[0]._pred_instances["labels"]
        pred_batch_scores = outputs[0]._pred_instances["scores"]

        for gt_bbox, gt_label in zip(gt_batch_bboxs, gt_batch_labels):
            gt_area = abs((gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]))
            for idx in range(self.bbox_size_class_num + 1):
                if (
                    gt_area > self.bbox_size_boundary[idx]
                    and gt_area <= self.bbox_size_boundary[idx + 1]
                ):
                    size_gt_batch_bboxs[idx].append(gt_bbox)
                    size_gt_batch_labels[idx].append(gt_label)

        pred_batch_dict = {
            "boxes": pred_batch_bboxs,
            "labels": torch.tensor(pred_batch_labels).to(gt_batch_bboxs.device),
            "scores": torch.tensor(pred_batch_scores).to(gt_batch_bboxs.device),
        }
        for idx, (gt_bbox, gt_label) in enumerate(
            zip(size_gt_batch_bboxs, size_gt_batch_labels)
        ):
            if gt_bbox:
                gt_batch_dict = {
                    "boxes": torch.stack(gt_bbox).to(gt_batch_bboxs.device),
                    "labels": torch.stack(gt_label).to(gt_batch_bboxs.device),
                }
                self.predict_bbox_size_dict[idx].append(
                    [gt_batch_dict, pred_batch_dict]
                )

    def save_bbox_count_class_data(self, outputs, data_batch):
        """
        outputs과 gt를 bbox와 label를 분리하는 함수
        args :
            outputs : model에 나오는 값들과 각종 실험들에 중요한 값
            data_batch : input에 나오는 값

        return :
            pred_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
            gt_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
        """
        count_gt_batch_bboxs = [[] for _ in range(self.bbox_count_class_num)]
        count_gt_batch_labels = [[] for _ in range(self.bbox_count_class_num)]

        gt_batch_bboxs = outputs[0].gt_instances["bboxes"]
        gt_batch_labels = outputs[0].gt_instances["labels"]

        pred_batch_bboxs = outputs[0]._pred_instances["bboxes"]
        pred_batch_labels = outputs[0]._pred_instances["labels"]
        pred_batch_scores = outputs[0]._pred_instances["scores"]

        image_bbox_count = len(outputs[0].gt_instances["labels"])

        for gt_bbox, gt_label in zip(gt_batch_bboxs, gt_batch_labels):
            for idx in range(self.bbox_count_class_num + 1):
                if (
                    image_bbox_count > self.bbox_count_boundary[idx]
                    and image_bbox_count <= self.bbox_count_boundary[idx + 1]
                ):
                    count_gt_batch_bboxs[idx].append(gt_bbox)
                    count_gt_batch_labels[idx].append(gt_label)

        pred_batch_dict = {
            "boxes": pred_batch_bboxs,
            "labels": torch.tensor(pred_batch_labels).to(gt_batch_bboxs.device),
            "scores": torch.tensor(pred_batch_scores).to(gt_batch_bboxs.device),
        }
        for idx, (gt_bbox, gt_label) in enumerate(
            zip(count_gt_batch_bboxs, count_gt_batch_labels)
        ):
            if gt_bbox:
                gt_batch_dict = {
                    "boxes": torch.stack(gt_bbox).to(gt_batch_bboxs.device),
                    "labels": torch.stack(gt_label).to(gt_batch_bboxs.device),
                }
                self.predict_bbox_count_dict[idx].append(
                    [gt_batch_dict, pred_batch_dict]
                )

    # def calculate_ap_category(
    #     self, pred_batch_dict, gt_batch_dict, labels, iou_threshold
    # ):
    #     """
    #     카테고리에 따라 분류되는 데이터에 대한 각각 카테고리 AP를 구하는 함수

    #     args :
    #         pred_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #         gt_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #         iou_thresold : map 몇으로 구할 것인지 정하는 인자 값

    #     return :
    #         클래스별로 AP 값 리스트
    #     """
    #     return self.calculate_map(pred_batch_dict, gt_batch_dict, labels, iou_threshold)

    # def calculate_ap50_class(self, outputs, data_batch):
    #     """
    #     클래스별 AP50을 구하는 함수
    #     args :
    #         outputs : model에 나오는 값들과 각종 실험들에 중요한 값
    #         data_batch : input에 나오는 값

    #     return :
    #         None
    #     """
    #     labels = [i for i in range(self.class_num)]

    #     pred_batch_dict, gt_batch_dict = self.split_class_data(outputs, data_batch)
    #     class_aps, class_exists = self.calculate_ap_category(
    #         pred_batch_dict, gt_batch_dict, labels, 0.5
    #     )

    #     pred_torch_batch_dict, gt_torch_batch_dict = self.split_bbox_torch_class_data(
    #         outputs, data_batch
    #     )
    #     metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    #     gt_torch_batch_dict["labels"] = torch.tensor(gt_torch_batch_dict["labels"])
    #     pred_torch_batch_dict["labels"] = torch.tensor(pred_torch_batch_dict["labels"])
    #     pred_torch_batch_dict["scores"] = torch.tensor(pred_torch_batch_dict["scores"])

    #     metric.update([pred_torch_batch_dict], [gt_torch_batch_dict])
    #     class_torch_map = metric.compute()

    #     for idx, (class_ap, class_exist) in enumerate(zip(class_aps, class_exists)):
    #         if class_exist:
    #             self.ap50_class_list[idx].append(class_ap)

    # def calculate_ap50_bbox_class(self, outputs, data_batch):
    #     """
    #     클래스별 AP50을 구하는 함수
    #     args :
    #         outputs : model에 나오는 값들과 각종 실험들에 중요한 값
    #         data_batch : input에 나오는 값

    #     return :
    #         None
    #     """
    #     labels = [i for i in range(self.bbox_class_num)]

    #     pred_batch_dict, gt_batch_dict = self.split_bbox_class_data(outputs, data_batch)
    #     class_aps, class_exists = self.calculate_ap_category(
    #         pred_batch_dict, gt_batch_dict, labels, 0.5
    #     )
    #     for idx, (class_ap, class_exist) in enumerate(zip(class_aps, class_exists)):
    #         if class_exist:
    #             self.ap50_bbox_class_list[idx].append(class_ap)

    # def calculate_map50(self, outputs, data_batch):
    #     """
    #     mAP를 구하는 함수
    #     args :
    #         outputs : model에 나오는 값들과 각종 실험들에 중요한 값
    #         data_batch : input에 나오는 값

    #     return :
    #         None
    #     """
    #     labels_num = 10
    #     labels = [i for i in range(labels_num)]

    #     pred_batch_dict, gt_batch_dict, labels = self.split_class_data(
    #         outputs, data_batch
    #     )
    #     return self.calculate_ap_category(
    #         self, pred_batch_dict, gt_batch_dict, labels, 0.5
    #     )
    # def calculate_iou(self, box1, box2):
    #     """Calculate the IoU of two bounding boxes"""
    #     # Get the coordinates of bounding boxes
    #     inter_rect_x1 = torch.max(box1[0], box2[0])
    #     inter_rect_y1 = torch.max(box1[1], box2[1])
    #     inter_rect_x2 = torch.min(box1[2], box2[2])
    #     inter_rect_y2 = torch.min(box1[3], box2[3])

    #     # Intersection area
    #     inter_area = torch.clamp(
    #         inter_rect_x2 - inter_rect_x1 + self.epsil, min=0
    #     ) * torch.clamp(inter_rect_y2 - inter_rect_y1 + self.epsil, min=0)

    #     # Union Area
    #     b1_area = (box1[2] - box1[0] + self.epsil) * (box1[3] - box1[1] + self.epsil)
    #     b2_area = (box2[2] - box2[0] + self.epsil) * (box2[3] - box2[1] + self.epsil)

    #     iou = inter_area / (b1_area + b2_area - inter_area)
    #     return iou

    # def get_sorted_detections(self, detections):
    #     """Sort detections by decreasing confidence scores"""
    #     return sorted(detections, key=lambda x: x["scores"], reverse=True)

    # def calculate_precision_recall_per_class(
    #     self, detections, ground_truths, iou_threshold
    # ):
    #     """Calculate precision and recall for a specific class"""
    #     # Sort detections by score
    #     sorted_detections = self.get_sorted_detections(detections)

    #     # Lists to store true positives and false positives
    #     tp = torch.zeros(len(sorted_detections))
    #     fp = torch.zeros(len(sorted_detections))

    #     # Used for IoU matching
    #     detected = []

    #     for i, detection in enumerate(sorted_detections):
    #         max_iou = 0
    #         max_gt_idx = -1

    #         for gt_idx, gt in enumerate(ground_truths):
    #             if gt["labels"] == detection["labels"]:
    #                 # iou = self.calculate_iou(torch.tensor(detection['boxes']), torch.tensor(gt['boxes']))
    #                 iou = torchvision.ops.box_iou(
    #                     boxes1=torch.tensor((detection["boxes"])).unsqueeze(0),
    #                     boxes2=torch.tensor(gt["boxes"]).unsqueeze(0),
    #                 )

    #                 if iou > max_iou:
    #                     max_iou = iou
    #                     max_gt_idx = gt_idx

    #         if max_iou >= iou_threshold:
    #             if max_gt_idx not in detected:
    #                 tp[i] = 1
    #                 detected.append(max_gt_idx)
    #             else:
    #                 fp[i] = 1
    #         else:
    #             fp[i] = 1

    #     # Calculate cumulative precision and recall
    #     tp_cumsum = torch.cumsum(tp, dim=0)
    #     fp_cumsum = torch.cumsum(fp, dim=0)

    #     precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.spacing(1))
    #     recalls = tp_cumsum / (len(ground_truths) + np.spacing(1))

    #     return precisions, recalls

    # def calculate_average_precision(self, precisions, recalls):
    #     """Calculate the average precision for a given precision-recall curve"""
    #     # Append start and end points to precision and recall
    #     precisions = torch.cat((torch.tensor([1]), precisions, torch.tensor([0])))
    #     recalls = torch.cat((torch.tensor([0]), recalls, torch.tensor([1])))

    #     # Calculate the area under the curve
    #     ap = torch.trapz(precisions, recalls)
    #     return ap.item()

    # def calculate_map(self, detections, ground_truths, class_labels, iou_threshold=0.5):
    #     """Calculate the mean Average Precision (mAP) for all classes"""
    #     aps = [0 for _ in range(len(class_labels))]
    #     exist = [False for _ in range(len(class_labels))]

    #     for label in class_labels:
    #         class_detections = [d for d in detections if d["labels"] == label]
    #         class_ground_truths = [gt for gt in ground_truths if gt["labels"] == label]
    #         if class_ground_truths:
    #             precisions, recalls = self.calculate_precision_recall_per_class(
    #                 class_detections, class_ground_truths, iou_threshold
    #             )
    #             ap = self.calculate_average_precision(precisions, recalls)
    #             aps[label] = ap
    #             exist[label] = True

    #     return aps, exist

    # def split_class_data(self, outputs, data_batch):
    #     """
    #     outputs과 gt를 bbox와 label를 분리하는 함수
    #     args :
    #         outputs : model에 나오는 값들과 각종 실험들에 중요한 값
    #         data_batch : input에 나오는 값

    #     return :
    #         pred_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #         gt_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #     """
    #     pred_batch_dict = []
    #     gt_batch_dict = []
    #     gt_batch_bboxs = outputs[0].gt_instances["bboxes"]
    #     gt_batch_labels = outputs[0].gt_instances["labels"]

    #     pred_batch_bboxs = outputs[0]._pred_instances["bboxes"]
    #     pred_batch_labels = outputs[0]._pred_instances["labels"]
    #     pred_batch_scores = outputs[0]._pred_instances["scores"]

    #     for gt_batch_bbox, gt_batch_label in zip(gt_batch_bboxs, gt_batch_labels):
    #         gt_batch_dict.append({"boxes": gt_batch_bbox, "labels": gt_batch_label})

    #     for pred_batch_bbox, pred_batch_label, pred_batch_score in zip(
    #         pred_batch_bboxs, pred_batch_labels, pred_batch_scores
    #     ):
    #         pred_batch_dict.append(
    #             {
    #                 "boxes": pred_batch_bbox,
    #                 "labels": pred_batch_label,
    #                 "scores": pred_batch_score,
    #             }
    #         )
    #     return pred_batch_dict, gt_batch_dict

    # def split_bbox_class_data(self, outputs, data_batch):
    #     """
    #     outputs과 gt를 bbox와 label를 분리하는 함수
    #     args :
    #         outputs : model에 나오는 값들과 각종 실험들에 중요한 값
    #         data_batch : input에 나오는 값

    #     return :
    #         pred_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #         gt_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #     """
    #     gt_batch_dict = []
    #     pred_batch_dict = []

    #     gt_batch_labels = []
    #     pred_batch_labels = []

    #     gt_batch_bboxs = outputs[0].gt_instances["bboxes"]

    #     pred_batch_bboxs = outputs[0]._pred_instances["bboxes"]
    #     pred_batch_scores = outputs[0]._pred_instances["scores"]

    #     for gt_bbox in gt_batch_bboxs:
    #         gt_area = abs((gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]))
    #         if gt_area <= 4096:
    #             gt_batch_labels.append(0)
    #         elif gt_area > 4096 and gt_area <= 262144:
    #             gt_batch_labels.append(1)
    #         elif gt_area > 262144:
    #             gt_batch_labels.append(2)
    #         else:
    #             exit()
    #     for pred_bbox in pred_batch_bboxs:
    #         pred_area = abs(
    #             (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    #         )
    #         if pred_area <= 4096:
    #             pred_batch_labels.append(0)
    #         elif pred_area > 4096 and pred_area <= 262144:
    #             pred_batch_labels.append(1)
    #         elif pred_area > 262144:
    #             pred_batch_labels.append(2)
    #         else:
    #             exit()

    #     for gt_batch_bbox, gt_batch_label in zip(gt_batch_bboxs, gt_batch_labels):
    #         gt_batch_dict.append({"boxes": gt_batch_bbox, "labels": gt_batch_label})

    #     for pred_batch_bbox, pred_batch_label, pred_batch_score in zip(
    #         pred_batch_bboxs, pred_batch_labels, pred_batch_scores
    #     ):
    #         pred_batch_dict.append(
    #             {
    #                 "boxes": pred_batch_bbox,
    #                 "labels": pred_batch_label,
    #                 "scores": pred_batch_score,
    #             }
    #         )
    #     return pred_batch_dict, gt_batch_dict

    # def split_torch_class_data(self, outputs, data_batch):
    #     """
    #     outputs과 gt를 bbox와 label를 분리하는 함수
    #     args :
    #         outputs : model에 나오는 값들과 각종 실험들에 중요한 값
    #         data_batch : input에 나오는 값

    #     return :
    #         pred_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #         gt_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #     """
    #     pred_batch_dict = []
    #     gt_batch_dict = []
    #     gt_batch_bboxs = outputs[0].gt_instances["bboxes"]
    #     gt_batch_labels = outputs[0].gt_instances["labels"]

    #     pred_batch_bboxs = outputs[0]._pred_instances["bboxes"]
    #     pred_batch_labels = outputs[0]._pred_instances["labels"]
    #     pred_batch_scores = outputs[0]._pred_instances["scores"]

    #     gt_batch_dict = {
    #         "boxes": gt_batch_bboxs,
    #         "labels": torch.tensor(gt_batch_labels),
    #     }
    #     pred_batch_dict = {
    #         "boxes": pred_batch_bboxs,
    #         "labels": torch.tensor(pred_batch_labels),
    #         "scores": torch.tensor(pred_batch_scores),
    #     }
    #     return pred_batch_dict, gt_batch_dict

    # def split_bbox_torch_class_data(self, outputs, data_batch):
    #     """
    #     outputs과 gt를 bbox와 label를 분리하는 함수
    #     args :
    #         outputs : model에 나오는 값들과 각종 실험들에 중요한 값
    #         data_batch : input에 나오는 값

    #     return :
    #         pred_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #         gt_batch_dict : {'bbox': [x1, y1, x2, y2], 'label': label}
    #     """
    #     gt_batch_dict = []
    #     pred_batch_dict = []

    #     gt_batch_labels = []
    #     pred_batch_labels = []

    #     gt_batch_bboxs = outputs[0].gt_instances["bboxes"]

    #     pred_batch_bboxs = outputs[0]._pred_instances["bboxes"]
    #     pred_batch_scores = outputs[0]._pred_instances["scores"]

    #     for gt_bbox in gt_batch_bboxs:
    #         gt_area = abs((gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]))
    #         if gt_area <= 4096:
    #             gt_batch_labels.append(0)
    #         elif gt_area > 4096 and gt_area <= 262144:
    #             gt_batch_labels.append(1)
    #         elif gt_area > 262144:
    #             gt_batch_labels.append(2)
    #         else:
    #             exit()
    #     for pred_bbox in pred_batch_bboxs:
    #         pred_area = abs(
    #             (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    #         )
    #         if pred_area <= 4096:
    #             pred_batch_labels.append(0)
    #         elif pred_area > 4096 and pred_area <= 262144:
    #             pred_batch_labels.append(1)
    #         elif pred_area > 262144:
    #             pred_batch_labels.append(2)
    #         else:
    #             exit()

    #     gt_batch_dict = {
    #         "boxes": gt_batch_bboxs,
    #         "labels": torch.tensor(gt_batch_labels),
    #     }
    #     pred_batch_dict = {
    #         "boxes": pred_batch_bboxs,
    #         "labels": torch.tensor(pred_batch_labels),
    #         "scores": torch.tensor(pred_batch_scores),
    #     }
    #     return pred_batch_dict, gt_batch_dict
