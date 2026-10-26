"""
AeroGuardian NanoDet ONNX Neural Detection Engine.
Executes lightweight ONNX neural models directly on CPU / Arm64 NEON via cv2.dnn.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball"
]


class NanoDetDetector:
    """
    OpenCV 5.0 native DNN NanoDet ONNX detector for aerial search-and-rescue.
    """

    def __init__(
        self,
        model_path: str = "models/nanodet.onnx",
        conf_threshold: float = 0.18,
        nms_threshold: float = 0.50,
        input_size: Tuple[int, int] = (416, 416),
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.has_dnn = False
        self.net: Optional[cv2.dnn.Net] = None

        self.strides = (8, 16, 32)
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.anchors_mlvl: List[np.ndarray] = []

        self._init_anchors()
        self._load_dnn()

    def _init_anchors(self):
        for stride in self.strides:
            feat_h = int(self.input_size[0] / stride)
            feat_w = int(self.input_size[1] / stride)
            shift_x = np.arange(0, feat_w) * stride
            shift_y = np.arange(0, feat_h) * stride
            xv, yv = np.meshgrid(shift_x, shift_y)
            cx = xv.flatten() + 0.5 * (stride - 1)
            cy = yv.flatten() + 0.5 * (stride - 1)
            self.anchors_mlvl.append(np.column_stack((cx, cy)))

    def _load_dnn(self):
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNet(self.model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.has_dnn = True
            except Exception as e:
                print(f"[NanoDetDetector] Warning: Could not initialize cv2.dnn: {e}")
                self.has_dnn = False

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs native OpenCV 5 DNN detection on the optical frame.
        """
        if not self.has_dnn or self.net is None:
            return []

        orig_h, orig_w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, self.input_size)
        img_norm = resized.astype(np.float32)
        img_norm = (img_norm - self.mean) / self.std

        blob = cv2.dnn.blobFromImage(img_norm)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        cls_scores = outs[:3]
        bbox_preds = outs[3:]
        bboxes_mlvl, scores_mlvl = [], []

        for stride, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.anchors_mlvl):
            if cls_score.ndim == 3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim == 3:
                bbox_pred = bbox_pred.squeeze(axis=0)

            x_exp = np.exp(bbox_pred.reshape(-1, self.reg_max + 1))
            x_sum = np.sum(x_exp, axis=1, keepdims=True)
            bbox_norm = x_exp / np.maximum(x_sum, 1e-6)
            bbox_pred = np.dot(bbox_norm, self.project).reshape(-1, 4) * stride

            points = anchors
            x1 = points[:, 0] - bbox_pred[:, 0]
            y1 = points[:, 1] - bbox_pred[:, 1]
            x2 = points[:, 0] + bbox_pred[:, 2]
            y2 = points[:, 1] + bbox_pred[:, 3]

            bboxes = np.column_stack([x1, y1, x2, y2])
            bboxes_mlvl.append(bboxes)
            scores_mlvl.append(cls_score)

        bboxes_all = np.concatenate(bboxes_mlvl, axis=0)
        scores_all = np.concatenate(scores_mlvl, axis=0)

        scale_x = orig_w / self.input_size[1]
        scale_y = orig_h / self.input_size[0]

        bboxes_wh = bboxes_all.copy()
        bboxes_wh[:, 2] = (bboxes_wh[:, 2] - bboxes_wh[:, 0]) * scale_x
        bboxes_wh[:, 3] = (bboxes_wh[:, 3] - bboxes_wh[:, 1]) * scale_y
        bboxes_wh[:, 0] = bboxes_wh[:, 0] * scale_x
        bboxes_wh[:, 1] = bboxes_wh[:, 1] * scale_y

        class_ids = np.argmax(scores_all, axis=1)
        confidences = np.max(scores_all, axis=1)

        indices = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )

        rescue_classes = {"person", "car", "truck", "bus", "motorcycle"}
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                cid = int(class_ids[idx])
                cls_name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
                if cls_name not in rescue_classes:
                    continue
                conf = float(confidences[idx])
                if cls_name in ("car", "truck", "bus", "motorcycle") and conf < 0.42:
                    continue

                box = [int(v) for v in bboxes_wh[idx]]
                if cls_name == "person":
                    bx, by, bw, bh = box
                    if bw < 14 or bh < 24:
                        cx, cy = bx + bw // 2, by + bh // 2
                        bw = max(bw, 14)
                        bh = max(bh, 24)
                        box = [max(0, cx - bw // 2), max(0, cy - bh // 2), bw, bh]

                detections.append({
                    "class": "human" if cls_name == "person" else cls_name,
                    "confidence": conf,
                    "bbox": box,
                    "source": "opencv5_dnn_nanodet"
                })

        suppressed_dets = []
        sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        for d1 in sorted_dets:
            x1, y1, w1, h1 = d1["bbox"]
            a1 = w1 * h1
            nested = False
            for d2 in suppressed_dets:
                if d1["class"] == d2["class"]:
                    x2, y2, w2, h2 = d2["bbox"]
                    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    inter = ix * iy
                    union = a1 + (w2 * h2) - inter
                    iou = inter / max(union, 1.0)
                    containment = inter / max(a1, 1.0)
                    if iou >= 0.30 or containment >= 0.45 or (d1["class"] in ("car", "truck", "bus") and d1["confidence"] < 0.51 and inter > 50):
                        nested = True
                        break
            if not nested:
                suppressed_dets.append(d1)

        return suppressed_dets
