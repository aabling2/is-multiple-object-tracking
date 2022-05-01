#!/usr/bin/env python3
import cv2
import numpy as np


class YOLO():
    def __init__(self, model, classes):
        self.input_width = 640
        self.input_height = 640
        self.score_threshold = 0.2
        self.nms_threshold = 0.4
        self.confidence_threshold = 0.4

        # Preset de cores
        self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

        # Carrega classes
        with open(classes, "r") as f:
            self.class_list = [cname.strip() for cname in f.readlines()]

        # Carrega modelo
        self.net = cv2.dnn.readNet(model)
        self._config_net()

    def _config_net(self):
        # Define backend com GPU ou CPU
        active_gpu = cv2.cuda.getCudaEnabledDeviceCount()
        print("GPUs:", active_gpu)
        if active_gpu > 0:
            print("Attempty to use CUDA")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _wrap_detection(self, input_image, output_data):
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / self.input_width
        y_factor = image_height / self.input_height

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.4:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = (
                        row[0].item(), row[1].item(),
                        row[2].item(), row[3].item())
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes

    def _format_yolov5(self, frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def detect(self, frame, draw=True):

        # Faz detecções com yolo
        inputImage = self._format_yolov5(frame)
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0,
            (self.input_width, self.input_height),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward()
        class_ids, confidences, boxes = self._wrap_detection(
            inputImage, outs[0])

        # Desenha detecções
        for (classid, conf, box) in zip(class_ids, confidences, boxes):
            color = self.colors[int(classid) % len(self.colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(
                frame, (box[0], box[1] - 20),
                (box[0] + box[2], box[1]), color, -1)
            cv2.putText(
                frame, self.class_list[classid],
                (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (0, 0, 0))

        return frame
