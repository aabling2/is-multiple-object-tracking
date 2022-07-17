import cv2
import numpy as np
from detection import Detection


class cvYOLO():
    def __init__(self, model, classes, gpu=False):
        self.input_width = 640
        self.input_height = 640
        self.score_threshold = 0.2
        self.nms_threshold = 0.4
        self.confidence_threshold = 0.4
        self.detections = []
        self.classids = []
        self.gpu = gpu

        # Preset de cores
        self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

        # Carrega classes
        with open(classes, "r") as f:
            self.class_list = [cname.strip() for cname in f.readlines()]

        # Carrega modelo
        self.net = cv2.dnn.readNetFromONNX(model)
        self._config_net()

    def _config_net(self):
        # Define backend com GPU ou CPU
        active_gpu = cv2.cuda.getCudaEnabledDeviceCount()
        print("GPUs:", active_gpu)
        if active_gpu > 0 and self.gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Running on GPU")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Running on CPU")

    def _wrap_detection(self, output_data, shape):
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]

        image_width, image_height = shape[:2][::-1]

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
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.input_width, self.input_height), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward()
        class_ids, confidences, boxes = self._wrap_detection(outs[0], frame.shape)

        conf = 1.
        feature = [-1, -1, -1]
        self.classids = class_ids
        self.detections = [
            Detection(id, box, conf, feature, self.class_list[id])
            for box, id in zip(boxes, class_ids)]

        if draw:
            self.draw(frame)

        return self.detections

    # Desenha detecções
    def draw(self, frame):
        for detection in self.detections:
            box = np.int32(detection.to_tlbr())
            label = detection.label
            id = detection.id
            np.random.seed(id)
            color = [int(x) for x in np.random.randint(0, 255, size=(3, ))]
            cv2.rectangle(frame, box[:2], box[2:4], color, 2)
            cv2.rectangle(frame, (box[0]-1, box[1] - 14), (box[2]+1, box[1]), color, -1)
            cv2.putText(frame, label, (box[0], box[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
