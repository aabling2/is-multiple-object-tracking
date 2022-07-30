import cv2
import torch
import numpy as np
from .detection import Detection


class torchYOLOv5():
    def __init__(self, model='yolov5s', target_classes=[], thresh_confidence=0.2, nms=True,
                 feature_model=None):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.labels = None
        self.cord = None
        self.model = self._load_model(model)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device Used:", self.device)

        self.target_classes = target_classes
        self.thresh_conf = thresh_confidence
        self.NMS = nms

        # Extrator de features para rastreio
        if feature_model is not None:
            from deep_sort import generate_detections as gdet
            self.encoder = gdet.create_box_encoder(feature_model, batch_size=1)
        else:
            self.encoder = None

    def _load_model(self, model):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', model, pretrained=True)
        return model

    def _selective_objects(self, confidences, labels):
        n = len(confidences)
        indexes = np.arange(0, n)
        thresh_conf = self.thresh_conf
        sel_classes = self.target_classes

        if thresh_conf > 0:
            indexes = indexes[confidences >= thresh_conf]

        if sel_classes != []:
            indexes = np.array([i for i in indexes if labels[i] in sel_classes])

        return indexes

    def detect(self, frame, draw=False):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        results = self.model([frame])

        # Extrai features
        x_shape, y_shape = frame.shape[:2][::-1]
        data = results.xyxyn[0]
        classids = data[:, -1]
        labels = [self.classes[int(i)] for i in classids]
        confidences = np.float32([float(conf) for conf in data[:, -2]])
        boxes = np.int32([
            [int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)]
            for row in data])  # tlbr
        if boxes.size > 0:
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]  # tlwh

        # Filtra objetos pela confiança e classe
        indexes = self._selective_objects(confidences, labels)

        # NMS
        if self.NMS and indexes != []:
            nms_indexes = cv2.dnn.NMSBoxes(boxes[indexes].tolist(), confidences[indexes], 0.25, 0.45)
            indexes = indexes[nms_indexes.ravel()]

        # Formata detecções
        features = self.encoder(frame, boxes[indexes]) if self.encoder else [[-1, -1, -1]]*len(indexes)
        self.detections = [
            Detection(
                id=int(classids[idx]),
                tlwh=boxes[idx],
                confidence=confidences[idx],
                feature=features[i],
                label=labels[idx])
            for i, idx in enumerate(indexes)]

        return self.detections

    # Desenha detecções
    def draw(self, frame, font_scale=0.5):
        for detection in self.detections:
            box = tuple(np.int32(detection.to_tlbr()))
            label = detection.label
            id = detection.id
            np.random.seed(id)
            color = [np.random.randint(0, 255) for _ in range(3)]

            # Bounding box
            cv2.rectangle(frame, box[:2], box[2:4], color, 2)  # bbox

            # Label
            t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=1)[0]
            cv2.rectangle(frame, (box[0]-1, box[1] - 14), (box[0] + t_size[0] + 1, box[1]), color, -1)  # label
            cv2.putText(frame, label, (box[0], box[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0))
