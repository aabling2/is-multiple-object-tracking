import cv2
import random
import numpy as np


class Detection():
    id: int
    label: str
    score: float
    bbox: list
    velocity: tuple

    @property
    def tlbr(self):
        x, y, w, h = self.bbox.copy()
        return [x, y, x+w, y+h]


class RandomDetector():
    def __init__(self, max_width, max_height, qtd=1, labels=['person', 'car']):
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.labels = labels
        self.curr_id = 0
        self.detections = [self.generate_detection() for _ in range(qtd)]

    def generate_detection(self):
        max_width = self.max_width
        max_height = self.max_height
        labels = self.labels
        self.curr_id += 1

        detection = Detection()
        detection.id = self.curr_id
        detection.label = random.choice(labels)
        detection.score = random.uniform(0.2, 1.0)

        x = random.randint(0, max_width)
        y = random.randint(0, max_height)
        w = random.randint(10, max_width//3)
        h = random.randint(10, max_height//3)
        detection.bbox = [x, y, w, h]

        detection.velocity = (random.randint(-10, 10), random.randint(1, +10))

        return detection

    def update(self):
        detections = self.detections
        for i, det in enumerate(detections):
            x, y, w, h = det.bbox
            vx, vy = det.velocity
            x += vx
            y += vy
            if x < 0 or x+w >= self.max_width or y < 0 or y+h >= self.max_height:
                self.detections.pop(0)
                self.detections.append(self.generate_detection())
            else:
                self.detections[i].bbox = [x, y, w, h]

        return self.detections

    def draw(self, frame, font_scale=0.5, font=cv2.FONT_HERSHEY_SIMPLEX):
        # Desenha bboxes de rastreio
        for detection in self.detections:
            sid = str(detection.id)
            label = detection.label
            box = detection.tlbr
            np.random.seed(detection.id)
            color = [np.random.randint(0, 255) for _ in range(3)]
            label_size = cv2.getTextSize(label+" ", 0, fontScale=font_scale, thickness=1)[0]
            id_size = cv2.getTextSize(sid, 0, fontScale=font_scale*1.3, thickness=1)[0]

            # Bounding box
            cv2.rectangle(frame, box[:2], box[2:4], color, 2)

            # Label
            pt1, pt2 = (box[0]-1, box[1] + 20), (box[0] + label_size[0] + id_size[0] + 1, box[1])
            cv2.rectangle(frame, pt1, pt2, color, -1)
            cv2.putText(frame, label, (box[0], box[1] + 8), font, font_scale, (0, 0, 0))

            # ID de referência do objeto
            pt1, pt2 = (box[0] + label_size[0] + 1, box[1] + 19), (box[0] + label_size[0] + id_size[0], box[1] - 1)
            cv2.rectangle(frame, pt1, pt2, (50, 50, 50), -1)
            cv2.putText(frame, sid, (pt1[0], pt2[1]+15), font, font_scale*1.3, (255, 255, 255))
