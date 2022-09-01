import cv2
import numpy as np
from bytetrack.tracker import BYTETracker


class MulticamBYTETracker():
    def __init__(self, num_src=1):

        self.trackers = None  # Rastreadores
        self.ids = []  # ids multicâmera
        self.labels = []  # labels conforme classes
        self.bboxes = []  # bbox multicâmera
        self.count_ids = 0

        # Rastreadores
        self.trackers = [
            BYTETracker(
                frame_rate=30, track_thresh=0.5, track_buffer=30,
                match_tresh=0.9, fuse=True, src_id=i)
            for i in range(num_src)]

    def update(self, detections):

        # Atualiza tracking de cada imagem
        ids = []
        labels = []
        bboxes = []
        for tracker, dets in zip(self.trackers, detections):
            tracker.update(dets)
            ids.append([t.track_id for t in tracker.tracks])
            labels.append([t.label for t in tracker.tracks])
            bboxes.append([np.int32(t.tlbr) for t in tracker.tracks])

        # Atualiza variáveis
        self.bboxes = bboxes
        self.labels = labels
        self.ids = ids
        self.count_ids = max(max([max(x) for x in ids]), self.count_ids) if ids else self.count_ids

    def draw(self, frames, detections=[], font_scale=0.5, font=cv2.FONT_HERSHEY_SIMPLEX):

        # Desenha bboxes de rastreio
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        for frame, bboxes, ids, labels in zip(frames, self.bboxes, self.ids, self.labels):
            for box, id, label in zip(bboxes, ids, labels):
                sid = str(id)
                np.random.seed(id)
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

        # Desenha bboxes de detecção
        for frame, all_detections in zip(frames, detections):
            for detection in all_detections:
                box = np.int32(detection.to_tlbr())
                pt1, pt2 = box[0:2], box[2:4]
                cv2.rectangle(frame, pt1=pt1, pt2=pt2, color=(0, 255, 255), thickness=1)
