import cv2
import numpy as np
from deep_sort.tracker import DeepSORT
from re_id.correlation import CrossCorrelationID


class IntelligentSpaceMOT():
    def __init__(self):

        self.trackers = None  # Rastreadores
        self.reid = CrossCorrelationID()  # ReID multicam
        self.ids = []  # ids multicâmera
        self.bboxes = []  # bbox multicâmera

    def _init_mem(self, num_src):
        # Rastreadores
        config_tracking = dict(max_iou_distance=0.7, max_age=30, n_init=3, matching_threshold=0.2)
        self.trackers = [DeepSORT(**config_tracking) for _ in range(num_src)]

    def update(self, frames, detections):

        # Inicia rastreadores
        if self.trackers is None:
            self._init_mem(len(frames))

        # Atualiza tracking
        bboxes = []
        for tracker, img, dets in zip(self.trackers, frames, detections):
            # Atualiza rastradores
            tracker.update(img, dets)
            bboxes.append([np.int32(t.to_tlbr()) for t in tracker.tracks])

        # Atualiza ReID
        self.ids = self.reid.apply(frames, bboxes)

        # Atualiza variáveis
        self.bboxes = bboxes

    def draw(self, frames):
        # Desenha bboxes de detecção
        j = 0
        for i, bboxes in enumerate(self.bboxes):
            for box in bboxes:
                pt1 = box[0:2]
                pt2 = box[2:4]
                label = self.ids[j] if len(self.ids) > j else j
                np.random.seed(label)
                color = [int(x) for x in np.random.randint(0, 255, size=(3, ))]
                # cv2.putText(images[i], str(j), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frames[i], str(label), pt2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frames[i], pt1, pt2, color=color, thickness=2)
                j += 1
