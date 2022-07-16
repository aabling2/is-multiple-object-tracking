import cv2
import numpy as np
from ..deep_sort.tracker import DeepSORT
from ..re_id.correlation import CrossCorrelationID


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

    def update(self, frames, detections, reid=False):

        # Inicia rastreadores
        if self.trackers is None:
            self._init_mem(len(frames))

        # Atualiza tracking
        bboxes = []
        ids = []
        for tracker, img, dets in zip(self.trackers, frames, detections):
            # Atualiza rastradores
            tracker.update(img, dets)
            bboxes.append([np.int32(t.to_tlbr()) for t in tracker.tracks])
            ids.append([t.track_id for t in tracker.tracks])

        # Atualiza ReID
        if reid:
            self.ids = self.reid.apply(frames, bboxes)
        else:
            self.ids = ids

        # Atualiza variáveis
        self.bboxes = bboxes

    def draw(self, frames):
        # Desenha bboxes de detecção
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        for frame, bboxes, ids in zip(frames, self.bboxes, self.ids):
            for box, id in zip(bboxes, ids):
                pt1 = box[0:2]
                pt2 = box[2:4]

                # Bounding box
                np.random.seed(id)
                color = [int(x) for x in np.random.randint(0, 255, size=(3, ))]
                cv2.rectangle(frame, pt1=pt1, pt2=pt2, color=color)

                # Label
                cv2.rectangle(frame, (pt1[0], pt1[1]-20), (pt2[0], pt1[1]), color, -1)
                cv2.putText(frame, str(id), (pt1[0], pt1[1]-10), font, font_scale, (0, 0, 0))

                # ID de referência do objeto
                cv2.rectangle(frame, (pt2[0]-2-12*len(str(id)), pt1[1]-19), (pt2[0]-1, pt1[1]-1), (50, 50, 50), -1)
                cv2.putText(frame, str(id), (pt2[0]-2-12*len(str(id)), pt1[1]-3), font, font_scale*1.2, (255, 255, 255))
