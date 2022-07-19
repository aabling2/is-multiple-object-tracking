import cv2
import numpy as np
from deep_sort import generate_detections as gdet
from deep_sort.tracker import DeepSORT
from re_id.correlation import CrossCorrelationID


class IntelligentSpaceMOT():
    def __init__(self, deep_model=None):

        self.trackers = None  # Rastreadores
        self.reid = CrossCorrelationID()  # ReID multicam
        self.ids = []  # ids multicâmera
        self.labels = []  # labels conforme classes
        self.bboxes = []  # bbox multicâmera
        self.encoder = gdet.create_box_encoder(deep_model, batch_size=1) if deep_model is not None else None

    def _init_mem(self, num_src):
        # Rastreadores
        config_tracking = dict(max_iou_distance=0.7, max_age=30, n_init=3, matching_threshold=0.2)
        self.trackers = [DeepSORT(**config_tracking) for _ in range(num_src)]

    def update(self, frames, detections, reid=False):

        # Inicia rastreadores
        if self.trackers is None:
            self._init_mem(len(frames))

        # Atualiza tracking de cada imagem
        ids = []
        labels = []
        bboxes = []
        for tracker, img, dets in zip(self.trackers, frames, detections):
            # Extração de features para atualizar nos objetos detectados
            if self.encoder is not None:
                features = self.encoder(img, [d.tlwh for d in dets])
                for i, feat in enumerate(features):
                    dets[i].feature = feat

            # Atualiza rastradores
            tracker.update(img, dets)
            ids.append([t.track_id for t in tracker.tracks])
            labels.append([t.label for t in tracker.tracks])
            bboxes.append([np.int32(t.to_tlbr()) for t in tracker.tracks])

        # Atualiza ReID
        self.ids = self.reid.apply(frames, bboxes) if reid else ids
        print("teste", self.ids)

        # Atualiza variáveis
        self.bboxes = bboxes
        self.labels = labels

    def draw(self, frames, detections=[]):

        for frame, all_detections in zip(frames, detections):
            for detection in all_detections:
                box = np.int32(detection.to_tlbr())
                pt1, pt2 = box[0:2], box[2:4]
                cv2.rectangle(frame, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=1)

        # Desenha bboxes de detecção
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        for frame, bboxes, ids, labels in zip(frames, self.bboxes, self.ids, self.labels):
            for box, id, label in zip(bboxes, ids, labels):
                pt1 = box[0:2]
                pt2 = box[2:4]

                # Bounding box
                np.random.seed(id)
                color = [int(x) for x in np.random.randint(0, 255, size=(3, ))]
                cv2.rectangle(frame, pt1=pt1, pt2=pt2, color=color, thickness=2)

                # Label
                id_init = pt2[0]-2-12*len(str(id))
                max_char = int((id_init-pt1[0])/7)
                label = label[:max_char] if max_char > 0 else ''
                cv2.rectangle(frame, (pt1[0]-1, pt1[1]-20), (pt2[0]+1, pt1[1]), color, -1)
                cv2.putText(frame, label[:max_char], (pt1[0], pt1[1]-7), font, font_scale, (0, 0, 0))

                # ID de referência do objeto
                cv2.rectangle(frame, (id_init, pt1[1]-19), (pt2[0]-1, pt1[1]-1), (50, 50, 50), -1)
                cv2.putText(frame, str(id), (id_init+1, pt1[1]-5), font, font_scale*1.3, (255, 255, 255))
