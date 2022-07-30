import cv2
import numpy as np
from deep_sort.tracker import DeepSORT


class IntelligentSpaceMOT():
    def __init__(self, deep_model=None, reid=False, deep=False):

        self.trackers = None  # Rastreadores
        self.ids = []  # ids multicâmera
        self.labels = []  # labels conforme classes
        self.bboxes = []  # bbox multicâmera
        self.reid = None if reid is False else True

        if str(deep_model) != 'None' and deep:
            print("Tracking DeepSORT with deep part.")
            from deep_sort import generate_detections as gdet
            self.encoder = gdet.create_box_encoder(deep_model, batch_size=1)
        else:
            self.encoder = None

    def _init_mem(self, num_src):
        # Inicia objeto de reidentificação
        if self.reid is True:
            from re_id.correlation import CrossCorrelationID
            self.reid = CrossCorrelationID(threshold=0.0, qtd=num_src)  # ReID multicam

        # Rastreadores com ReID embutido
        self.trackers = [
            DeepSORT(max_iou_distance=0.5, max_age=30, n_init=1, matching_threshold=0.2, reid=self.reid, ref=i)
            for i in range(num_src)]

    def update(self, frames, detections):

        # Inicia rastreadores
        if self.trackers is None:
            self._init_mem(len(frames))

        # Atualização global reid
        if self.reid is not None:
            self.reid.update_global(frames, self.trackers)

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

            # Atualiza rastreadores
            tracker.update(dets)
            ids.append([t.track_id for t in tracker.tracks])
            labels.append([t.label for t in tracker.tracks])
            bboxes.append([np.int32(t.to_tlbr()) for t in tracker.tracks])

        # Atualiza variáveis
        self.bboxes = bboxes
        self.labels = labels
        self.ids = ids

    def draw(self, frames, detections=[], font_scale=0.5, font=cv2.FONT_HERSHEY_SIMPLEX):

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
                sid = str(id)
                np.random.seed(id)
                color = [np.random.randint(0, 255) for _ in range(3)]
                label_size = cv2.getTextSize(label+" ", 0, fontScale=font_scale, thickness=1)[0]
                id_size = cv2.getTextSize(sid, 0, fontScale=font_scale*1.3, thickness=1)[0]

                # Bounding box
                cv2.rectangle(frame, box[:2], box[2:4], color, 2)

                # Label
                pt1, pt2 = (box[0]-1, box[1] - 20), (box[0] + label_size[0] + id_size[0] + 1, box[1])
                cv2.rectangle(frame, pt1, pt2, color, -1)
                cv2.putText(frame, label, (box[0], box[1] - 7), font, font_scale, (0, 0, 0))

                # ID de referência do objeto
                pt1, pt2 = (box[0] + label_size[0] + 1, box[1] - 19), (box[0] + label_size[0] + id_size[0], box[1] - 1)
                cv2.rectangle(frame, pt1, pt2, (50, 50, 50), -1)
                cv2.putText(frame, sid, (pt1[0], pt2[1]-2), font, font_scale*1.3, (255, 255, 255))
