import cv2
import numpy as np
from mot_pis_bytetrack.bytetrack.tracker import BYTETracker


class MulticamBYTETracker():
    def __init__(self, max_src=-1, refs=[1, 2, 3, 4]):

        self.max_src = max_src
        self.trackers = None  # Rastreadores
        self.ids = []  # ids multicâmera
        self.labels = []  # labels conforme classes
        self.bboxes = []  # bbox multicâmera
        self.trackers = []
        self.refs = []

        # Aloca rastreadores pré definidos
        for ref in refs:
            self._create_tracker(ref)

    def _create_tracker(self, ref):
        if len(self.trackers) < self.max_src or self.max_src == -1:
            self.refs.append(ref)
            self.ids.append([])
            self.labels.append([])
            self.bboxes.append([])
            self.trackers.append(BYTETracker(
                frame_rate=30, track_thresh=0.5, track_buffer=30,
                match_tresh=0.9, fuse=True, src_id=ref))

    def update(self, detections, refs=[]):

        refs = self.refs if refs == [] else refs

        # Atualiza tracking de cada imagem
        for ref in refs:
            # Cria rastreador se não existir nas referências
            if ref not in self.refs:
                self._create_tracker(ref)

            # Atualiza tracker
            if ref in self.refs:
                idx = self.refs.index(ref)
                tracker = self.trackers[idx]
                tracker.update(detections[idx])

                # Atualiza variáveis
                self.ids[idx] = [t.track_id for t in tracker.tracks]
                self.labels[idx] = [t.label for t in tracker.tracks]
                self.bboxes[idx] = [np.int32(t.tlbr) for t in tracker.tracks]

    def draw(self, frames, detections=[], font_scale=0.5, font=cv2.FONT_HERSHEY_SIMPLEX, refs=[]):

        refs = self.refs if refs == [] else refs

        # Desenha bboxes de rastreio
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        for ref, frame in zip(refs, frames):
            if ref not in self.refs:
                continue

            idx = self.refs.index(ref)
            for box, id, label in zip(self.bboxes[idx], self.ids[idx], self.labels[idx]):
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
        for ref, frame, alldets in zip(refs, frames, detections):
            if ref not in self.refs:
                continue

            for detection in alldets:
                box = np.int32(detection.to_tlbr())
                pt1, pt2 = box[0:2], box[2:4]
                cv2.rectangle(frame, pt1=pt1, pt2=pt2, color=(0, 255, 255), thickness=1)
