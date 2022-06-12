import cv2
import numpy as np


class TrackedID():
    src = 0
    local_id = 0
    global_id = 0
    coef = 0.


class CrossCorrelationID():
    def __init__(self, tlbr_bbox='tlbr'):

        self.ids = None
        self.features = None
        self.tlbr_bbox = tlbr_bbox
        self.correlation_mat = None
        self.assigns = None

    def _extract_features(self, frame, bboxes):
        samples = []
        for box in bboxes:
            if self.tlbr_bbox:
                x1, y1, x2, y2 = box
                samples.append(frame[y1:y2, x1:x2])
            else:
                x, y, w, h = box
                samples.append(frame[y:y+h, x:x+w])

        return samples

    def _calc_correlation(self, feat1, feat2):

        shape1, shape2 = feat1.shape[:2], feat2.shape[:2]
        if 0 in shape1 or 0 in shape2:
            return 0.

        # Encontra template (menor imagem)
        src, template = feat1, feat2
        if shape2 > shape1:
            src, template = feat2, feat1

        # Reduz resolução se alguma medida maior que imagem src
        template = cv2.resize(template, src.shape[:2][::-1])

        # Cálculo de correlação
        coef = cv2.matchTemplate(src, template, method=cv2.TM_CCOEFF_NORMED)

        return np.round(np.max(coef), 3)

    def apply(self, frames, detections):

        # Extração de features e ids dos objetos
        n, gid = 0, 0
        features, tracked_ids = [], []
        for img, bboxes in zip(frames, detections):
            samples = self._extract_features(img, bboxes)
            features.append(samples)
            for i in range(len(samples)):
                tracked_ids.append(TrackedID())
                t = tracked_ids[-1]
                t.src = n
                t.local_id = i
                t.global_id = gid
                gid += 1
            n += 1

        # Compara features
        for t1 in tracked_ids:
            src_ref = 0
            score = []
            for t2 in tracked_ids:

                if t1.src == t2.src:
                    score.append(-1.)
                    continue

                coef = self._calc_correlation(features[t1.src][t1.local_id], features[t2.src][t2.local_id])
                score.append(coef)
                src_ref += 1

            match_id = np.argmax(score)
            match_score = score[match_id]
            t = tracked_ids[match_id]
            if match_score > 0. and match_score > t.coef:
                t.coef = match_score
                t.global_id = t1.global_id

        ids = [t.global_id for t in tracked_ids]

        return ids
