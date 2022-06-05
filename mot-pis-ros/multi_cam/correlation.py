import cv2
import numpy as np


class CrossCorrelationID():
    def __init__(self, tlbr_bbox='tlbr'):

        self.ids = None
        self.features = None
        self.tlbr_bbox = tlbr_bbox
        self.correlation_mat = None
        self.assigns = None

    def _extract_features(self, frame, bboxes):
        features = []
        for box in bboxes:
            if self.tlbr_bbox:
                x1, y1, x2, y2 = box
                features.append(frame[y1:y2, x1:x2])
            else:
                x, y, w, h = box
                features.append(frame[y:y+h, x:x+w])

        return features

    def _calc_correlation(self, feat1, feat2):

        # Encontra template (menor imagem)
        src, template = feat1, feat2
        if feat2.shape[:2] > feat1.shape[:2]:
            src, template = feat2, feat1

        # Reduz resolução se alguma medida maior que imagem src
        template = cv2.resize(template, src.shape[:2][::-1])

        # Cálculo de correlação
        coef = cv2.matchTemplate(src, template, method=cv2.TM_CCOEFF_NORMED)

        return np.round(np.max(coef), 3)

    def apply(self, frames, detections):

        # Extração de features e ids dos objetos
        self.features = []
        self.ids = []
        packs = []
        start = 0
        end = 0
        for img, bboxes in zip(frames, detections):
            features = self._extract_features(img, bboxes)
            n = len(features)
            self.features.extend(features)
            self.ids.extend(np.arange(n, dtype=np.int32))

            # Referência de inicio e fim do pacote de detecções
            start += end
            end += n
            packs.extend([(start, end) for i in range(n)])

        # Compara features
        N = len(packs)  # Número máx. de imagens
        coefs = np.zeros((N, N), dtype=np.float32)  # Matriz de coeficientes
        for i, ref1 in enumerate(packs):
            for j, ref2 in enumerate(packs):
                if j == i or ref1 == ref2:
                    coefs[i][j] = -1
                    continue

                coefs[i, j] = self._calc_correlation(self.features[i], self.features[j])

        scores = np.where(coefs > 0.2, coefs, -1)
        print(scores)
        #rever prq pode ter matching pra cada câmera
