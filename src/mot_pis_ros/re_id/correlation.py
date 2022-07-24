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

    def apply(self, frames, multiboxes, old_ids):

        # Extrai features
        features, qtds, clusters = [], [], []
        ref = 0
        for img, boxes in zip(frames, multiboxes):
            features.extend(self._extract_features(img, boxes))
            qtds.append(len(boxes))
            clusters.extend([ref]*qtds[-1])
            ref += 1

        # Calcula métricas
        M = sum(qtds)
        global_cost_matrix = np.zeros((M, M))
        peers = []
        thresh = 0.2
        clusters = np.array(clusters)
        for i in range(M):
            for j in range(M):
                if i == j or (i, j) in peers or clusters[i] == clusters[j]:
                    continue

                # Calcula métrica de correlação
                coef = self._calc_correlation(features[i], features[j])

                # Aplica threshold
                global_cost_matrix[i][j] = coef if coef >= thresh else 0
                peers.extend([(i, j), (j, i)])

                # Reset min values row
                pack_idxs = clusters == clusters[j]
                pack_values = global_cost_matrix[i, pack_idxs]
                if np.count_nonzero(pack_values) > 1:
                    min_idxs = pack_values < np.max(pack_values)
                    pack_values[min_idxs] = 0.
                    global_cost_matrix[i, pack_idxs] = pack_values

                # Reset min values col
                pack_idxs = clusters == clusters[i]
                pack_values = global_cost_matrix[pack_idxs, j]
                if np.count_nonzero(pack_values) > 1:
                    min_idxs = pack_values < np.max(pack_values)
                    pack_values[min_idxs] = 0.
                    global_cost_matrix[pack_idxs, j] = pack_values

        matches = np.where(global_cost_matrix > 0)
        removes = []
        for i in range(len(matches[0])):
            if matches[0][i] in matches[1][:i]:
                corresp = np.where(matches[1] == matches[0][i])[0][0]
                matches[0][i] = matches[0][corresp]

            if matches[1][i] in matches[1][:i]:
                removes.append(i)

        matches = np.delete(matches, removes, axis=1)

        print(global_cost_matrix, matches)

        new_ids = old_ids.copy()
        next_val = max(matches[0]) + 1
        idx = 0
        for i, ids in enumerate(old_ids):
            for j, id in enumerate(ids):
                if idx in matches[0]:
                    new_ids[i][j] = idx

                elif idx in matches[1]:
                    new_ids[i][j] = int(matches[0][matches[1] == idx])

                else:
                    new_ids[i][j] = next_val
                    next_val += 1

                idx += 1

        print("new ids", new_ids)

        return new_ids
