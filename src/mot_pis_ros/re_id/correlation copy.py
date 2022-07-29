import cv2
import numpy as np


class TrackedID():
    src = 0
    local_id = 0
    global_id = 0
    coef = 0.


class CrossCorrelationID():
    def __init__(self, box_type='tlwh', threshold=0.2):

        self.box_type = box_type
        self.ids = []
        self.features = []
        self.frames = []
        self.trackers = []
        self.next_id = 0
        self.thresh = threshold

    def _extract_features(self, frame, bboxes):
        samples = []
        for box in bboxes:
            if self.box_type == 'tlbr':
                x1, y1, x2, y2 = np.int32(box)
                samples.append(frame[y1:y2, x1:x2])
            elif self.box_type == 'tlwh':
                x, y, w, h = np.int32(box)
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
                pack_values = global_cost_matrix[:, j]
                if np.count_nonzero(pack_values) > 1:
                    min_idxs = pack_values < np.max(pack_values)
                    pack_values[min_idxs] = 0.
                    global_cost_matrix[:, j] = pack_values

        # Associações por métrica
        matches = np.where(global_cost_matrix > 0)

        # Atribui ids
        raw_ids = np.array([id for ids in old_ids for id in ids])
        idx_matches, idx_unmatches = [], []
        for i in range(M):
            if i in matches[1]:
                raw_ids[i] = raw_ids[matches[0][np.where(matches[1] == i)[0][0]]]
                idx_matches.append(i)
            elif clusters[i] != 0:
                idx_unmatches.append(i)

        next_val = len(idx_matches)
        for i in range(M):
            if i in idx_unmatches:
                raw_ids[i] = next_val
                next_val += 1

        # Mapeia raw ids para clusters
        N = len(multiboxes)
        new_ids = [raw_ids[np.where(clusters == i)] for i in range(N)]

        return new_ids

    def update(self, cam, tracker):
        self.trackers[cam] = tracker
        self.features[cam] = None

    def associate(self, detections, cam_ref, idx_detections, ids):

        # Extrai feature da imagem e detecção respectiva
        undetections = [detections[i] for i in idx_detections]
        img = self.frames[cam_ref]
        bboxes = [det.tlwh for det in undetections]
        ft_detections = self._extract_features(img, bboxes)

        # Extrai features se não extraidas ainda
        # indices, score = [], []
        M, N = len(self.frames), len(undetections)
        for i in range(M):
            if i == cam_ref:
                continue

            img = self.frames[i]
            if self.features[i] is None:
                bboxes = [np.int32(t.to_tlwh()) for t in self.trackers[i].tracks]
                self.features[i] = self._extract_features(img, bboxes)

            cost_matrix = np.zeros((N, len(self.features[i])))
            for j in range(N):
                # Calcula métrica de correlação
                coefs = [self._calc_correlation(ft_tracker, ft_detections[j]) for ft_tracker in self.features[i]]
                cost_matrix[i, :] = np.float32(coefs)
                """if len(coefs) > 0:
                    score.append(max(coefs))
                    indices.append((j, coefs.index(score[-1])))"""

            print("cost_matrix", cost_matrix)

        # Associa detecção com rastreador se score atender
        match_ids = ids
        """best_id = -1
        if len(score) > 0:
            best_choice = max(score)
            if best_choice >= self.thresh:
                idx_choice = score.index(best_choice)
                n_cam, n_tracker = indices[idx_choice]
                best_id = self.trackers[n_cam].tracks[n_tracker].track_id
                print("match", n_cam, n_tracker)

        # Se não encontrar associação
        if best_id == -1:
            best_id = self.next_id
            self.next_id += 1

        # Verifica se id já não foi atribuido nessa camera
        already_exist = [True for t in self.trackers[cam_ref].tracks if t.track_id == best_id]
        if not any(already_exist):
            target_id = result_id"""

        return match_ids
