import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


class TrackedID():
    src = 0
    local_id = 0
    global_id = 0
    coef = 0.


class CrossCorrelationID():
    def __init__(self, box_type='tlwh', threshold=0.2, qtd=1):

        self.box_type = box_type
        self.ids = []
        self.next_id = 0
        self.thresh = threshold
        self.frames = [None]*qtd
        self.trackers = [None]*qtd

    def update_global(self, frames, trackers):
        self.frames = frames
        self.trackers = trackers

    def update_local(self, cam, tracker):
        self.trackers[cam] = tracker

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

        # Retorna se qualquer tamanho zerado
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

        return np.max(coef)

    def associate(self, detections, cam_ref, idx_undetections, ids):

        # Extrai feature da imagem e detecção respectiva
        undetections = [detections[i] for i in idx_undetections]
        img = self.frames[cam_ref]
        bboxes = [det.tlwh for det in undetections]
        ft_detections = self._extract_features(img, bboxes)

        # Extrai features se não extraidas ainda
        # indices, score = [], []
        M, N = len(self.frames), len(undetections)
        global_cost_matrix = []
        map_id_trackers = []
        cam_id_trackers = []
        for i in range(M):
            if i == cam_ref:
                cam_id_trackers = [t.track_id for t in self.trackers[i].tracks]
                continue

            # Extrai features das regiões dos objetos rastreados na imagem atual
            bboxes = [np.int32(t.tlwh) for t in self.trackers[i].tracks]
            features = self._extract_features(self.frames[i], bboxes)

            # Calcula matriz de custo local, features de detecção vs features de rastreio
            n = (len(features))
            local_cost_matrix = np.zeros((n, N))
            for j in range(N):
                # Calcula métrica de correlação cruzada para cada detecção e todas features
                coefs = [self._calc_correlation(ft_tracker, ft_detections[j]) for ft_tracker in features]
                local_cost_matrix[:, j] = coefs

            # Adiciona a matriz local à matriz de custo global
            global_cost_matrix.append(local_cost_matrix)
            map_id_trackers.extend([t.track_id for t in self.trackers[i].tracks])

        # Encontra associação entre detecções e objetos rastreados
        if global_cost_matrix != []:
            global_cost_matrix = np.vstack(global_cost_matrix)
            matches = linear_assignment(global_cost_matrix, maximize=True)
            thresh_matches = np.array(np.where(global_cost_matrix[matches] > self.thresh))

            # Define ids dos objetos rastreados às detecções associadas
            idx_detections = list(matches[1][thresh_matches][0]) if thresh_matches.size > 0 else []
            idx_trackers = list(matches[0][thresh_matches][0]) if thresh_matches.size > 0 else []
            for i in range(N):
                is_matched_id = False
                if i in idx_detections:
                    # Atribui id de objeto rastreado associado a essa detecção
                    id = map_id_trackers[idx_trackers[idx_detections.index(i)]]

                    # Se não houver reincidência de id na mesma câmera
                    if id not in cam_id_trackers:
                        ids[idx_undetections[i]] = id
                        is_matched_id = True

                # Se não houve associação entre ids, usa próximo id do registro
                if not is_matched_id:
                    ids[idx_undetections[i]] = self.next_id
                    self.next_id += 1

        return ids
