import numpy as np
from deep_sort import linear_assignment
from deep_sort import kalman_filter
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort import generate_detections as gdet


class AppearenceMatcher():
    def __init__(self, deep_model, matching_threshold=0.2, budget=None, max_depth=1) -> None:
        self.metric = NearestNeighborDistanceMetric("cosine", matching_threshold, budget)
        self.kf = kalman_filter.KalmanFilter()
        self.max_depth = max_depth
        self.encoder = gdet.create_box_encoder(deep_model, batch_size=1)

    # Associate confirmed tracks using appearance features.
    def apply(self, images, multitrackers):

        # Extração de features para atualizar nos objetos detectados
        N = len(multitrackers)
        detection_features = []
        confirmed_tracks = []
        for i in range(N):
            detection_features.append(self.encoder(images[i], [t.to_tlwh() for t in multitrackers[i].tracks]))
            confirmed_tracks.append([n for n, t in enumerate(multitrackers[i].tracks) if t.is_confirmed()])

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([detection_features[i] for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Roda métrica de associação para cada grupo de rastreadores
        for i in range(N):
            for j in range(N):
                """if i == j:
                    continue"""

                matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                    distance_metric=gated_metric,
                    max_distance=self.metric.matching_threshold,
                    cascade_depth=self.max_depth,
                    tracks=multitrackers[i].tracks, detections=multitrackers[j].tracks,
                    track_indices=confirmed_tracks[i])

                print("test", matches_a, unmatched_tracks_a, unmatched_detections)

        return []
