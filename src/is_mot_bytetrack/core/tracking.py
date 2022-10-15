import numpy as np
from .drawings import draw_tracks
from is_mot_bytetrack.bytetrack.tracker import BYTETracker


class MulticamBYTETracker():
    def __init__(self, max_src=-1, refs=[1, 2, 3, 4], track_threshold=0.5):

        self.max_src = max_src
        self.trackers = None  # Rastreadores
        self.ids = []  # ids multicâmera
        self.labels = []  # labels conforme classes
        self.bboxes = []  # bbox multicâmera
        self.trackers = []
        self.refs = []
        self.track_thresh = track_threshold

        # Aloca rastreadores pré definidos
        if refs != ['*']:
            for ref in refs:
                self._create_tracker(ref)

    def _create_tracker(self, ref):
        if len(self.trackers) < self.max_src or self.max_src == -1:
            self.refs.append(ref)
            self.ids.append([])
            self.labels.append([])
            self.bboxes.append([])
            self.trackers.append(BYTETracker(
                frame_rate=30, track_thresh=self.track_thresh, track_buffer=30,
                match_tresh=0.9, fuse=True, src_id=ref))

    def update(self, detections, refs=[]):

        refs = self.refs if refs == [] else refs

        # Atualiza tracking de cada imagem
        tracks = []
        for dets, ref in zip(detections, refs):
            # Cria rastreador se não existir nas referências
            if ref not in self.refs:
                self._create_tracker(ref)

            # Atualiza tracker
            if ref in self.refs:
                idx = self.refs.index(ref)
                tracker = self.trackers[idx]
                tracker.update(dets)
                tracks.append(tracker.tracks)

                # Atualiza variáveis
                self.ids[idx] = [t.track_id for t in tracker.tracks]
                self.labels[idx] = [t.label for t in tracker.tracks]
                self.bboxes[idx] = [np.int32(t.tlbr) for t in tracker.tracks]

        return tracks

    def draw(self, frames, refs=[]):

        refs = self.refs if refs == [] else refs

        # Desenha bboxes de rastreio
        for ref, frame in zip(refs, frames):
            if ref not in self.refs:
                continue

            idx = self.refs.index(ref)
            draw_tracks(frame, self.trackers[idx].tracks)
