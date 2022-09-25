import numpy as np


class Detection():
    id: int
    label: str
    score: float
    bboxes: list

    def __init__(self, seed=1, labels='unknown'):
        self.id = seed
        self.label = labels[0]
        self.score = 1.0
        self.bboxes = np.array([0, 0, 50, 50])

    @property
    def tlbr(self):
        ret = self.bboxes.copy()
        ret[2:] += ret[:2]
        return ret
