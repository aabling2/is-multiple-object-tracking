class Detection():
    id: int
    label: str
    score: float
    bbox: list
    velocity: tuple

    def to_tlbr(self, bbox=[]):
        x, y, w, h = self.bbox.copy() if bbox == [] else bbox
        return [x, y, x+w, y+h]

    @staticmethod
    def to_tlwh(tlbr):
        x1, y1, x2, y2 = tlbr
        return [x1, y1, x2-x1, y2-y1]
