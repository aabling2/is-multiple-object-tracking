import cv2
import numpy as np


def draw_tracks(frame, tracks, font_scale=0.4, font=cv2.FONT_HERSHEY_SIMPLEX, only_bbox=False):
    for t in tracks:
        sid = str(t.id)
        label = t.label
        box = np.int32(t.to_tlbr())
        np.random.seed(t.id)
        color = [np.random.randint(0, 255) for _ in range(3)]
        label_size = cv2.getTextSize(label+" ", 0, fontScale=font_scale, thickness=1)[0]
        id_size = cv2.getTextSize(sid, 0, fontScale=font_scale*1.3, thickness=1)[0]

        # Bounding box
        color = (0, 0, 255) if only_bbox else color
        thickness = 1 if only_bbox else 2
        cv2.rectangle(frame, box[:2], box[2:4], color, thickness)

        if not only_bbox:
            # Label
            pt1, pt2 = (box[0]-1, box[1] + 20), (box[0] + label_size[0] + id_size[0] + 1, box[1])
            cv2.rectangle(frame, pt1, pt2, color, -1)
            cv2.putText(frame, label, (box[0], box[1] + 10), font, font_scale, (0, 0, 0))

            # ID de referência do objeto
            pt1, pt2 = (box[0] + label_size[0] + 1, box[1] + 19), (box[0] + label_size[0] + id_size[0], box[1] - 1)
            cv2.rectangle(frame, pt1, pt2, (50, 50, 50), -1)
            cv2.putText(frame, sid, (pt1[0], pt2[1]+15), font, font_scale*1.3, (255, 255, 255))
