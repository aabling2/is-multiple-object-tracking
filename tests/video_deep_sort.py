#!/usr/bin/env python3
import cv2
import argparse
import numpy as np
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker


def main(source, nms_max_overlap, max_cosine_distance, nn_budget):
    # Abre captura de vídeo
    cap = cv2.VideoCapture(source)

    # Objeto de segmentação
    backsub = cv2.createBackgroundSubtractorMOG2()

    # Cria rastreador
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        # Segmentação por subtração de fundo
        img = backsub.apply(frame)
        ret, thresh = cv2.threshold(
            img, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

        # Detecta contornos dos objetos segmentados
        contours, _ = cv2.findContours(
            image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # Cria bboxes
        detections = []
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 and w < 200 and h > 30 and h < 200:
                # w2, h2 = w//2, h//2
                # bbox = (x-w2, y-h2, x+w2, y+h2)
                bbox = (x, y, w, h)
                confidence = 1.0
                feature = np.ones(128)
                detections.append(Detection(bbox, confidence, feature))

        # Passa non-maxima suppression
        if len(detections) > 0:
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Atualiza rastreio
            tracker.predict()
            tracker.update(detections)

            # Desenha detecções
            for d in detections:
                det = d.tlwh
                pt1 = int(det[0]), int(det[1])
                pt2 = int(det[0]+det[2]), int(det[1]+det[3])
                cv2.rectangle(frame, pt1, pt2, (0, 255, 255), 3)

            # Desenha objetos rastreados
            for t in tracker.tracks:
                x, y, w, h = t.to_tlwh().astype(np.int)
                pt1 = int(x), int(y)
                pt2 = int(x + w), int(y + h)
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)

        # Results
        cv2.imshow("Result", frame)

        # Key
        key = cv2.waitKey(10)
        if key == 27:
            break

    cv2.destroyAllWindows()


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Open Video Deep SORT")
    parser.add_argument(
        "--video",  help="Source video.", required=True, type=str)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.video, args.nms_max_overlap,
        args.max_cosine_distance, args.nn_budget)
