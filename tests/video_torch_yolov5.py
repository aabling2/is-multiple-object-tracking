#!/usr/bin/env python3
import cv2
import time
import argparse
from src.yolo_detector.torch_yolo import torchYOLOv5


def main(source, model):
    # Abre captura de vídeo
    cap = cv2.VideoCapture(source)

    # YOLO detector
    detector = torchYOLOv5(model=model)

    # Variáveis para fps
    fps = 0
    count_frames = 0
    max_count = 20
    delay = 10

    while cap.isOpened():
        if count_frames == 0:
            start = time.time()

        ret, frame = cap.read()

        if not ret or frame is None:
            break

        # Faz detecções com yolo
        detector.detect(frame, draw=True)

        # Calcula fps
        if count_frames >= max_count:
            fps = count_frames/(time.time() - start)
            count_frames = 0
        else:
            count_frames += 1

        # Desenha fps na imagem
        fps_label = "FPS: %.2f" % fps
        cv2.putText(
            frame, fps_label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Results
        cv2.imshow("Result", frame)

        # Key
        key = cv2.waitKey(delay)
        if key == 27:
            break
        elif key == 32:
            delay = 0 if delay > 0 else 10

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Open Video Deep SORT")
    parser.add_argument(
        "--video", type=str, required=True, help="Fonte de vídeo.")
    parser.add_argument(
        "--model", type=str, default="yolov5s",
        help="Modelo pré-treinado (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(source=args.video, model=args.model)
