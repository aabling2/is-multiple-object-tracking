#!/usr/bin/env python3
import os
import cv2
import argparse
from yolo_detector.cv_yolo import YOLO
from .core.tracking import IntelligentSpaceMOT
from .core.image import map_cam_image_files


def main(source, detector):
    """try:
        from is_msgs import decoder, encoder
    except Exception as e:
        print("PIS modules error:", e)
        return"""

    # Objeto de rastreio
    tracker = IntelligentSpaceMOT()

    # Mapeia arquivos de imagens para carregar
    img_files = map_cam_image_files(source, ext='jpeg')

    resized = False

    for samples in img_files:

        # Obtém imagem
        # frames = decoder.consume_image()
        frames = [cv2.imread(os.path.join(source, filename)) for filename in samples][:1]
        if resized:
            frames = [cv2.resize(img, dsize=(420, 380)) for img in frames]

        # Obtém anotação do detector
        # detector = decoder.consume_annotation()
        detections = [detector.detect(frame, draw=True) for frame in frames]

        # Atualiza rastreio
        tracker.update(frames, detections, reid=False)

        # Desenha detecções
        # tracker.draw(frames)

        # Objetos de saída do rastreio
        # encoder.publish_annotation(objects=tracker.tracks)

        # Frames de saída
        # encoder.publish_image(frames=[frames])

        # Results
        output = cv2.hconcat(src=frames)
        cv2.imshow("Result", output)

        # Fonte de imagem
        print("Source:", samples)

        # Key
        key = cv2.waitKey(10)
        if key == 27:
            break

    cv2.destroyAllWindows()
    print("\nFinish!")


if __name__ == "__main__":

    # Descrição
    description = "Multiple Object Tracking for Programable Intelligent Space"
    print(f" {description} ".center(80, "*"))

    # Argumentos de entrada
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--src", type=str, required=True, help="Fonte de vídeo.")
    parser.add_argument("--model", type=str, default="../../datasets/YOLO/yolov5/yolov5n.onnx", help="Modelo treinado.")
    parser.add_argument("--classes", type=str, default="../../datasets/YOLO/yolov5/classes.txt", help="Lista de classes.")
    parser.add_argument("--gpu", action="store_true", default=False, help="Usa GPU como backend.")
    args = parser.parse_args()

    # Objeto de detecção
    detector = YOLO(
        model=args.model,
        classes=args.classes,
        gpu=args.gpu)

    # Framework de detecção e rastreio
    main(source=args.src, detector=detector)
