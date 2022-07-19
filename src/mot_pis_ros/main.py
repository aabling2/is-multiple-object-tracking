#!/usr/bin/env python3
import os
import cv2
import argparse
from .core.tracking import IntelligentSpaceMOT
from .core.image import map_cam_image_files
# from yolo_detector.cv_yolo import cvYOLO
from yolo_detector.torch_yolo import torchYOLOv5


def main(source, detector, deep_sort_model=None):
    """try:
        from is_msgs import decoder, encoder
    except Exception as e:
        print("PIS modules error:", e)
        return"""

    # Objeto de rastreio
    tracker = IntelligentSpaceMOT(deep_model=None)

    # Mapeia arquivos de imagens para carregar
    img_files = map_cam_image_files(source, ext='jpeg')

    scale = 0.5
    delay = 10

    for samples in img_files:

        # Obtém imagem
        # frames = decoder.consume_image()
        frames = [cv2.imread(os.path.join(source, filename)) for filename in samples]
        if scale < 1:
            frames = [cv2.resize(
                img, dsize=(int(img.shape[1]*scale), int(img.shape[0]*scale)))
                for img in frames]

        # Obtém anotação do detector
        # detector = decoder.consume_annotation()
        detections = [detector.detect(frame, draw=False) for frame in frames]

        # Atualiza rastreio
        tracker.update(frames, detections, reid=True)

        # Desenha detecções
        tracker.draw(frames, detections=detections)

        # Objetos de saída do rastreio
        # encoder.publish_annotation(objects=tracker.tracks)

        # Frames de saída
        # encoder.publish_image(frames=[frames])

        # Results
        if len(frames) <= 2:
            output = cv2.hconcat(src=frames)
        elif len(frames) <= 4:
            output = cv2.vconcat([
                cv2.hconcat(src=frames[0:2]),
                cv2.hconcat(src=frames[2:4])])

        cv2.imshow("Result", output)

        # Fonte de imagem
        # print("Source:", samples)

        # Key
        key = cv2.waitKey(delay)
        if key == 27:
            break
        elif key == 32:
            delay = 0
        elif key == 13:
            delay = 10

    cv2.destroyAllWindows()
    print("\nFinish!")


if __name__ == "__main__":

    # Descrição
    description = "Multiple Object Tracking for Programable Intelligent Space"
    print(f" {description} ".center(80, "*"))

    # Argumentos de entrada
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--src", type=str, required=True, help="Fonte de vídeo.")
    parser.add_argument("--model", type=str, default="yolov5s", help="Modelo treinado.")
    parser.add_argument("--classes", type=str, default="classes.txt", help="Lista de classes.")
    parser.add_argument("--gpu", action="store_true", default=False, help="Usa GPU como backend.")
    parser.add_argument(
        "--tracking_model", type=str, default="mars-small128.pb", help="Modelo pré-treinado para rastreio.")
    args = parser.parse_args()

    # Objeto de detecção
    # detector = cvYOLO(model=args.model, classes=args.classes, gpu=args.gpu)
    detector = torchYOLOv5(model=args.model, target_classes=['person'], thresh_confidence=0.7)

    # Framework de detecção e rastreio
    main(source=args.src, detector=detector, deep_sort_model=args.tracking_model)
