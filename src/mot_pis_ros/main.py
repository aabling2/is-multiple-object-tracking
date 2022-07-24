#!/usr/bin/env python3
import os
import cv2
import argparse
from core.multitracking import IntelligentSpaceMOT
from core.image import map_cam_image_files
from yolo.torch_yolo import torchYOLOv5


def main(source, detector, deep_model=None, deep=False, reid=False):
    """try:
        from is_msgs import decoder, encoder
    except Exception as e:
        print("PIS modules error:", e)
        return"""

    # Objeto de rastreio
    multitracker = IntelligentSpaceMOT(deep_model=deep_model, deep=deep, reid=reid)

    # Mapeia arquivos de imagens para carregar
    img_files = map_cam_image_files(source, ext='jpeg')

    scale = 0.5
    delay = 0

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
        multitracker.update(frames, detections)

        # Desenha detecções
        multitracker.draw(frames, detections=detections)

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
    parser.add_argument("--deep", action='store_true', default=False, help="Habilita parte deep do rastreio.")
    parser.add_argument("--reid", action='store_true', default=False, help="Habilita reidentificação.")
    parser.add_argument("--model", type=str, default="yolov5s", help="Modelo treinado.")
    parser.add_argument(
        "--deep_model", type=str, default="../../datasets/DeepSORT/networks/mars-small128.pb",
        help="Modelo pré-treinado para rastreio.")
    args = parser.parse_args()

    # Objeto de detecção
    detector = torchYOLOv5(model=args.model, target_classes=['person'], thresh_confidence=0.7)

    # Framework de detecção e rastreio
    main(source=args.src, detector=detector, deep_model=args.deep_model, deep=args.deep, reid=args.reid)
