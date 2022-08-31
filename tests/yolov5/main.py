#!/usr/bin/env python3
import os
import cv2
import argparse
from .torch_yolo import torchYOLOv5
from messages import encoder
from core.image import map_cam_image_files


def main(args, detector=None):

    # Detector de objetos YOLO
    detector = torchYOLOv5(
        model=args.model,
        target_classes=args.classes.split(','),
        thresh_confidence=args.thresh,
        nms=args.nms)

    # Mapeia arquivos de imagens para carregar
    img_files = map_cam_image_files(args.source, ext='jpeg')

    scale = 0.5
    delay = 0
    args.max_frames += args.jump_frames

    for i, samples in enumerate(img_files):

        if i < args.jump_frames:
            continue

        if args.max_frames != 0 and i >= args.max_frames:
            break

        # Obtém imagem
        frames = [cv2.imread(os.path.join(args.source, filename)) for filename in samples][:args.max_src]
        if scale < 1:
            frames = [cv2.resize(
                img, dsize=(int(img.shape[1]*scale), int(img.shape[0]*scale)))
                for img in frames]

        # Obtém detecções feitas pelo detector
        detections = [detector.detect(frame) for frame in frames]

        # Objetos de saída do rastreio
        if args.publish:
            encoder.publish_image(frames=frames)
            encoder.publish_annotation(objects=detector.detections)

        # Desenha detecções
        for j, det in enumerate(detections):
            detector.detections = det
            detector.draw(frames[j])

        # Results
        if len(frames) <= 2:
            output = cv2.hconcat(src=frames)
        elif len(frames) <= 4:
            output = cv2.vconcat([
                cv2.hconcat(src=frames[0:2]),
                cv2.hconcat(src=frames[2:4])])
        cv2.imshow("Detector", output)

        if args.verbose:
            print("Source:", samples)

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

    # Argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Fonte de vídeo.")
    parser.add_argument("--model", type=str, default="yolov5s", help="Modelo treinado.")
    parser.add_argument("--max_src", type=int, default=4, help="Quantidade de imagens simultâneas.")
    parser.add_argument("--jump_frames", type=int, default=0, help="Pula frames iniciais com problemas.")
    parser.add_argument("--max_frames", type=int, default=0, help="Quantidade de imagens simultâneas.")
    parser.add_argument("--verbose", action='store_true', default=False, help="Exibe informações no terminal.")
    parser.add_argument("--classes", type=str, default='person,', help="Classes a filtrar do modelo.")
    parser.add_argument("--thresh", type=float, default=0.0, help="Threshold de confiança.")
    parser.add_argument("--nms", action='store_true', default=False, help="Aplica Non-maximum Supression.")
    parser.add_argument("--publish", action='store_true', default=False, help="Publica mensagens dos dados extraídos.")
    args = parser.parse_args()

    # Framework de detecção e rastreio
    main(args)
