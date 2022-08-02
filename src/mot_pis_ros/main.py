#!/usr/bin/env python3
import os
import cv2
import time
import argparse
from core.multitracking import IntelligentSpaceMOT
from core.image import map_cam_image_files
from yolo.torch_yolo import torchYOLOv5


def main(args):
    """try:
        from is_msgs import decoder, encoder
    except Exception as e:
        print("PIS modules error:", e)
        return"""

    # Objeto de rastreio
    multicam_tracker = IntelligentSpaceMOT(reid=args.reid)

    # Mapeia arquivos de imagens para carregar
    img_files = map_cam_image_files(args.src, ext='jpeg')

    scale = 0.5
    delay = 0

    for samples in img_files:

        start = time.time()

        # Obtém imagem
        #frames = decoder.consume_image()
        frames = [cv2.imread(os.path.join(args.src, filename)) for filename in samples][:args.max_src]
        if scale < 1:
            frames = [cv2.resize(
                img, dsize=(int(img.shape[1]*scale), int(img.shape[0]*scale)))
                for img in frames]

        # Obtém anotação do detector
        #detector = decoder.consume_annotation()
        detections = [detector.detect(frame) for frame in frames]

        # Atualiza rastreio multicam
        multicam_tracker.update(frames, detections)

        # Desenha detecções
        multicam_tracker.draw(frames, detections=detections)

        # Objetos de saída do rastreio
        #encoder.publish_annotation(objects=tracker.tracks)

        # Results
        if len(frames) <= 2:
            output = cv2.hconcat(src=frames)
        elif len(frames) <= 4:
            output = cv2.vconcat([
                cv2.hconcat(src=frames[0:2]),
                cv2.hconcat(src=frames[2:4])])
        cv2.imshow("Result", output)
        # print("Source:", samples)

        print("FPS:", round(1/(time.time()-start), 2))

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
    parser.add_argument("--max_src", type=int, default=4, help="Quantidade de imagens simultâneas.")
    parser.add_argument("--src", type=str, required=True, help="Fonte de vídeo.")
    parser.add_argument("--reid", action='store_true', default=False, help="Habilita reidentificação.")
    parser.add_argument("--model", type=str, default="yolov5s", help="Modelo treinado.")
    args = parser.parse_args()

    # Objeto de detecção
    detector = torchYOLOv5(model=args.model, target_classes=['person'], thresh_confidence=0.7)

    # Framework de detecção e rastreio
    main(args)
