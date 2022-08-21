#!/usr/bin/env python3
import os
import cv2
import time
import argparse
from core.multitracking import IntelligentSpaceMOT
from yolov5.core.image import map_cam_image_files
from msgs import decoder, encoder


def main(args, detector=None):

    # Objeto de rastreio
    multicam_tracker = IntelligentSpaceMOT(reid=args.reid)

    # Mapeia arquivos de imagens para carregar
    img_files = map_cam_image_files(args.src, ext='jpeg')

    scale = 0.5
    delay = 0
    args.max_frames += args.jump_frames

    for i, samples in enumerate(img_files):

        if i < args.jump_frames:
            continue

        if args.max_frames != 0 and i >= args.max_frames:
            break

        start = time.time()

        # Obtém imagem
        #frames_test = decoder.consume_image()
        frames = [cv2.imread(os.path.join(args.src, filename)) for filename in samples][:args.num_src]
        if scale < 1:
            frames = [cv2.resize(
                img, dsize=(int(img.shape[1]*scale), int(img.shape[0]*scale)))
                for img in frames]

        # Obtém anotação do detector
        #detector = decoder.consume_annotation()
        detections = [detector.detect(frame) for frame in frames]

        # Atualiza rastreio multicam
        if not args.no_tracking:
            multicam_tracker.update(frames, detections)
            multicam_tracker.draw(frames, detections=detections)

        else:
            for j, det in enumerate(detections):
                detector.detections = det
                detector.draw(frames[j])

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

        if args.measure:
            print("".center(50, ' '), end='\r')
            print(
                " |--> FPS: {:.2f}".format(1/(time.time()-start)), "/",
                "IDs: {:}".format(multicam_tracker.count_ids), end='\r')

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
    parser.add_argument("--num_src", type=int, default=4, help="Quantidade de imagens simultâneas.")
    parser.add_argument("--jump_frames", type=int, default=0, help="Pula frames iniciais com problemas.")
    parser.add_argument("--max_frames", type=int, default=0, help="Quantidade de imagens simultâneas.")
    parser.add_argument("--reid", action='store_true', default=False, help="Habilita reidentificação.")
    parser.add_argument("--no_tracking", action='store_true', default=False, help="Desabilita rastreio.")
    parser.add_argument("--measure", action='store_true', default=False, help="Métricas em tempo real.")
    args = parser.parse_args()

    # Objeto de detecção com YOLO, subtitui serviço de detecção
    from yolov5.torch_yolo import torchYOLOv5
    detector = torchYOLOv5(model=args.model, target_classes=['person'], thresh_confidence=0.0, nms=False)

    # Framework de detecção e rastreio
    main(args, detector=detector)
