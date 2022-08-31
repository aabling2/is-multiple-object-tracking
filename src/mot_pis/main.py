#!/usr/bin/env python3
import cv2
import time
import argparse
from core.multitracking import IntelligentSpaceMOT
from pis import decoder, encoder


def main(args):

    # Objeto de rastreio
    multicam_tracker = IntelligentSpaceMOT(num_src=args.max_src)

    delay = 0

    while True:

        if args.measure:
            start = time.time()

        # Obtém imagem
        frames = decoder.consume_image(n=args.max_src)

        # Obtém anotação do detector
        detections = decoder.consume_annotation()
        print("detections", detections)

        # Atualiza rastreio multicam
        """if not args.no_tracking:
            multicam_tracker.update(frames, detections)
            multicam_tracker.draw(frames, detections=detections)"""

        # Objetos de saída do rastreio
        #encoder.publish_annotation(objects=tracker.tracks)

        # Results
        if len(frames) <= 2:
            output = cv2.hconcat(src=frames)
        elif len(frames) <= 4:
            output = cv2.vconcat([
                cv2.hconcat(src=frames[0:2]),
                cv2.hconcat(src=frames[2:4])])
        cv2.imshow("MOT-PIS", output)

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
    parser.add_argument("--max_src", type=int, default=4, help="Quantidade de imagens simultâneas.")
    parser.add_argument("--measure", action='store_true', default=False, help="Métricas em tempo real.")
    args = parser.parse_args()

    # Framework de detecção e rastreio
    main(args)
