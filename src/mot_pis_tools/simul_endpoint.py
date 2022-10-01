#!/usr/bin/env python3
import cv2
import argparse
from is_wire.core import Logger
from mot_pis_bytetrack.pis.decoder import MessageConsumer
from mot_pis_bytetrack.core.drawings import draw_tracks


SRC_APPLICATION_NAME = "BYTETrack"
DST_APPLICATION_NAME = "VideoEndpoint"
SERVICE_NAME = f"MultipleObjectTracking.{DST_APPLICATION_NAME}"
BROKER_URI = "amqp://10.10.2.30:30000"


def main(args):

    # Serviço
    log = Logger(name=SERVICE_NAME)

    # Fontes de vídeo e tópicos
    ids = [int(x) for x in args.ids.split(',')] if args.ids != "*" else ["*"]

    # Encoder de mensages para frames e detecções
    src_streamer = MessageConsumer(
        name=SERVICE_NAME,
        # broker=BROKER_URI,
        main_topic=SRC_APPLICATION_NAME, ids=ids, logger=log)

    # Variáveis
    delay = 1
    detection_tracks = {}
    output = None
    last_id = None
    while src_streamer.status is True:

        # Obtém imagem
        id, frame, tracks = src_streamer.consume(targets=['Frame', 'Annotation'])
        if id is not None:
            if tracks is not None:
                detection_tracks[id] = tracks

            # Resultados no frame
            if frame is not None:
                last_id = id
                output = frame.copy()
                if args.draw:
                    tracks = detection_tracks.get(id) if id in detection_tracks else []
                    draw_tracks(output, tracks)

        if output is not None and last_id is not None:
            cv2.imshow(f"Result-{last_id}", output)

            # Key
            key = cv2.waitKey(delay)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                break

            elif key == 13:  # Enter
                delay = 1

            elif key == 32:  # Espaço
                delay = 0

    cv2.destroyAllWindows()
    log.info("Finish")


if __name__ == "__main__":

    # Argumentos de entrada
    parser = argparse.ArgumentParser(description="Multicam BYTETracker for Programable Intelligent Space")
    parser.add_argument("--ids", type=str, default="*", help="IDs das fontes de consumo (* para todos tópicos).")
    parser.add_argument("--draw", action='store_true', default=False, help="Desenha objetos rastreados no frame.")
    args = parser.parse_args()

    # Framework de exibição de resultados
    main(args)
