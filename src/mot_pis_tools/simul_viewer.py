#!/usr/bin/env python3
import cv2
import argparse
from is_wire.core import Logger
from mot_pis_bytetrack.core. config import *
from mot_pis_bytetrack.core.drawings import draw_tracks
from mot_pis_bytetrack.pis.decoder import MessageConsumer


def main(args):

    # Serviço
    service_name = f"{SERVICE_NAME}.VideoEndpoint"
    log = Logger(name=service_name)

    # Define broker
    broker_uri = PIS_BROKER_URI if args.broker == "pis" else args.custom_broker

    # Fontes de vídeo e tópicos
    ids = [int(x) for x in args.ids.split(',')] if args.ids != "*" else ["*"]

    # Encoder de mensages para frames e detecções
    src_streamer = MessageConsumer(
        name=service_name, broker=broker_uri, ids=ids, logger=log,
        frame_topic=args.src_frame_topic,
        annotation_topic=args.src_annotation_topic)

    # Variáveis
    delay = 1
    detection_tracks = {}
    output = None
    last_id = None
    while src_streamer.status is True:

        # Obtém imagem
        id, frame, tracks = src_streamer.consume()
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
    parser.add_argument("--broker", type=str, default="pis", choices=['pis', 'custom'], help="Escolha do broker.")
    parser.add_argument("--custom_broker", type=str, default="amqp://guest:guest@localhost:5672", help="URI amqp do broker (default is_wire).")
    parser.add_argument("--src_frame_topic", type=str, default="BYTETrack.*.Frame", help="Tópico fonte dos frames.")
    parser.add_argument("--src_annotation_topic", type=str, default="BYTETrack.*.Tracklets", help="Tópico fonte das detecções.")
    args = parser.parse_args()

    # Framework de exibição de resultados
    main(args)
