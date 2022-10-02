#!/usr/bin/env python3
import os
import cv2
import argparse
from is_wire.core import Logger
from mot_pis_bytetrack.core.config import *
from mot_pis_bytetrack.core.detection import RandomDetector
from mot_pis_bytetrack.pis.encoder import MessagePublisher


def main(args):

    # Serviço
    app_name = "ObjectDetector"
    service_name = f"{SERVICE_NAME}.{app_name}"
    log = Logger(name=service_name)

    # Define broker
    broker_uri = PIS_BROKER_URI if args.broker == "pis" else args.custom_broker

    # Fontes de vídeo e tópicos
    sources = [src.replace('~', os.environ.get('HOME')) for src in args.source.split(',')]
    ids = range(1, len(sources)+1)
    log.info(f"Source video from: {sources}")

    # Encoder de mensages para frames e detecções
    dst_streamer = MessagePublisher(
        name=service_name, broker=broker_uri, ids=ids, logger=log,
        frame_topic=args.dst_frame_topic,
        annotation_topic=args.dst_annotation_topic)

    # Abre captura de vídeo
    caps = [cv2.VideoCapture(src) for src in sources]
    dsizes = [(cap.get(3), cap.get(4)) for cap in caps]
    rsizes = [(int(ds[0]*args.scale), int(ds[1]*args.scale)) for ds in dsizes]

    # Detecções geradas aleatóriamente para testes
    random_detector = [
        RandomDetector(max_width=rs[0], max_height=rs[1], qtd=1, min_score=0.5)
        for rs in rsizes]

    # Variáveis
    delay = 0

    while dst_streamer.status is True:

        for i, cap in enumerate(caps):

            if not cap.isOpened():
                log.info("Capture closed")

            # Obtém imagem
            ret, frame = cap.read()
            if not ret or frame is None:
                log.info("No available frame")
                dst_streamer.status = False
                break

            # Redimensiona imagem
            frame = cv2.resize(frame, rsizes[i])
            detections = random_detector[i].update()

            # Publica dados do frame e detecções
            dst_streamer.publish_frame(frame, ids[i])
            dst_streamer.publish_detections(detections=detections, id=ids[i], width=rsizes[i][0], height=rsizes[i][1])

            # Results
            random_detector[i].draw(frame)
            cv2.imshow(sources[i], frame)

        # Key
        key = cv2.waitKey(delay)
        if key == 27:  # ESC
            break

        elif key == 13:  # Enter
            delay = 50

        elif key == 32:  # Espaço
            delay = 0

    cv2.destroyAllWindows()
    log.info("Finish")


if __name__ == "__main__":

    # Argumentos de entrada
    parser = argparse.ArgumentParser(description="Simulador de mensagens do PIS, publica frames e anotações de vídeos.")
    parser.add_argument("--source", type=str, default="~/Videos/video1.mp4", help='Fonte de vídeo. Use "," para multiplos.')
    parser.add_argument("--scale", type=float, default=0.5, help="Escala para redimensionamento.")
    parser.add_argument("--broker", type=str, default="pis", choices=['pis', 'custom'], help="Escolha do broker.")
    parser.add_argument("--custom_broker", type=str, default="amqp://guest:guest@localhost:5672", help="URI amqp do broker (default is_wire).")
    parser.add_argument("--dst_frame_topic", type=str, default="ObjectDetector.*.Rendered", help="Tópico destino dos frames.")
    parser.add_argument("--dst_annotation_topic", type=str, default="ObjectDetector.*.Detection", help="Tópico destino das detecções.")
    args = parser.parse_args()

    # Framework de detecção
    main(args)
