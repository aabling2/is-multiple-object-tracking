#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from is_wire.core import Logger
from mot_pis_bytetrack.core.detection import RandomDetector
from mot_pis_bytetrack.pis.encoder import MessagePublisher
from mot_pis_bytetrack.pis.decoder import MessageConsumer


APPLICATION_NAME = "RandomDetector"
SERVICE_NAME = f"VideoDetector.{APPLICATION_NAME}"
BROKER_URI = "amqp://10.10.2.30:30000"
POSSIBLE_LABELS = ['person', 'car', 'unknown']


def main(args):

    # Serviço
    log = Logger(name=SERVICE_NAME)

    # Fontes de vídeo e tópicos
    sources = [src.replace('~', os.environ.get('HOME')) for src in args.source.split(',')]
    ids = range(1, len(sources)+1)
    log.info(f"Source video from: {sources}")

    # Encoder de mensages para frames e detecções
    streamer = MessagePublisher(
        name=SERVICE_NAME,
        # broker=BROKER_URI,
        main_topic=APPLICATION_NAME, ids=ids, logger=log)

    # Decoder de mensagens para exibição
    results = MessageConsumer(
        name=SERVICE_NAME,
        # broker=BROKER_URI,
        main_topic=APPLICATION_NAME, ids=ids, logger=log)

    # Abre captura de vídeo
    caps = [cv2.VideoCapture(src) for src in sources]
    dsizes = [(cap.get(3), cap.get(4)) for cap in caps]
    rsizes = [(int(ds[0]*args.scale), int(ds[1]*args.scale)) for ds in dsizes]

    # Detecções geradas aleatóriamente para testes
    random_detector = [RandomDetector(max_width=rs[0], max_height=rs[1], qtd=1) for rs in rsizes]

    # Variáveis
    delay = 0

    while streamer.status is True:

        for i, cap in enumerate(caps):

            if not cap.isOpened():
                log.info("Capture closed")

            # Obtém imagem
            ret, frame = cap.read()
            if not ret or frame is None:
                log.info("No available frame")
                streamer.status = False
                break

            # Redimensiona imagem
            frame = cv2.resize(frame, rsizes[i])
            detections = random_detector[i].update()

            # Publica dados do frame e detecções
            streamer.publish_frame(frame, ids[i], src=sources[i])
            streamer.publish_detections(detections=detections, width=rsizes[i][0], height=rsizes[i][1])

            # Results
            random_detector[i].draw(frame)
            cv2.imshow(sources[i], frame)

        # Key
        key = cv2.waitKey(delay)
        if key == 27:  # ESC
            break

        elif key == 13:  # Enter
            delay = 10

        elif key == 32:  # Espaço
            delay = 0

    cv2.destroyAllWindows()
    log.info("Finish")


if __name__ == "__main__":

    # Argumentos de entrada
    parser = argparse.ArgumentParser(description="Simulador de mensagens do PIS, publica frames e anotações de vídeos.")
    parser.add_argument(
        "--source", type=str, default="~/Videos/video1.mp4", help='Fonte de vídeo. Use "," para multiplos.')
    parser.add_argument("--scale", type=float, default=0.5, help="Escala para redimensionamento.")
    args = parser.parse_args()

    # Framework de detecção e rastreio
    main(args)
