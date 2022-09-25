#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from is_wire.core import Logger
from mot_pis_bytetrack.pis.encoder import MessagePublisher
from mot_pis_bytetrack.core.detection import Detection


APPLICATION_NAME = "RandomDetector"
SERVICE_NAME = f"VideoDetector.{APPLICATION_NAME}"
BROKER_URI = "amqp://10.10.2.30:30000"
# BROKER_URI = "amqp://guest:guest@localhost:5672"
POSSIBLE_LABELS = ['person', 'car', 'unknown']


def main(args):

    # Serviço
    log = Logger(name=SERVICE_NAME)

    # Fontes de vídeo e tópicos
    sources = [src.replace('~', os.environ.get('HOME')) for src in args.source.split(',')]
    ids = range(len(sources))
    log.info(f"Source video from: {sources}")

    # Encoder de mensages
    streamer = MessagePublisher(
        broker=BROKER_URI, main_topic=APPLICATION_NAME, ids=ids, logger=log)

    # Abre captura de vídeo
    caps = [cv2.VideoCapture(src) for src in sources]

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
            dsize = frame.shape[:2][::-1]
            frame = cv2.resize(frame, np.int32(np.int32(dsize)*args.scale))
            random_detections = [Detection(seed=i, labels=POSSIBLE_LABELS) for i in range(5)]

            # Obtém anotação do detector
            streamer.publish_frame(frame, ids[i], src=sources[i])
            streamer.publish_annotations(detections=random_detections, width=dsize[0], height=dsize[1])

            # Results
            if args.show:
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
    parser.add_argument("--show", action='store_true', default=False, help="Exibe janela das imagens resultantes.")
    args = parser.parse_args()

    # Framework de detecção e rastreio
    main(args)
