#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from is_wire.core import Logger
from .core.multitracking import MulticamBYTETracker
from .pis.decoder import MessageConsumer


SRC_APPLICATION_NAME = "RandomDetector"
DST_APPLICATION_NAME = "BYTETrack"
SERVICE_NAME = f"MultipleObjectTracking.{DST_APPLICATION_NAME}"
BROKER_URI = "amqp://10.10.2.30:30000"
# BROKER_URI = "amqp://guest:guest@localhost:5672"


def main(args):

    # Serviço
    log = Logger(name=SERVICE_NAME)

    # Decoder de mensages
    src_streamer = MessageConsumer(
        broker=BROKER_URI, main_topic=SRC_APPLICATION_NAME, ids=["*"], logger=log)

    # Encoder de mensages
    """ids = range(args.max_src)
    dst_streamer = MessageConsumer(
        broker=BROKER_URI, main_topic=SRC_APPLICATION_NAME, ids=ids, logger=log)"""

    # Objeto de rastreio em multiplas câmeras
    # tracker = MulticamBYTETracker(num_src=args.max_src)

    # Variáveis
    delay = 0
    frame_ids = {}
    count_drops = 0

    while src_streamer.status is True:

        # Obtém imagem
        frame, id = src_streamer.consume_image()
        if frame is None:
            log.info(f"No available frame for id {id}")
            count_drops += 1

        # Results
        else:
            frame_ids[id] = frame
            if args.show:
                frames = frame_ids.values()
                if len(frames) <= 2:
                    output = cv2.hconcat(src=frames)
                elif len(frames) <= 4:
                    output = cv2.vconcat([
                        cv2.hconcat(src=frames[0:2]),
                        cv2.hconcat(src=frames[2:4])])
                cv2.imshow("BYTETrack", output)

        # Key
        key = cv2.waitKey(delay)
        if key == 27 or count_drops > 10:  # ESC
            break

        elif key == 13:  # Enter
            delay = 10

        elif key == 32:  # Espaço
            delay = 0

    cv2.destroyAllWindows()
    log.info("Finish")


if __name__ == "__main__":

    # Argumentos de entrada
    parser = argparse.ArgumentParser(description="Multicam BYTETracker for Programable Intelligent Space")
    parser.add_argument("--max_src", type=int, default=1, help="Quantidade máxima de fontes de vídeo aceitáveis.")
    parser.add_argument("--show", action='store_true', default=False, help="Exibe janela das imagens resultantes.")
    args = parser.parse_args()

    # Framework de detecção e rastreio
    main(args)
