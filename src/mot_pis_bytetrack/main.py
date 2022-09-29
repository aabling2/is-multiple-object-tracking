#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from is_wire.core import Logger
from .core.tracking import MulticamBYTETracker
from .pis.decoder import MessageConsumer


SRC_APPLICATION_NAME = "RandomDetector"
DST_APPLICATION_NAME = "BYTETrack"
SERVICE_NAME = f"MultipleObjectTracking.{DST_APPLICATION_NAME}"
BROKER_URI = "amqp://10.10.2.30:30000"


def main(args):

    # Serviço
    log = Logger(name=SERVICE_NAME)

    # Decoder de mensages
    ids = args.ids.split(',')
    src_streamer = MessageConsumer(
        name=SERVICE_NAME,
        # broker=BROKER_URI,
        main_topic=SRC_APPLICATION_NAME, ids=ids, logger=log)

    # Encoder de mensages
    """ids = range(args.max_src)
    dst_streamer = MessageConsumer(
        broker=BROKER_URI, main_topic=SRC_APPLICATION_NAME, ids=ids, logger=log)"""

    # Objeto de rastreio em multiplas câmeras
    multi_tracker = MulticamBYTETracker(refs=ids)

    # Variáveis
    delay = 0
    targets = ['Frame', 'Annotation'] if args.show else ['Annotation']

    while src_streamer.status is True and src_streamer.drops < 100:

        # Obtém imagem
        id, frame, detections = src_streamer.consume(targets)
        if id is None:
            continue

        # Atualização de rastreio
        """if detections is not None:
            #atualizar no rastreador correto pelo id, adicionar rastreador se não existir na lista de ids
            log.info(f"Detections: {detections}")"""

        # Resultados no frame
        if frame is not None:

            output = frame.copy()
            if args.draw:
                multi_tracker.draw(frames=[output], refs=[id])

            if args.show:
                cv2.imshow(f"BYTETrack-{id}", output)

                # Key
                key = cv2.waitKey(delay)
                if key == 27:  # ESC
                    cv2.destroyAllWindows()
                    break

                elif key == 13:  # Enter
                    delay = 10

                elif key == 32:  # Espaço
                    delay = 0

    log.info("Finish")


if __name__ == "__main__":

    # Argumentos de entrada
    parser = argparse.ArgumentParser(description="Multicam BYTETracker for Programable Intelligent Space")
    parser.add_argument("--ids", type=str, default="0", help="IDs das fontes de consumo (* para todos tópicos).")
    parser.add_argument("--show", action='store_true', default=False, help="Exibe janela das imagens resultantes.")
    parser.add_argument("--draw", action='store_true', default=False, help="Desenha objetos rastreados no frame.")
    args = parser.parse_args()

    # Framework de detecção e rastreio
    main(args)
