#!/usr/bin/env python3
import cv2
import argparse
from is_wire.core import Logger
from .core.tracking import MulticamBYTETracker
from .pis.decoder import MessageConsumer
from .pis.encoder import MessagePublisher


SRC_APPLICATION_NAME = "RandomDetector"
DST_APPLICATION_NAME = "BYTETrack"
SERVICE_NAME = f"MultipleObjectTracking.{DST_APPLICATION_NAME}"
BROKER_URI = "amqp://10.10.2.30:30000"


def main(args):

    # Serviço
    log = Logger(name=SERVICE_NAME)

    # Decoder de mensages
    ids = [int(x) for x in args.ids.split(',')] if args.ids != "*" else ["*"]
    src_streamer = MessageConsumer(
        name=SERVICE_NAME,
        # broker=BROKER_URI,
        main_topic=SRC_APPLICATION_NAME, ids=ids, logger=log)

    # Encoder de mensages
    dst_streamer = MessagePublisher(
        # broker=BROKER_URI,
        main_topic=DST_APPLICATION_NAME, ids=ids, logger=log)

    # Objeto de rastreio em multiplas câmeras
    multi_tracker = MulticamBYTETracker(refs=ids, track_threshold=args.threshold)

    # Variáveis
    delay = 1
    targets = ['Frame', 'Annotation'] if args.show else ['Annotation']
    width, height = None, None
    output = None
    last_id = None

    while src_streamer.status is True and src_streamer.drops < 100:

        # Obtém imagem
        id, frame, detections = src_streamer.consume(targets)
        if id is not None:

            # Atualização de rastreio
            if detections is not None:
                tracks = multi_tracker.update(detections=[detections], refs=[id])

                # Publica dados do rastreio
                if tracks:
                    dst_streamer.publish_detections(tracks[0], id, width, height)

            # Resultados no frame
            if frame is not None:
                last_id = id
                width, height = frame.shape[:2][::-1]

                output = frame.copy()
                if args.draw:
                    multi_tracker.draw(frames=[output], refs=[id])

                # Publica dados do frame
                dst_streamer.publish_frame(output, id)

        if args.show and output is not None and last_id is not None:
            cv2.imshow(f"BYTETrack-{last_id}", output)

            # Key
            key = cv2.waitKey(delay)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                break

            elif key == 13:  # Enter
                delay = 1

            elif key == 32:  # Espaço
                delay = 0

    log.info("Finish")


if __name__ == "__main__":

    # Argumentos de entrada
    parser = argparse.ArgumentParser(description="Multicam BYTETracker for Programable Intelligent Space")
    parser.add_argument("--ids", type=str, default="*", help="IDs das fontes de consumo (* para todos tópicos).")
    parser.add_argument("--show", action='store_true', default=False, help="Exibe janela das imagens resultantes.")
    parser.add_argument("--draw", action='store_true', default=False, help="Desenha objetos rastreados no frame.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold de rastreio low/high.")
    args = parser.parse_args()

    # Framework de rastreio
    main(args)
