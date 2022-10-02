#!/usr/bin/env python3
import cv2
import argparse
from is_wire.core import Logger
from .core.config import *
from .core.tracking import MulticamBYTETracker
from .pis.decoder import MessageConsumer
from .pis.encoder import MessagePublisher


def main(args):

    # Serviço
    service_name = f"{SERVICE_NAME}.{TRACKING_APPLICATION_NAME}"
    log = Logger(name=service_name)

    # Define broker
    broker_uri = PIS_BROKER_URI if args.broker == "pis" else args.custom_broker

    # Decoder de mensages
    ids = [int(x) for x in args.camera_ids.split(',')] if args.camera_ids != "*" else ["*"]
    src_streamer = MessageConsumer(
        name=service_name, broker=broker_uri, logger=log, ids=ids,
        frame_topic=args.src_frame_topic if args.show or args.draw else "",
        annotation_topic=args.src_annotation_topic)

    # Encoder de mensages
    dst_streamer = MessagePublisher(
        broker=broker_uri, ids=ids, logger=log,
        frame_topic=f"{TRACKING_APPLICATION_NAME}.*.{TRACKING_DST_FRAMES}" if args.draw else "",
        annotation_topic=f"{TRACKING_APPLICATION_NAME}.*.{TRACKING_DST_ANNOTATIONS}")

    # Objeto de rastreio em multiplas câmeras
    multi_tracker = MulticamBYTETracker(refs=ids, track_threshold=args.threshold)

    # Variáveis
    delay = 1
    width, height = None, None
    output = None
    last_id = None

    while src_streamer.status is True and src_streamer.drops < 100:

        # Obtém imagem
        id, frame, detections = src_streamer.consume()
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
    parser.add_argument("--show", action='store_true', default=False, help="Exibe janela das imagens resultantes.")
    parser.add_argument("--draw", action='store_true', default=False, help="Desenha objetos rastreados no frame.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold de rastreio low/high.")
    parser.add_argument("--broker", type=str, default="pis", choices=['pis', 'custom'], help="Escolha do broker.")
    parser.add_argument("--custom_broker", type=str, default="amqp://guest:guest@localhost:5672", help="URI amqp do broker (default is-wire).")
    parser.add_argument("--src_frame_topic", type=str, default="CameraGateway.*.Frame", help="Tópico fonte dos frames.")
    parser.add_argument("--src_annotation_topic", type=str, default="ObjectDetector.*.Detection", help="Tópico fonte das detecções.")
    parser.add_argument("--camera_ids", type=str, default="*", help="IDs dos tópicos fonte (* para todos tópicos).")
    args = parser.parse_args()

    # Framework de rastreio
    main(args)
