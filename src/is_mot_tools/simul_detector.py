#!/usr/bin/env python3
import os
import cv2
import argparse
from is_wire.core import Logger
from .scripts.detection import RandomDetector
from is_mot_bytetrack.core.utils import load_json_config
from is_mot_bytetrack.pis.encoder import MessagePublisher


TOPIC_MAIN_NAME = "ObjectDetector"
TOPIC_DST_FRAMES = "CameraGateway.*.Frame"  # f"{TOPIC_MAIN_NAME}.*.Rendered"
TOPIC_DST_ANNOTATIONS = f"{TOPIC_MAIN_NAME}.*.Detection"


def main(args):

    # Carrega configurações
    config = load_json_config(args.config)
    if config is None:
        exit()

    # Serviço
    service_name = f"{config['service_name']}.{TOPIC_MAIN_NAME}"
    log = Logger(name=service_name)

    # Define broker
    broker_uri = config['broker_uri']

    # Fontes de vídeo e tópicos
    sources = [src.replace('~', os.environ.get('HOME')) for src in args.source.split(',')]
    ids = range(1, len(sources)+1)
    log.info(f"Source video from: {sources}")

    # Encoder de mensages para frames e detecções
    dst_streamer = MessagePublisher(
        name=service_name, broker=broker_uri, ids=ids, logger=log,
        frame_topic=TOPIC_DST_FRAMES,
        annotation_topic=TOPIC_DST_ANNOTATIONS)

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
    parser.add_argument("--config", type=str, default="options.json", help="Arquivo de configurações.")
    parser.add_argument("--scale", type=float, default=0.5, help="Escala para redimensionamento.")
    parser.add_argument(
        "--source", type=str, default="~/Videos/video1.mp4", help='Fonte de vídeo. Use "," para multiplos.')
    args = parser.parse_args()

    # Framework de detecção
    main(args)
