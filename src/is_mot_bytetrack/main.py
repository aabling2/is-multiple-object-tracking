#!/usr/bin/env python3
import cv2
import sys
import argparse
from is_wire.core import Logger
from ._version import __version__
from .core.utils import load_json_config
from .core.tracking import MulticamBYTETracker
from .pis.decoder import MessageConsumer
from .pis.encoder import MessagePublisher


def main():

    # Carrega configurações
    filename = sys.argv[1] if len(sys.argv) == 2 else "../etc/conf/options.json"
    config = load_json_config(filename)
    if config is None:
        exit()

    # Serviço
    service_name = f"{config['service_name']}.{config['topic_main_name']}"
    log = Logger(name=service_name)
    log.info(f'{service_name}: version {__version__}')

    # Define broker
    broker_uri = config['broker_uri']

    # Define consumo e publicação de frames
    _consume_frames = True if config['publish_dst_frames'] else False
    _publish_frames = True if config['publish_dst_frames'] else False

    # Decoder de mensages
    ids = config['camera_ids']
    src_streamer = MessageConsumer(
        name=service_name, broker=broker_uri, logger=log, ids=ids,
        frame_topic=config['topic_src_frames'] if _consume_frames else "",
        annotation_topic=config['topic_src_annotations'])

    # Encoder de mensages
    dst_streamer = MessagePublisher(
        name=service_name, broker=broker_uri, ids=ids, logger=log,
        frame_topic=f"{config['topic_main_name']}.*.{config['topic_dst_frames']}" if _publish_frames else "",
        annotation_topic=f"{config['topic_main_name']}.*.{config['topic_dst_annotations']}")

    # Objeto de rastreio em multiplas câmeras
    multi_tracker = MulticamBYTETracker(
        refs=ids,
        track_threshold=config['min_detection_confidence'])

    # Variáveis
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

                # Desenha objetos rastreados
                output = frame.copy()
                multi_tracker.draw(frames=[output], refs=[id])

                # Publica dados do frame
                if _publish_frames:
                    dst_streamer.publish_frame(output, id)

    log.info("Finish")


if __name__ == "__main__":

    # Argumentos de entrada
    # parser = argparse.ArgumentParser(description="Multicam BYTETracker for Programable Intelligent Space")
    # parser.add_argument("--config", type=str, default="options.json", help="Arquivo de configurações.")
    # parser.add_argument("--show", action='store_true', default=False, help="Exibe janela das imagens resultantes.")
    # args = parser.parse_args()

    # # Framework de rastreio
    # main(args)
    main()
