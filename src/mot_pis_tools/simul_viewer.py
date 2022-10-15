#!/usr/bin/env python3
import cv2
import argparse
from is_wire.core import Logger
from mot_pis_bytetrack.core.utils import load_json_config
from mot_pis_bytetrack.core.drawings import draw_tracks
from mot_pis_bytetrack.pis.decoder import MessageConsumer


def main(args):

    # Carrega configurações
    config = load_json_config(args.config)
    if config is None:
        exit()

    # Serviço
    service_name = f"{config['service_name']}.ObjectViewer"
    log = Logger(name=service_name)

    # Define broker
    broker_uri = config['broker_uri']

    # Encoder de mensages para frames e detecções
    src_streamer = MessageConsumer(
        name=service_name, broker=broker_uri, ids=["*"], logger=log,
        frame_topic=f"{config['topic_main_name']}.*.{config['topic_dst_frames']}",
        annotation_topic=f"{config['topic_main_name']}.*.{config['topic_dst_annotations']}")

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
                tracks = detection_tracks.get(id) if id in detection_tracks else []
                if tracks:
                    draw_tracks(output, tracks, only_bbox=True)

        if output is not None and last_id is not None:
            cv2.imshow(f"ObjectViewer-{last_id}", output)

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
    parser = argparse.ArgumentParser(description="Simulador de visualização de vídeo e anotações.")
    parser.add_argument("--config", type=str, default="options.json", help="Arquivo de configurações.")
    args = parser.parse_args()

    # Framework de exibição de resultados
    main(args)
