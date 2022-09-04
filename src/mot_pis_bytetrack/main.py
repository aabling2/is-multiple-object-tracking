#!/usr/bin/env python3
import cv2
import time
import argparse
from core.multitracking import MulticamBYTETracker
#from pis import message_decoder, message_encoder
from is_msgs.image_pb2 import Image
from is_wire.core import Logger, Subscription, Message
from pis.stream_channel import StreamChannel
from pis.image_tools import to_image, to_np, to_objectDetections
from pis.utils_functions import get_topic_id


BROKER_URI = "amqp://10.10.2.30:30000"


def main(args):

    # Serviço
    service_name = "ObjectTracker.Tracking"
    log = Logger(name=service_name)

    # Objeto de rastreio em multiplas câmeras
    multicam_tracker = MulticamBYTETracker(num_src=args.max_src)

    # Canal das câmeras
    channel = StreamChannel(BROKER_URI, exchange='is')
    log.info('Connected to broker {}', BROKER_URI)
    subs = Subscription(channel=channel, name=service_name)
    subs.subscribe(topic='CameraGateway.*.Frame')

    # Variáveis
    delay = 0

    while True:

        if args.measure:
            start = time.time()

        #teste imagem
        msg, dropped = channel.consume_last(return_dropped=True)
        img = msg.unpack(Image)
        img_np = to_np(img)
        camera_id = get_topic_id(msg.topic)
        print("camera_id", camera_id)

        # Obtém imagem
        #frames = message_decoder.consume_image(n=args.max_src)

        # Obtém anotação do detector
        #detections = message_decoder.consume_annotation()

        # Atualiza rastreio multicam
        """multicam_tracker.update(frames, detections)
        if args.show:
            multicam_tracker.draw(frames, detections=detections)"""

        #teste publish
        """objects_det = object_detector.return_obj_detec(im_np)
        objects = to_objectDetections(objects_det, object_detector.return_names(), im_np)
        obj_msg = Message()
        obj_msg.topic = 'ObjectDetector.{}.Detection'.format(camera_id)
        obj_msg.pack(objects)
        channel.publish(obj_msg)
        img_rendered = object_detector.to_img_detec(im_np, objects_det)
        rendered_msg = Message()
        rendered_msg.topic = 'ObjectDetector.{}.Rendered'.format(camera_id)
        rendered_msg.pack(to_image(img_rendered))
        channel.publish(rendered_msg)"""

        # Objetos de saída do rastreio
        #encoder.publish_annotation(objects=tracker.tracks)

        # Results
        if args.show:
            frames = [img]
            if len(frames) <= 2:
                output = cv2.hconcat(src=frames)
            elif len(frames) <= 4:
                output = cv2.vconcat([
                    cv2.hconcat(src=frames[0:2]),
                    cv2.hconcat(src=frames[2:4])])
            cv2.imshow("MOT-PIS", output)

        if args.measure:
            print("".center(50, ' '), end='\r')
            print(
                "FPS: {:.2f}".format(1/(time.time()-start)), "/",
                "IDs: {:}".format(multicam_tracker.count_ids), end='\r')

        # Key
        key = cv2.waitKey(delay)
        if key == 27:
            break
        elif key == 32:
            delay = 0
        elif key == 13:
            delay = 10

    cv2.destroyAllWindows()
    print("\nFinish!")


if __name__ == "__main__":

    # Descrição
    description = "Multicam BYTETracker for Programable Intelligent Space"
    print(f" {description} ".center(80, "*"))

    # Argumentos de entrada
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--max_src", type=int, default=4, help="Quantidade de imagens simultâneas.")
    parser.add_argument("--measure", action='store_true', default=False, help="Métricas em tempo real.")
    parser.add_argument("--show", action='store_true', default=False, help="Exibe janela das imagens resultantes.")
    args = parser.parse_args()

    # Framework de detecção e rastreio
    main(args)
