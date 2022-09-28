import re
import cv2
import socket
import numpy as np
from is_msgs.image_pb2 import Image, ObjectAnnotation
from .base import BaseMSGS


class MessageConsumer(BaseMSGS):
    def _catch_topic(self, topic="CameraGateway.1.Frame", targets=[]):
        id, target = None, None
        for t in targets:
            result = re.search(pattern=rf'{self.main_topic}.(.*?).{t}', string=topic)
            if result:
                id = int(result.group(1))
                target = t
                break

        return id, target

    # Decodifica imagem para frame
    def _decode_image(self, message):
        img_bytes = message.unpack(Image)
        img_decode = np.frombuffer(img_bytes.data, np.uint8)
        frame = cv2.imdecode(img_decode, cv2.IMREAD_COLOR)
        return frame

    # Decodifica anotação para detecções
    def _decode_annotation(self, message):
        annotation_bytes = message.unpack(ObjectAnnotation)
        detections = annotation_bytes
        return detections

    def consume(self, targets=['Frame', 'Annotation']):
        id, frame, detections = None, None, None
        try:
            # Consome mensagem
            message = self.channel.consume(timeout=1.0)

            # Segmenta tópico para destinar dados e decodificação (Frame | Annotation)
            id, target = self._catch_topic(message.topic, targets)

            # Decodifica mensagens
            if target == 'Frame':
                frame = self._decode_image(message)

            elif target == 'Annotation':
                detections = self._decode_annotation(message)

        except socket.timeout:
            if self.log:
                self.drops += 1
                self.log.info("error: socket timeout")

        except Exception as e:
            if self.log:
                self.drops += 1
                self.log.info(f"error: {e}")

        return id, frame, detections
