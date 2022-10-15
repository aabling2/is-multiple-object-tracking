import re
import cv2
import socket
import numpy as np
from is_msgs.image_pb2 import Image, ObjectAnnotations
from .base import BaseMSGS
from is_mot_bytetrack.core.detection import Detection


class MessageConsumer(BaseMSGS):
    def _catch_topic_id(self, topic="CameraGateway.1.Frame"):
        id = None
        for t in [self.frame_topic, self.annot_topic]:
            pattern = t.replace('*', '(.*?)')
            result = re.search(pattern=rf'{pattern}', string=topic)
            if result:
                id = int(result.group(1))
                break

        return id

    # Decodifica imagem para frame
    def _decode_image(self, message):
        img_bytes = message.unpack(Image)
        img_decode = np.frombuffer(img_bytes.data, np.uint8)
        frame = cv2.imdecode(img_decode, cv2.IMREAD_COLOR)
        return frame

    # Decodifica anotação para detecções
    def _decode_annotation(self, message):
        annotation_bytes = message.unpack(ObjectAnnotations)
        detections = []
        for annotation in annotation_bytes.objects:
            detection = Detection()
            detection.id = annotation.id
            detection.label = annotation.label
            detection.score = annotation.score
            v = annotation.region.vertices
            detection.bbox = detection.to_tlwh([v[0].x, v[0].y, v[1].x, v[1].y])
            detections.append(detection)

        return detections

    def consume(self):
        id, frame, detections = None, None, None
        try:
            # Consome mensagem
            message = self.channel.consume(timeout=0.0)

            # Segmenta tópico para destinar dados e decodificação (Frame | Annotation)
            id = self._catch_topic_id(message.topic)

            # Decodifica mensagens
            base_topic = message.topic.replace(f".{id}.", ".*.")
            if base_topic in self.frame_subscriptions:
                frame = self._decode_image(message)

            elif base_topic in self.annotation_subscriptions:
                detections = self._decode_annotation(message)

        except socket.timeout:
            pass

        except Exception as e:
            if self.log:
                self.drops += 1
                self.log.info(f"error: {e}")

        return id, frame, detections
