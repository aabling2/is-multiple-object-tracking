import re
import cv2
import socket
import numpy as np
from is_msgs.image_pb2 import Image, ObjectAnnotation
from .base import BaseMSGS


class MessageConsumer(BaseMSGS):
    def get_topic_id(self, topic="CameraGateway.1.Frame", target='Frame'):
        return re.search(pattern=rf'{self.main_topic}.(.*?).{target}', string=topic).group(1)

    def consume_image(self):
        try:
            message = self.channel.consume(timeout=0.0)

            # Decodifica imagem
            img_bytes = message.unpack(Image)
            img_decode = np.frombuffer(img_bytes.data, np.uint8)
            frame = cv2.imdecode(img_decode, cv2.IMREAD_COLOR)

            # Obtém id do tópico
            id = self.get_topic_id(message.topic, target='Frame')
            print("id test", id)

        except socket.timeout:
            if self.log:
                self.log.info("error: socket timeout")

            return None, None

        except Exception as e:
            if self.log:
                self.log.info(f"error: {e}")

            return None, None

        else:
            return frame, id

    def consume_annotation(section_topic=".annotation", n=1):
        detections = []
        """for i in range(n):
            subscription = Subscription(channel)
            subscription.subscribe(topic+f".{i}")
            message = channel.consume()
            annotation_bytes = message.unpack(ObjectAnnotation)
            detections.append(annotation_bytes)"""

        """try:
            message = self.channel.consume(timeout=0.0)

            # Decodifica imagem
            img_bytes = message.unpack(Image)
            img_decode = np.frombuffer(img_bytes.data, np.uint8)
            frame = cv2.imdecode(img_decode, cv2.IMREAD_COLOR)

            # Obtém id do tópico
            id = self.get_topic_id(message.topic, target='Frame')

        except socket.timeout:
            if self.log:
                self.log.info("error: socket timeout")

            return None, None

        except Exception as e:
            if self.log:
                self.log.info(f"error: {e}")

            return None, None

        else:
            return frame, id"""

        return detections
