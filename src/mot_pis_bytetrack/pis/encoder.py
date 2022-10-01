import cv2
import socket
from is_wire.core import Message
from is_msgs.image_pb2 import Image, ObjectAnnotations
from .base import BaseMSGS


class MessagePublisher(BaseMSGS):
    def publish_frame(self, frame, id):
        message = Message()
        image = Image()
        topic = f"{self.main_topic}.{id}.Frame"
        try:
            # Codifica imagem para enviar como mensagem
            img_encode = cv2.imencode(".jpg", frame)[1]
            image.data = img_encode.tobytes()

            message.pack(image)
            self.channel.publish(message, topic=topic)

        except socket.timeout:
            if self.log:
                self.drops += 1
                self.log.info("error: socket timeout")

            return False

        except Exception as e:
            if self.log:
                self.drops += 1
                self.log.info(f"error: {e}")

            return False

        else:
            return True

    def publish_detections(self, detections, id, width, height):
        message = Message()
        annotations = ObjectAnnotations()
        topic = f"{self.main_topic}.{id}.Annotation"
        try:
            # Adiciona informações das detecções às anotações
            for det in detections:
                x1, y1, x2, y2 = det.to_tlbr()
                obj = annotations.objects.add()
                obj.id = det.id
                obj.label = det.label
                obj.score = det.score
                v1 = obj.region.vertices.add()
                v1.x = x1
                v1.y = y1
                v2 = obj.region.vertices.add()
                v2.x = x2
                v2.y = y2

            if width is not None and height is not None:
                annotations.resolution.width = width
                annotations.resolution.height = height

            message.pack(annotations)
            self.channel.publish(message, topic=topic)

        except socket.timeout:
            if self.log:
                self.drops += 1
                self.log.info("error: socket timeout")

            return False

        except Exception as e:
            if self.log:
                self.drops += 1
                self.log.info(f"error: {e}")

            return False

        else:
            return True
