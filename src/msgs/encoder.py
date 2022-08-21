import cv2
from is_wire.core import Channel, Message
from is_msgs.image_pb2 import Image, ObjectAnnotation


channel = Channel("amqp://guest:guest@localhost:5672")


def publish_image(frames, topic="Detector.YOLOv5.image"):
    message = Message()
    for i, img in enumerate(frames):
        image = Image()
        img_encode = cv2.imencode('.jpg', img)[1]
        image.data = img_encode.tobytes()
        # image.uri = f"source-{i}"
        message.pack(image)
        channel.publish(message, topic=topic+f".{i}")


def publish_annotation(objects, topic="Detector.YOLOv5.annotation"):
    message = Message()
    for i, obj in enumerate(objects):
        annotation = ObjectAnnotation()
        # annotation.label = obj.label.encode()
        # annotation.id = obj.id
        # annotation.score = obj.confidence
        #annotation.region = obj.tlwh
        message.pack(annotation)
        channel.publish(message, topic=topic+f".{i}")
