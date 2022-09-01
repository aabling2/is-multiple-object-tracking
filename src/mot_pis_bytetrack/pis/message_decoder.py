import cv2
import numpy as np
from is_wire.core import Channel, Subscription
from is_msgs.image_pb2 import Image, ObjectAnnotation

channel = Channel("amqp://guest:guest@localhost:5672", exchange="is")


def consume_image(topic="Detector.YOLOv5.image", n=1):
    frames = []
    for i in range(n):
        subscription = Subscription(channel)
        subscription.subscribe(topic+f".{i}")
        message = channel.consume()
        img_bytes = message.unpack(Image)
        img_decode = np.frombuffer(img_bytes.data, np.uint8)
        image = cv2.imdecode(img_decode, cv2.IMREAD_COLOR)
        frames.append(image)

    return frames


def consume_annotation(topic="Detector.YOLOv5.annotation", n=1):
    detections = []
    for i in range(n):
        subscription = Subscription(channel)
        subscription.subscribe(topic+f".{i}")
        message = channel.consume()
        annotation_bytes = message.unpack(ObjectAnnotation)
        detections.append(annotation_bytes)

    return detections
