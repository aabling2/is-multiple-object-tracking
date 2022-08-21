from is_wire.core import Channel, Subscription
from is_msgs.image_pb2 import Image, ObjectAnnotation


def consume_image(channel_uri="amqp://guest:guest@10.10.35.9:5672", topic="Detector.image"):
    channel = Channel(channel_uri)
    subscription = Subscription(channel)
    subscription.subscribe(topic)
    message = channel.consume()
    image = message.unpack(Image)
    return image


def consume_annotation(channel_uri="amqp://guest:guest@10.10.35.9:5672", topic="Detector.annotation"):
    channel = Channel(channel_uri)
    subscription = Subscription(channel)
    subscription.subscribe(topic)
    message = channel.consume()
    annotation = message.unpack(ObjectAnnotation)
    return annotation
