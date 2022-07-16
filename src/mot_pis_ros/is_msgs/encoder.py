from is_wire.core import Channel, Message
from is_msgs.common_pb2 import Image, ObjectAnnotation
#from is_msgs.robot_pb2 import BasicMoveTask


def publish_image(frames, channel_uri="amqp://guest:guest@10.10.2.30:30000", topic="Tracking.image"):
    channel = Channel(channel_uri)
    message = Message()
    for img in frames:
        image = Image()
        image.data = img
        image.uri = ""

        message.pack(image)
        channel.publish(message, topic=topic)


def publish_annotation(objects, channel_uri="amqp://guest:guest@10.10.2.30:30000", topic="Tracking.annotation"):
    channel = Channel(channel_uri)
    message = Message()
    for obj in objects:
        annotation = ObjectAnnotation()
        annotation.label = obj.label
        annotation.id = obj.id
        annotation.score = obj.score
        annotation.region = obj.bbox  # vertex?
        annotation.keypoints = None

        message.pack(annotation)
        channel.publish(message, topic=topic)
