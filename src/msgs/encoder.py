from is_wire.core import Channel, Message
from is_msgs.image_pb2 import Image, ObjectAnnotation


channel = Channel("amqp://guest:guest@localhost:5672")


def publish_image(frames, topic="Detector.image"):
    message = Message()
    for img in frames:
        image = Image()
        image.data = img
        image.uri = ""

        message.pack(image)
        channel.publish(message, topic=topic)


def publish_annotation(objects, topic="Detector.annotation"):
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
