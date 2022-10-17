from is_wire.core import Channel,Subscription
from is_msgs.image_pb2 import Image
import numpy as np
import cv2
import json
import time


def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image

if __name__ == '__main__':
   
    print('---RUNNING EXAMPLE ---')

    broker_uri = "amqp://guest:guest@10.10.2.30:30000"
    channel = Channel(broker_uri)
    subscription = Subscription(channel=channel)
    for i in range(4):
        subscription.subscribe(topic='BYTETrack.{}.Frame'.format(i))
        print(f'Subscribed to ID: {i}')

    while True:
        msg = channel.consume()  
        im = msg.unpack(Image)
        frame = to_np(im)
        window = 'ObjectTracker'
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
