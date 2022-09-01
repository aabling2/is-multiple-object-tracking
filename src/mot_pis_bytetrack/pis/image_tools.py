import cv2
import numpy as np

from is_msgs.image_pb2 import Image, ObjectAnnotations


def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image


def to_image(input_image, encode_format='.jpeg', compression_level=0.8):
    if isinstance(input_image, np.ndarray):
        if encode_format == '.jpeg':
            params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
        elif encode_format == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
        else:
            return Image()
        cimage = cv2.imencode(ext=encode_format, img=input_image, params=params)
        return Image(data=cimage[1].tobytes())
    elif isinstance(input_image, Image):
        return input_image
    else:
        return Image()

def to_objectDetections(objects, names, image):
    obs = ObjectAnnotations()
    for x1, y1, x2, y2, conf, classe in objects:
        ob = obs.objects.add()
        ob.label = ""
        ob.score = conf
        ob.id = int(classe)
        v1 = ob.region.vertices.add()
        v1.x = x1
        v1.y = y1
        v2 = ob.region.vertices.add()
        v2.x = x1
        v2.y = y2
    obs.resolution.width = image.shape[1]
    obs.resolution.height = image.shape[0]
    return obs
