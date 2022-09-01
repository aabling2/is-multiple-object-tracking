from is_msgs.image_pb2 import Image, ObjectAnnotations
from is_wire.core import Channel, Logger, Status, StatusCode
from is_wire.rpc import ServiceProvider, LogInterceptor, TracingInterceptor
from pis.image_tools import to_np, to_objectDetections
#from .object_detector import ObjectDetector
from pis.utils_functions import load_options, create_exporter


class RPCObjectDetector(ObjectDetector):
    def __init__(self, options):
        super().__init__(options)

    def detect(self, image, ctx):
        try:
            det = super().detect(to_np(image))
            return to_objectDetections(det, return_names(), to_np(image).shape)

        except:
            return Status(code=StatusCode.INTERNAL_ERROR)


def main():
    service_name = 'ObjectDetector.Detect'
    log = Logger(name=service_name)

    op = load_options()
    obj_detector = RPCObjectDetector(op)

    channel = Channel(op.broker_uri)
    log.info('Connected to broker {}', op.broker_uri)

    provider = ServiceProvider(channel)
    provider.add_interceptor(LogInterceptor())

    exporter = create_exporter(service_name=service_name, uri=op.zipkin_uri)
    tracing = TracingInterceptor(exporter=exporter)
    provider.add_interceptor(tracing)

    provider.delegate(topic='ObjectDetector.Detect',
                      function=obj_detector.detect,
                      request_type=Image,
                      reply_type=ObjectAnnotations)

    provider.run()


if __name__ == "__main__":
    main()
