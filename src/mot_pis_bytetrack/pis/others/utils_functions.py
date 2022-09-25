import re
import sys

from google.protobuf.json_format import Parse
from is_wire.core import Logger


def get_topic_id(topic):
    re_topic = re.compile(r'CameraGateway.(\d+).Frame')
    result = re_topic.match(topic)
    if result:
        return result.group(1)


def create_exporter(service_name, uri):
    log = Logger(name="CreateExporter")
    zipkin_ok = re.match("http:\\/\\/([a-zA-Z0-9\\.]+)(:(\\d+))?", uri)
    if not zipkin_ok:
        log.critical("Invalid zipkin uri \"{}\", expected http://<hostname>:<port>", uri)
    exporter = ZipkinExporter(service_name=service_name,
                              host_name=zipkin_ok.group(1),
                              port=zipkin_ok.group(3),
                              transport=AsyncTransport)
    return exporter


# def load_options():
#     log = Logger(name='LoadingOptions')
#     op_file = sys.argv[1] if len(sys.argv) > 1 else 'options.json'
#     try:
#         with open(op_file, 'r') as f:
#             try:
#                 op = Parse(f.read())
#                 log.info('ObjectDetectorOptions: \n{}', op)
#                 return op
#             except Exception as ex:
#                 log.critical('Unable to load options from \'{}\'. \n{}', op_file, ex)
#     except Exception as ex:
#         log.critical('Unable to open file \'{}\'', op_file)
