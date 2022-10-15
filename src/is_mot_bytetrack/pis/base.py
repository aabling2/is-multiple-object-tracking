from is_wire.core import Channel, Subscription


class BaseMSGS():
    def __init__(self, name=None, broker="amqp://guest:guest@localhost:5672", exchange="is",
                 ids=['*'], frame_topic="", annotation_topic="",
                 logger=None):

        self.status = True
        self.log = logger
        self.drops = 0
        self.ids = ids
        self.frame_topic = frame_topic  # tópico para consumo dos frames
        self.annot_topic = annotation_topic  # tópico para consumo das anotações

        try:
            self.channel = Channel(uri=broker, exchange=exchange)
            self.frame_subscriptions = {}
            self.annotation_subscriptions = {}

            # Subscrição para consumo de frames
            if frame_topic:
                self.frame_subscriptions = {
                    f"{frame_topic.replace('*', str(i))}": Subscription(channel=self.channel, name=name)
                    for i in ids}
                self._subscribe(self.frame_subscriptions)

            # Subscrição para consumo de anotações
            if annotation_topic:
                self.annotation_subscriptions = {
                    f"{annotation_topic.replace('*', str(i))}": Subscription(channel=self.channel, name=name)
                    for i in ids}
                self._subscribe(self.annotation_subscriptions)

            if self.log:
                self.log.info(f"Successful connection with broker: {broker}")
                self.log.info(f"Frame subscription topics: {list(self.frame_subscriptions.keys())}")
                self.log.info(f"Annotation subscription topics: {list(self.annotation_subscriptions.keys())}")

        except Exception as e:
            self.status = False

            if logger:
                self.log.info(f"Error connection with broker: {broker} | {e}")

    def _subscribe(self, subscriptions={}):
        for topic, sub in subscriptions.items():
            sub.subscribe(topic)
