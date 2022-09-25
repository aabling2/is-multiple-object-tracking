from is_wire.core import Channel, Subscription


class BaseMSGS():
    def __init__(self, broker="amqp://guest:guest@localhost:5672", exchange="is",
                 main_topic="CameraGateway", ids=['*'], logger=None):
        # Canal com broker
        self.status = True
        self.main_topic = main_topic  # tópico principal
        self.ids = ids
        self.log = logger
        try:
            self.channel = Channel(broker, exchange=exchange)

            # Subscrição para consumo de frames
            self.frame_subscriptions = {
                f"{main_topic}.{stp}.Frame": Subscription(self.channel)
                for stp in ids}
            self._subscribe(self.frame_subscriptions)

            # Subscrição para consumo de anotações
            self.annotation_subscriptions = {
                f"{main_topic}.{stp}.Annotation": Subscription(self.channel)
                for stp in ids}
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
