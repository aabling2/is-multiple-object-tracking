FROM ultralytics/yolov5:latest

WORKDIR /is-object-detector

COPY /src src
COPY etc/conf/options.json .
COPY requirements.txt .

RUN pip3 install -r requirements.txt