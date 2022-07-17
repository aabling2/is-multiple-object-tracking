#!/usr/bin/env python3
import cv2
import time
import argparse
import numpy as np


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4


def detect(image, net):
    blob = cv2.dnn.blobFromImage(
        image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds


def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = (
                    row[0].item(), row[1].item(),
                    row[2].item(), row[3].item())
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


def main(source, model, classes):
    # Preset de cores
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    # Abre captura de vídeo
    cap = cv2.VideoCapture(source)

    # Carrega modelo
    net = cv2.dnn.readNet(model)

    # Define backend com GPU ou CPU
    active_gpu = cv2.cuda.getCudaEnabledDeviceCount()
    print("GPUs:", active_gpu)
    if active_gpu > 0:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Carrega classes
    with open(classes, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    # Variáveis para fps
    fps = 0
    count_frames = 0
    max_count = 20

    while cap.isOpened():
        if count_frames == 0:
            start = time.time()

        ret, frame = cap.read()

        if not ret or frame is None:
            break

        # Faz detecções com yolo
        inputImage = format_yolov5(frame)
        outs = detect(inputImage, net)
        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

        # Desenha detecções
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(
                frame, (box[0], box[1] - 20),
                (box[0] + box[2], box[1]), color, -1)
            cv2.putText(
                frame, class_list[classid],
                (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (0, 0, 0))

        # Calcula fps
        if count_frames >= max_count:
            fps = count_frames/(time.time() - start)
            count_frames = 0
        else:
            count_frames += 1

        # Desenha fps na imagem
        fps_label = "FPS: %.2f" % fps
        cv2.putText(
            frame, fps_label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Results
        cv2.imshow("Result", frame)

        # Key
        key = cv2.waitKey(10)
        if key == 27:
            break

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Open Video Deep SORT")
    parser.add_argument(
        "--video", type=str, required=True, help="Fonte de vídeo.")
    parser.add_argument(
        "--model", type=str, default="../datasets/YOLO/yolov5/yolov5s.onnx",
        help="Modelo treinado.")
    parser.add_argument(
        "--classes", type=str, default="../datasets/YOLO/yolov5/classes.txt",
        help="Lista de classes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        source=args.video,
        model=args.model,
        classes=args.classes)
