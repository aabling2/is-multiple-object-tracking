#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from multi_cam.correlation import CrossCorrelationID


def parse_args():
    parser = argparse.ArgumentParser(description="EPFL Dataset - multiclass_ground_thuth")
    parser.add_argument("--images", type=str, required=True, help="Imagens multiplas câmeras.")
    parser.add_argument("--annotation", type=str, required=True, help="Anotação das bounding boxes.")
    return parser.parse_args()


def build_associations(img_path, bbox_path):

    batch = {}
    cams = len(os.listdir(img_path))

    # Associa path das imagens
    for root, _, file in sorted(os.walk(img_path)):
        count = 1
        for f in sorted(file):
            if batch.get(count) is None:
                batch[count] = {'img': [], 'bboxes': [[] for i in range(cams)]}

            batch[count]['img'].append(os.path.join(root, f))
            count += 1

    # Associa path das bboxes
    for root, dirs, file in os.walk(bbox_path):
        for f in file:
            with open(os.path.join(root, f)) as txt:
                data = txt.readlines()
                detections = []
                for d in data:
                    detections.append([float(val) for val in d.strip().split(" ") if val != ""])
                name, ext = os.path.splitext(f)
                _, frame, cam = name.split('_')
                batch[int(frame[5:])]['bboxes'][int(cam[3:])].extend(np.int32(detections))

    return batch


def main(batch):

    ccid = CrossCorrelationID()

    for i, sample in batch.items():

        # Carrega imagens do frame corrente
        images = [cv2.imread(img) for img in sample['img']]

        # Aplica associação entre objetos
        ids = ccid.apply(frames=images, detections=sample['bboxes'])

        # Desenha bboxes de detecção
        j = 0
        for i, bboxes in enumerate(sample['bboxes']):
            for box in bboxes:
                pt1 = box[0:2]
                pt2 = box[2:4]
                label = ids[j]
                np.random.seed(label)
                color = [int(x) for x in np.random.randint(0, 255, size=(3, ))]
                # cv2.putText(images[i], str(j), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(images[i], str(label), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(images[i], pt1, pt2, color=color, thickness=2)
                j += 1

        # Mostra imagens
        concat1 = cv2.hconcat(images[:3])
        concat2 = cv2.hconcat(images[3:])
        concat = cv2.vconcat([concat1, concat2])
        cv2.imshow("output", concat)

        # Captura de teclas
        key = cv2.waitKey()
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Argumentos de entrada
    args = parse_args()

    batch = build_associations(img_path=args.images, bbox_path=args.annotation)

    # Framework de detecção e rastreio
    main(batch)
