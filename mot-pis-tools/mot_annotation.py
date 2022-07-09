import os
import cv2
import argparse
from utils import save_json, load_json


VERSION = "0.1.0"
LABELS = ['pessoa', 'robo']
MAX_OBJECTS = 50


# Argumentos de entrada
def parse_args():
    """ Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Annotation dataset")
    parser.add_argument("--images", type=str, required=True, help="Caminho das imagens, main folder.")

    return parser.parse_args()


# Mapeia arquivos de imagens
def map_images(src):
    n_dir = len(src)
    batch_files = []
    for root, _, files in os.walk(src):
        if not files:
            continue

        batch_files.extend([
            os.path.join(root[n_dir:], x) for x in sorted(files)
            if os.path.splitext(x)[1] in ['.jpg', '.jpeg']
        ])

    return batch_files


# Cria arquivo JSON padrão
def create_annotation_file(dst):

    annotation = {
        "source": None,  # fonte do dataset, apenas para conhecimento
        "version": VERSION,  # versão do código de anotação de vídeo
        "samples": 0,  # qtd. de amostras/imagens
        "objects": [],  # ids rastreados
        "data": {"images": [], "bboxes": [], "labels": [], "ids": []}  # dados de anotação relevantes
    }

    save_json(dst, annotation)

    return annotation


# Atualiza arquivo JSON com dados novos
def update_annotation_file(filename, data, images, bboxes, labels, ids):
    idxs = [images.index(x) for x in sorted(images)]
    data['data']['images'] = [images[i] for i in idxs]
    data['data']['bboxes'] = [bboxes[i] for i in idxs]
    data['data']['labels'] = [labels[i] for i in idxs]
    data['data']['ids'] = [ids[i] for i in idxs]
    data['objects'] = list(set(data['data']['ids']))
    save_json(filename, data, message=True)


def nothing(x):
    pass


# Registra anotação de vídeo em arquivo JSON padrão
def main(src, imgfiles):

    # Carrega ou cria arquivo de anotação
    filename = os.path.join(src, "annotation")
    annotation = load_json(filename)
    if annotation is None:
        annotation = create_annotation_file(filename)

    # Registra dataset fonte
    annotation['source'] = src
    annotation['samples'] = len(imgfiles)

    # Carrega dados existentes
    data = annotation['data']
    images = data['images']
    bboxes = data['bboxes']
    labels = data['labels']
    ids = data['ids']

    # Cria janela e trackbars
    cv2.namedWindow('crop')
    cv2.createTrackbar('label', 'crop', 0, len(LABELS)-1, nothing)
    cv2.createTrackbar('id', 'crop', 0, MAX_OBJECTS, nothing)

    # Escrita
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 255)
    line = cv2.LINE_AA

    # Processo de anotação por frame
    i, n = 0, 0
    while True:

        # Carrega dados da amostra
        image = imgfiles[n]
        label = ""
        bbox = []
        id = 0

        # Atualiza amostras
        if image in images:
            i = images.index(image)
            bbox = bboxes[i]
            label = labels[i]
            id = ids[i]
        else:
            images.append(image)
            bboxes.append(bbox)
            labels.append(label)
            ids.append(id)

        # Lê imagens
        frame = cv2.imread(os.path.join(src, image))
        out = frame.copy()

        # Desenha objetos
        if bbox != []:
            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[0]+bbox[2], bbox[1]+bbox[3])

            # Bounding box
            cv2.rectangle(out, pt1=pt1, pt2=pt2, color=(0, 225, 0))

            # Label
            cv2.rectangle(out, (pt1[0], pt1[1]-20), (pt2[0], pt1[1]), (0, 255, 0), -1)
            cv2.putText(out, label, (pt1[0], pt1[1]-10), font, font_scale, (0, 0, 0))

            # ID de referência do objeto
            cv2.rectangle(out, (pt2[0]-2-12*len(str(id)), pt1[1]-19), (pt2[0]-1, pt1[1]-1), (50, 50, 50), -1)
            cv2.putText(out, str(id), (pt2[0]-2-12*len(str(id)), pt1[1]-3), font, font_scale*1.2, (255, 255, 255))

        # Indicativos na tela
        cv2.putText(out, "ESC - sair", (5, 15), font, font_scale, color, thickness, line)
        cv2.putText(out, "a - voltar", (5, 35), font, font_scale, color, thickness, line)
        cv2.putText(out, "d - avancar", (5, 55), font, font_scale, color, thickness, line)
        cv2.putText(out, "e - editar", (5, 75), font, font_scale, color, thickness, line)
        cv2.putText(out, "z - excluir", (5, 95), font, font_scale, color, thickness, line)

        # Mostra imagem
        cv2.imshow("frame", out)

        # Captura de teclas
        edit = False
        key = cv2.waitKey()
        if key == 27 or key == 32:
            update_annotation_file(filename, annotation, images, bboxes, labels, ids)
            if key == 27:
                break
        elif key == ord('d') and n < annotation['samples']:
            n += 1
        elif key == ord('a') and n > 0:
            n -= 1
        elif key == ord('e'):
            edit = True
        elif key == ord('z'):
            labels[i] = ""
            bboxes[i] = []
            ids[i] = 0
            print("Dados excluidos no frame:", image)

        # Edita anotação
        if edit:
            # Select ROI
            print()
            cv2.putText(frame, "Editando...", (5, 15), font, font_scale, color, thickness, line)
            r = cv2.selectROI("frame", frame, showCrosshair=False)
            if r == (0, 0, 0, 0):
                print("Invalid crop!")
                continue

            # Crop image
            crop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

            cv2.imshow("crop", crop)

            print("Selecione as opções e confirme com ENTER ou cancele com ESC!")
            while True:
                key = cv2.waitKey()
                if key == 27:
                    break
                elif key == 13:
                    # Atualiza valores de referência
                    bboxes[i] = r
                    labels[i] = LABELS[int(cv2.getTrackbarPos('label', 'crop'))]
                    ids[i] = int(cv2.getTrackbarPos('id', 'crop'))
                    break

            cv2.destroyWindow('crop')


if __name__ == '__main__':

    print(" Anotação de vídeo para MOT ".center(60, "*"))
    args = parse_args()
    mapfiles = map_images(src=args.images)
    main(src=args.images, imgfiles=mapfiles)
