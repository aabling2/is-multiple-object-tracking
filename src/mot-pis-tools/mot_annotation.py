import os
import cv2
import argparse
from utils import save_json, load_json


VERSION = "0.1.0"
LABELS = ['pessoa', 'robo']


# Argumentos de entrada
def parse_args():
    """ Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Annotation dataset")
    parser.add_argument("--images", type=str, required=True, help="Caminho das imagens, main folder.")
    parser.add_argument("--max_objects", type=int, default=50, help="Máx. número de objetos unicos para registrar.")
    parser.add_argument("--autocomplete", action='store_true', help="Completa frames com objetos já rastreados.")

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
def update_annotation_file(filename, data, images, bboxes, labels, ids, msg=True):
    idxs = [images.index(x) for x in sorted(images)]
    data['data']['images'] = [images[i] for i in idxs]
    data['data']['bboxes'] = [bboxes[i] for i in idxs]
    data['data']['labels'] = [labels[i] for i in idxs]
    data['data']['ids'] = [ids[i] for i in idxs]
    data['objects'] = list(set(x for fids in data['data']['ids'] for x in fids))
    save_json(filename, data, message=msg)


# Trackbar return
def nothing(x):
    pass


# Captura do mouse
class CallbackMouse():
    def __init__(self, window, shape=None) -> None:
        self.point = (0, 0)
        self.left_button_press = False
        self.right_button_press = False
        self.middle_button_press = False
        self.editing_id = None
        self.shape = shape
        cv2.setMouseCallback(window, self.mouse_events)

    def mouse_events(self, event, x, y, flags, param):

        if x < 0 or y < 0:
            self.left_button_press = False
            self.right_button_press = False
            self.middle_button_press = False
            return

        if self.shape is not None:
            if x >= self.shape[0] or y >= self.shape[1]:
                self.left_button_press = False
                self.right_button_press = False
                self.middle_button_press = False
                self.editing_id = None
                return

        self.point = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_button_press = True

        if event == cv2.EVENT_LBUTTONUP:
            self.left_button_press = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self.right_button_press = True

        if event == cv2.EVENT_RBUTTONUP:
            self.right_button_press = False

        if event == cv2.EVENT_MBUTTONDOWN:
            self.middle_button_press = True

        if event == cv2.EVENT_MBUTTONUP:
            self.middle_button_press = False


# Redefine posição da bounding box do objeto
def move_box(bbox, mouse, id):
    if mouse.editing_id is None or mouse.editing_id == id:
        mx, my = mouse.point
        x, y, w, h = bbox
        if mx > x and mx < x+w-1 and my > y and my < y+h-1:
            bbox[0] = mx-(w//2)
            bbox[1] = my-(h//2)
            mouse.editing_id = id

    return bbox


# Redefine posição do ponto da bounding box do objeto
def move_point(bbox, mouse, id, proximity=20):
    if mouse.editing_id is None or mouse.editing_id == id:
        mx, my = mouse.point
        x, y, w, h = bbox
        p = proximity
        if mx > x-p and mx < x+w-1+p and my > y-p and my < y+h-1+p:
            if abs(mx-x) <= proximity:
                w -= mx-x
                x = mx
            if abs(my-y) <= proximity:
                h -= my-y
                y = my
            if abs(mx-x-w) <= proximity:
                w = mx - x
            if abs(my-y-h) <= proximity:
                h = my - y

            if bbox != [x, y, w, h]:
                mouse.editing_id = id
                bbox = [x, y, w, h]

    return bbox


# Registra anotação de vídeo em arquivo JSON padrão
def main(src, imgfiles, max_objects, autocomplete=None):

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
    all_images = data['images']
    all_bboxes = data['bboxes']
    all_labels = data['labels']
    all_ids = data['ids']

    # Escrita
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 255)
    line = cv2.LINE_AA

    # Preset de cores
    obj_colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    # Variáveis auxiliares autocomplete
    if autocomplete:
        auto_bboxes = []
        auto_labels = []
        auto_ids = []

    # Cria janela
    cv2.namedWindow("frame")

    # Callback do mouse
    mouse = CallbackMouse(window="frame")

    # Processo de anotação por frame
    i, n = 0, 0
    while True:

        # Carrega dados da amostra
        frame_path = imgfiles[n]
        frame_bboxes = []
        frame_labels = []
        frame_ids = []

        # Atualiza amostras
        if frame_path in all_images:
            i = all_images.index(frame_path)
            frame_bboxes = all_bboxes[i]
            frame_labels = all_labels[i]
            frame_ids = all_ids[i]
        else:
            all_images.append(frame_path)
            all_bboxes.append(frame_bboxes)
            all_labels.append(frame_labels)
            all_ids.append(frame_ids)

        # Verifica se existem bounding boxes para autocompletar
        if autocomplete and frame_bboxes == []:
            frame_bboxes = auto_bboxes
            frame_labels = auto_labels
            frame_ids = auto_ids

        # Lê imagens
        frame = cv2.imread(os.path.join(src, frame_path))
        out = frame.copy()

        # Index da amostra
        i = all_images.index(frame_path)

        # Edita anotação com mouse
        mouse.shape = frame.shape[:2][::-1]
        mouse_status = [mouse.left_button_press, mouse.middle_button_press]
        if mouse.middle_button_press:
            frame_bboxes = [move_box(box, mouse, j) for j, box in enumerate(frame_bboxes)]
        if mouse.left_button_press:
            frame_bboxes = [move_point(box, mouse, j) for j, box in enumerate(frame_bboxes)]
        if True in mouse_status:
            all_bboxes[i] = frame_bboxes
        else:
            mouse.editing_id = None

        # Desenha objetos
        for bbox, label, id in zip(frame_bboxes, frame_labels, frame_ids):
            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[0]+bbox[2], bbox[1]+bbox[3])

            # Bounding box
            obj_color = obj_colors[id % len(obj_colors)]
            cv2.rectangle(out, pt1=pt1, pt2=pt2, color=obj_color)

            # Label
            cv2.rectangle(out, (pt1[0], pt1[1]-20), (pt2[0], pt1[1]), obj_color, -1)
            cv2.putText(out, label, (pt1[0], pt1[1]-10), font, font_scale, (0, 0, 0))

            # ID de referência do objeto
            cv2.rectangle(out, (pt2[0]-2-12*len(str(id)), pt1[1]-19), (pt2[0]-1, pt1[1]-1), (50, 50, 50), -1)
            cv2.putText(out, str(id), (pt2[0]-2-12*len(str(id)), pt1[1]-3), font, font_scale*1.2, (255, 255, 255))

        # Indicativos na tela
        cv2.putText(out, f"FRAME: {n+1}/{annotation['samples']}", (5, 25), font, font_scale*2, color, thickness, line)
        cv2.putText(out, "ESC - sair", (5, 45), font, font_scale, color, thickness, line)
        cv2.putText(out, "a - voltar", (5, 65), font, font_scale, color, thickness, line)
        cv2.putText(out, "d - avancar", (5, 85), font, font_scale, color, thickness, line)
        cv2.putText(out, "e - editar", (5, 105), font, font_scale, color, thickness, line)
        cv2.putText(out, "z - excluir", (5, 125), font, font_scale, color, thickness, line)

        # Mostra imagem
        cv2.imshow("frame", out)

        # Captura de teclas
        delay = 10 if True in mouse_status else 100
        key = cv2.waitKey(delay)
        if key == 27:
            update_annotation_file(filename, annotation, all_images, all_bboxes, all_labels, all_ids)
            break

        elif key == ord('d') and n < annotation['samples']:
            n += 1
            all_bboxes[i] = frame_bboxes
            all_labels[i] = frame_labels
            all_ids[i] = frame_ids
            # Salva dados do frame anterior
            if autocomplete:
                auto_bboxes = frame_bboxes
                auto_labels = frame_labels
                auto_ids = frame_ids

        elif key == ord('a') and n > 0:
            n -= 1
            all_bboxes[i] = frame_bboxes
            all_labels[i] = frame_labels
            all_ids[i] = frame_ids
            # Salva dados do frame anterior
            if autocomplete:
                auto_bboxes = frame_bboxes
                auto_labels = frame_labels
                auto_ids = frame_ids

        elif key == ord('z'):
            all_bboxes[i] = []
            all_labels[i] = []
            all_ids[i] = []
            if autocomplete:
                auto_bboxes = []
                auto_labels = []
                auto_ids = []
            print("Dados excluidos no frame:", frame_path)

        elif key == ord('e'):
            # Select ROI
            print()
            cv2.putText(frame, "Selecione a ROI...", (5, 15), font, font_scale, color, thickness, line)
            r = cv2.selectROI("frame", frame, showCrosshair=False)
            if r == (0, 0, 0, 0):
                print("Invalid crop!")
                continue

            # Crop image
            crop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            cv2.putText(crop, "ESC - cancelar", (5, 15), font, font_scale, color, thickness, line)
            cv2.putText(crop, "Enter - confirmar", (5, 35), font, font_scale, color, thickness, line)
            cv2.imshow("crop", crop)

            # Cria trackbars
            cv2.createTrackbar('label', 'crop', 0, len(LABELS)-1, nothing)
            cv2.createTrackbar('id', 'crop', 0, max_objects-1, nothing)

            while True:
                key = cv2.waitKey()
                if key == 27:
                    break
                elif key == 13:
                    # Atualiza valores de referência
                    all_bboxes[i].append(r)
                    all_labels[i].append(LABELS[int(cv2.getTrackbarPos('label', 'crop'))])
                    all_ids[i].append(int(cv2.getTrackbarPos('id', 'crop')))
                    print("Objeto adicionado no frame:", frame_path)
                    break

            cv2.destroyWindow('crop')

            # Novo callback do mouse
            mouse = CallbackMouse(window="frame")

        # Salva dados no arquivo JSON
        if key != -1 or not mouse.editing_id:
            update_annotation_file(
                filename, annotation,
                all_images, all_bboxes, all_labels, all_ids, msg=False)


if __name__ == '__main__':

    print(" Anotação de vídeo para MOT ".center(60, "*"))
    args = parse_args()
    mapfiles = map_images(src=args.images)
    main(src=args.images, imgfiles=mapfiles, max_objects=args.max_objects, autocomplete=args.autocomplete)

    # Objetos não ficando salvos na ordem definida
    # Objetos sumindo do nada