import os
import cv2
import argparse
from utils import create_folder, save_json, load_json


VERSION = "0.1.0"
LABELS = ['pessoa', 'robo']


# Argumentos de entrada
def parse_args():
    """ Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Annotation dataset")
    parser.add_argument("--images", type=str, required=True, help="Caminho das imagens, main folder.")
    parser.add_argument("--max_objects", type=int, default=10, help="Máx. número de objetos unicos para registrar.")

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

    return sorted(batch_files)


# Cria arquivo JSON padrão
def create_annotation_file(dst):

    annotation = {
        "source": None,  # fonte do dataset, apenas para conhecimento
        "version": VERSION,  # versão do código de anotação de vídeo
        "frames": 0,  # qtd. de amostras/imagens
        "objects": [],  # ids rastreados
        "data": {"images": [], "bboxes": [], "labels": [], "ids": []}  # dados de anotação relevantes
    }

    save_json(dst, annotation)

    return annotation


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
        self.left_double_click = False
        self.editing_id = None
        self.shape = shape
        cv2.setMouseCallback(window, self.mouse_events)

    def mouse_events(self, event, x, y, flags, param):

        if x < 0 or y < 0:
            self.left_button_press = False
            self.right_button_press = False
            self.middle_button_press = False
            self.editing_id = None
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

        self.left_double_click = True if event == cv2.EVENT_LBUTTONDBLCLK else False


# Redefine posição da bounding box do objeto
def move_box(bbox, mouse, id):
    if mouse.editing_id is None or mouse.editing_id == id:
        mx, my = mouse.point
        x, y, w, h = bbox
        if (mx > x and mx < x+w-1 and my > y and my < y+h-1) or mouse.editing_id is not None:
            bbox = tuple([mx-(w//2), my-(h//2), w, h])
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
                bbox = tuple([x, y, w, h])

    return bbox


# Verifica se ponteiro do mouse está sobre objeto e retorna o indexador do objeto
def get_object_index(bboxes, mouse):
    mx, my = mouse.point
    for i, (x, y, w, h) in enumerate(bboxes):
        if (mx > x and mx < x+w-1 and my > y and my < y+h-1):
            return i

    return -1


# Registra anotação de vídeo em arquivo JSON padrão
def main(src, imgfiles, max_objects):

    # Barra no final
    if src[-1] == "/":
        src = src[:-1]

    # Carrega ou cria arquivo de anotação
    dst_folder = os.path.join("data", os.path.basename(src))
    create_folder(dst_folder)
    filename = os.path.join(dst_folder, "annotation")
    annotation = load_json(filename)
    if annotation is None:
        annotation = create_annotation_file(filename)

    # Registra dataset fonte e info
    max_frames = len(imgfiles)
    annotation['source'] = src
    annotation['frames'] = max_frames

    # Preenche data conforme mapping do dataset
    data_annotation = annotation['data']
    data_images = data_annotation['images']
    data_bboxes = data_annotation['bboxes']
    data_labels = data_annotation['labels']
    data_ids = data_annotation['ids']
    fill_idx = [data_images.index(x) if x in data_images else None for x in imgfiles]
    data = {"images": imgfiles, "bboxes": [], "labels": [], "ids": []}
    for idx in fill_idx:
        if idx is not None:
            data['bboxes'].append(data_bboxes[idx])
            data['labels'].append(data_labels[idx])
            data['ids'].append(data_ids[idx])
        else:
            data['bboxes'].append([])
            data['labels'].append([])
            data['ids'].append([])

    # Escrita
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 255)
    color_active = (0, 0, 255)
    line = cv2.LINE_AA

    # Preset de cores
    obj_colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    # Cria janela
    cv2.namedWindow("frame")

    # Callback do mouse
    mouse = CallbackMouse(window="frame")

    # Processo de anotação por frame
    idx, idx_auto, c = 0, 0, 1

    # Completa anotação automaticamente
    autocomplete = False

    # Tecla pressionada
    key = -1

    while True:

        # Carrega dados da amostra
        frame_path = data['images'][idx]
        frame_bboxes = data['bboxes'][idx]
        frame_labels = data['labels'][idx]
        frame_ids = data['ids'][idx]

        # Verifica se existem bounding boxes para autocompletar
        if autocomplete and frame_bboxes == []:
            frame_bboxes = data['bboxes'][idx_auto]
            frame_labels = data['labels'][idx_auto]
            frame_ids = data['ids'][idx_auto]

        # Lê imagens
        frame = cv2.imread(os.path.join(src, frame_path))
        out = frame.copy()

        # Edita anotação com mouse
        mouse.shape = frame.shape[:2][::-1]
        mouse_status = [mouse.left_button_press, mouse.middle_button_press]
        if mouse.left_button_press:
            frame_bboxes = [move_point(box, mouse, id) for box, id in zip(frame_bboxes, frame_ids)]
        if mouse.middle_button_press:
            frame_bboxes = [move_box(box, mouse, id) for box, id in zip(frame_bboxes, frame_ids)]
        if True not in mouse_status:
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

        # Atualiza cores teclas/mouse
        color_next = color_active if key == ord('d') else color
        color_prev = color_active if key == ord('a') else color
        color_del = color_active if key == ord('z') else color
        color_autocomplete = color_active if autocomplete else color
        color_jumping = color_active if c > 1 else color
        color_mouse_left_clk = color_active if mouse.left_button_press else color
        color_mouse_middle_clk = color_active if mouse.middle_button_press else color

        # Indicativos na tela
        cv2.putText(out, f"FRAME: {idx+1}/{max_frames}", (4, 25), font, font_scale*2, color, thickness, line)
        cv2.putText(out, f"source: {frame_path}", (5, 50), font, font_scale, color, thickness, line)
        cv2.putText(out, "".center(20, "-"), (5, 70), font, font_scale, color, thickness, line)
        cv2.putText(out, "ESC - sair", (5, 90), font, font_scale, color, thickness, line)
        cv2.putText(out, "s - autocomplete", (5, 110), font, font_scale, color_autocomplete, thickness, line)
        cv2.putText(out, "d - avancar", (5, 130), font, font_scale, color_next, thickness, line)
        cv2.putText(out, "a - voltar", (5, 150), font, font_scale, color_prev, thickness, line)
        cv2.putText(out, "j - jumping", (5, 170), font, font_scale, color_jumping, thickness, line)
        cv2.putText(out, "e - editar", (5, 190), font, font_scale, color, thickness, line)
        cv2.putText(out, "z - excluir", (5, 210), font, font_scale, color_del, thickness, line)
        cv2.putText(out, "mouse left clk - adjust", (5, 230), font, font_scale, color_mouse_left_clk, thickness, line)
        cv2.putText(out, "mouse middle clk - move", (5, 250), font, font_scale, color_mouse_middle_clk, thickness, line)

        # Mostra imagem
        cv2.imshow("frame", out)

        # Atualiza amostra na anotação
        data['bboxes'][idx] = frame_bboxes
        data['labels'][idx] = frame_labels
        data['ids'][idx] = frame_ids

        # Captura de teclas
        delay = 10 if True in mouse_status else 100
        key = cv2.waitKey(delay)
        if key == 27:  # ESC
            annotation['data'] = data
            annotation['objects'] = list(set(x for ids in data['ids'] for x in ids))
            save_json(filename, annotation, indent=False, message=True)
            break

        elif key == ord('s'):
            autocomplete = not(autocomplete)

        elif key == ord('d') and idx+c < max_frames:
            idx_auto = idx if autocomplete and not c > 1 else idx+c
            idx += c

        elif key == ord('a') and idx-c >= 0:
            idx -= c
            idx_auto = idx

        elif key == ord('j'):
            c = 1 if c > 1 else 10

        elif key == ord('z'):
            obj_idx = get_object_index(frame_bboxes, mouse)
            if obj_idx == -1:
                data['bboxes'][idx] = []
                data['labels'][idx] = []
                data['ids'][idx] = []
            else:
                data['bboxes'][idx].pop(obj_idx)
                data['labels'][idx].pop(obj_idx)
                data['ids'][idx].pop(obj_idx)
            idx_auto = idx
            print("Dados excluidos no frame:", frame_path)

        if key == ord('e'):
            # Select ROI
            print()
            cv2.putText(
                frame, "Select a ROI and then press SPACE or ENTER button!",
                (5, 15), font, font_scale, color, thickness, line)
            cv2.putText(
                frame, "Cancel the selection process by pressing c button!",
                (5, 35), font, font_scale, color, thickness, line)
            r = cv2.selectROI("frame", frame, showCrosshair=False)
            if r != (0, 0, 0, 0):

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
                        data['bboxes'][idx].append(r)
                        data['labels'][idx].append(LABELS[int(cv2.getTrackbarPos('label', 'crop'))])
                        data['ids'][idx].append(int(cv2.getTrackbarPos('id', 'crop')))
                        print("Objeto adicionado no frame:", frame_path)
                        break

                cv2.destroyWindow('crop')

            # Novo callback do mouse
            mouse = CallbackMouse(window="frame")

        # Salva dados no arquivo JSON
        if key != -1:
            print("Dados salvos para o frame:", frame_path, end='\r')
            annotation['data'] = data
            save_json(filename, annotation, indent=False, message=False)


if __name__ == '__main__':

    print(" Anotação de vídeo para MOT ".center(60, "*"))
    args = parse_args()
    mapfiles = map_images(src=args.images)
    main(src=args.images, imgfiles=mapfiles, max_objects=args.max_objects)
