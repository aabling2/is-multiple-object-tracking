import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox


# Complementary function to make the 3D axes aspect equal
def set_axes_equal(ax):
    # Make axes of 3D plot have equal scale so that spheres appear as spheres,
    # cubes as cubes, etc..  This is one possible solution to Matplotlib's
    # ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    # Input
    #  ax: a matplotlib axis, e.g., as output from plt.gca().

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# Transformações geométricas
class Transformations():
    def __init__(self):
        self.eye = np.eye(4)

    def z_rotation(self, angle):
        angle = angle * pi/180
        rotation_matrix = np.array([
            [cos(angle), -sin(angle), 0, 0],
            [sin(angle), cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        return rotation_matrix

    def x_rotation(self, angle):
        angle = angle * pi/180
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, cos(angle), -sin(angle), 0],
            [0, sin(angle), cos(angle), 0],
            [0, 0, 0, 1]])
        return rotation_matrix

    def y_rotation(self, angle):
        angle = angle * pi/180
        rotation_matrix = np.array([
            [cos(angle), 0, sin(angle), 0],
            [0, 1, 0, 0],
            [-sin(angle), 0, cos(angle), 0],
            [0, 0, 0, 1]])
        return rotation_matrix

    def translation(self, dx, dy, dz):
        translation_matrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]])
        return translation_matrix

    def apply(self, obj, move_coord=[0., 0., 0.], angles=[0., 0., 0.]):
        dx, dy, dz = move_coord
        angx, angy, angz = angles

        T = self.translation(dx, dy, dz)

        if angx != 0:
            R = self.x_rotation(angx)
            M = np.dot(R, T)
            obj = np.dot(R, obj)
        elif angy != 0:
            R = self.y_rotation(angy)
            M = np.dot(R, T)
            obj = np.dot(M, obj)
        elif angz != 0:
            R = self.z_rotation(angz)
            M = np.dot(R, T)
            obj = np.dot(M, obj)
        elif angx == 0 and angy == 0 and angz == 0:
            obj = np.dot(T, obj)

        return obj


# Camera pin-hole com matrizes de cálculo de projeção para imagem
class CameraProjection(object):
    def __init__(self, sensor_size=np.array([1280, 720]), focal=1.,
                 cam_center=np.array([1280//2, 720//2]),
                 cam_pos=np.array([0., 0., 0.]),
                 cam_rot=np.array([0., 0., 0.])):

        self.cam_pos = cam_pos
        self.cam_rot = cam_rot
        self.focal = focal  # Distância focal
        self.sx, self.sy = sensor_size  # Escala da projeção por eixo
        self.cam_center = cam_center  # Centro óptico no plano de projeção
        self.M = Transformations()  # Transformações geométricas

    # Atualiza matrizes e calcula projeção dos pontos
    def calc_projection(self, points):
        # Matriz de parâmetros intrínsecos
        Kf = np.array([
            [self.focal, 0, 0],
            [0, self.focal, 0],
            [0, 0, 1]])
        Ks = np.array([
            [self.sx, 0, self.cam_center[0]],
            [0, self.sy, self.cam_center[1]],
            [0, 0, 1]])
        K = np.dot(Ks, Kf)

        # Matriz de projeção canônica
        Pi = np.append(np.eye(3), np.zeros((3, 1)), axis=1)

        # Matriz de parâmetros extrínsecos
        R = np.dot(np.dot(
            self.M.z_rotation(self.cam_rot[2]),
            self.M.y_rotation(self.cam_rot[1])),
            self.M.x_rotation(self.cam_rot[0]))
        t = self.M.translation(
            self.cam_pos[0],
            self.cam_pos[1],
            self.cam_pos[2])
        Rt = np.dot(R, t)

        # Matriz calibrada de transformação dos pontos
        P = np.dot(K, np.dot(Pi, Rt))

        # Faz projeção geométrica
        x = np.dot(P, points)

        # Ajusta pontos para a imagem 2D
        x = self.process_points(x)

        return x

    # Processa pontos antes de plotar na imagem
    def process_points(self, points):
        # Pontos onde z é negativo são descartados, atraś da câmera
        points = points[:, points[2] >= 0]

        # Contorna problema da divisão por zero em z
        # Gera linhas falsas na plotagem se manter
        points[2, points[2] == 0] = 1.e-20

        # Traz coordenadas para o plano 2D dividindo pelo z
        # para ficar igual a 1
        for i in range(3):
            points[i] /= points[2]

        # Desloca centro da câmera/imagem
        # Origem fica no canto superior esquerdo
        xs, ys = points[:2]
        ys = ys*-1 + self.sy  # Inverte pontos por causa do eixo invertido

        # Junta novamente os pontos para forma imagem
        points = np.stack((xs, ys), axis=0)

        return points


# Carrega objeto para inserir no espaço 3D
class SceneObject():
    def __init__(self, size_vector=1):
        # Pontos para origem e ponta do vetor, eixo rgb(xyz)
        self.origin = np.array([0., 0., 0., 1.])
        self.vector = np.append(np.eye(3)*size_vector, np.ones((1, 3)), axis=0)

        # Escala do objeto
        scale = 5

        # Criar array de pontos interligados
        points = np.array([
            [-1, 1, 0], [1, 1, 0], [1, 1, 1], [1, -1, 1],
            [1, -1, 0], [-1, -1, 0], [-1, -1, 1], [1, -1, 1],
            [1, -1, 0], [1, 1, 0], [-1, 1, 1], [1, 1, 1],
            [-1, 1, 0], [-1, 1, 1], [-1, -1, 1], [-1, -1, 0],
            [-1, 1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0],
            [0, -1, 0], [-1, 0, 0]]).T*scale

        # Cria objeto 3D na estrutura de coordenadas homogêneas
        self.obj = np.vstack((points, np.ones(points.shape[1])))


# Faz calculos de projeção e plota na figura
class ProjectionPlots():
    def __init__(self, model=None, cam=None, cam_proj=None):
        # Transformações geométricas
        self.transforms = Transformations()

        self.mod_copy = copy.deepcopy(model)
        self.cam_copy = copy.deepcopy(cam)
        self.mod = model
        self.cam = cam
        self.cam_proj = cam_proj
        self.sel = "mod"
        self.ref = "world"
        self.move_coords = [0., 0., 0.]
        self.rotate_coords = [0., 0., 0.]

        # Posiciona objetos fora da origem do mundo
        self.cam_proj.cam_pos -= [0., 40., -60.]
        self.cam.obj = self.transform_obj(self.cam.obj, [0., 40., -60.])
        self.cam.origin = self.transform_obj(self.cam.origin, [0., 40., -60.])
        self.cam.vector = self.transform_obj(self.cam.vector, [0., 40., -60.])
        self.mod.obj = self.transform_obj(self.mod.obj, [0., 0., 50.])
        self.mod.origin = self.transform_obj(self.mod.origin, [0., 0., 50.])
        self.mod.vector = self.transform_obj(self.mod.vector, [0., 0., 50.])

        # Cria figura para configurações
        self.fig1 = plt.figure(figsize=[5, 4])

        # Cria callback do mouse, para pegar retorno do click nos sliders
        self.fig1.canvas.mpl_connect('button_release_event', self._onrelease)

        # Cria caixas de texto para inserir escalas
        self.text_sx = self._create_text_box(
            "scale x", initial=self.cam_proj.sx,
            start=.13, end=.1, pos=.93, function=self._get_sx)
        self.text_sy = self._create_text_box(
            "scale y", initial=self.cam_proj.sy,
            start=.13, end=.1, pos=.88, function=self._get_sy)
        self.text_centerx = self._create_text_box(
            "principal x", initial=self.cam_proj.cam_center[0],
            start=.45, end=.1, pos=.93, function=self._get_centerx)
        self.text_centery = self._create_text_box(
            "principal y", initial=self.cam_proj.cam_center[1],
            start=.45, end=.1, pos=.88, function=self._get_centery)

        # Cria seletores de referência da origem
        ax_radios1 = plt.axes([0.62, 0.53, 0.25, 0.2], facecolor='lightgrey')
        self.radios1 = RadioButtons(
            ax_radios1, ('ref. object', 'ref. world'), active=1)
        self.radios1.on_clicked(self._select_ref)

        # Cria seletores de objeto, modelo/câmera
        ax_radios2 = plt.axes([0.3, 0.53, 0.25, 0.2], facecolor='lightgrey')
        self.radios2 = RadioButtons(
            ax_radios2, ('obj. model', 'obj. camera'), active=0)
        self.radios2.on_clicked(self._select_object)

        # Cria slide bar para ajuste de distância focal, zoom
        self.slider_focal = self._create_slider(
            pos=.43, start=.13, end=.1, height=.36, valstep=0.1,
            name='focal dist.', orientation='vertical', valinit=1.,
            valmin=1., valmax=50., function=self._get_focal)

        # Cria slide bars para mover e rotacionar objetos
        move_vals = dict(valmin=-100, valmax=100, function=self._get_sliders)
        ang_vals = dict(valmin=-360, valmax=360, function=self._get_sliders)
        self.slider_x_mov = self._create_slider(
            pos=0.31, name="move-x", **move_vals)
        self.slider_y_mov = self._create_slider(
            pos=0.255, name="move-y", **move_vals)
        self.slider_z_mov = self._create_slider(
            pos=0.2, name="move-z", **move_vals)
        self.slider_x_ang = self._create_slider(
            pos=0.13, name="rotate-x", **ang_vals)
        self.slider_y_ang = self._create_slider(
            pos=0.075, name="rotate-y", **ang_vals)
        self.slider_z_ang = self._create_slider(
            pos=0.02, name="rotate-z", **ang_vals)

        # Cria botão de reset, volta ao estado inicial
        ax_reset = plt.axes([0.70, 0.9, 0.1, 0.06])
        self.button = Button(
            ax_reset, 'Reset', color='lightgrey', hovercolor='0.975')
        self.button.on_clicked(self._reset_all)

        # Cria figura template para as plotagens
        self.fig2 = plt.figure(figsize=[9, 4])
        # Cria plotagem 3D para os objetos
        self.axes0 = plt.axes([0.01, 0.2, 0.4, 0.7], projection="3d")
        # Cria plotagem 2D para a projeção da câmera
        self.axes1 = plt.axes([0.5, 0.25, 0.45, 0.6])

    # Cria slide bar
    def _create_slider(self, pos=0.2, start=.15, end=.75, height=.05,
                       name="slider", color='lightgrey',
                       valmin=-100., valmax=100., valstep=1.,
                       valinit=0., orientation='horizontal',
                       function=None):
        ax = plt.axes([start, pos, end, height], facecolor=color)
        slider = Slider(ax, name, valmin=valmin, valmax=valmax,
                        valinit=valinit, valstep=valstep,
                        orientation=orientation)
        slider.on_changed(function)
        return slider

    # Cria caixa de texto editável
    def _create_text_box(self, name="box", initial=1, color='white',
                         start=0.05, end=.07, pos=.5, function=None):
        ax = plt.axes([start, pos, end, 0.05], facecolor=color)
        box = TextBox(ax, label=name, initial=initial, color=color)
        box.on_text_change(function)
        return box

    # Cria circulos para origem
    def _set_origin_circle(self, xy=True, yz=True, zx=True, color='c'):
        theta = np.linspace(0, 2 * np.pi, 201)
        x = 0 * np.cos(theta)
        y = 5 * np.cos(theta)
        z = 5 * np.sin(theta)
        if xy:
            self.axes0.plot(z, x, y, color)
        if yz:
            self.axes0.plot(y, z, x, color)
        if zx:
            self.axes0.plot(x, y, z, color)

    # Cria setas para referência dos eixos, rgb(xyz)
    # y e z invertidos para representação comum
    def _set_arrows_orientation(self, origin=np.array([0., 0., 0.]),
                                vector=np.append(np.eye(3)*15, np.ones((1, 3)),
                                axis=0)):
        x, y, z = origin
        xv, yv, zv = vector[:-1, 0]
        xs, ys, zs = xv-x, yv-y, zv-z
        self.axes0.quiver(
            x, z, y,
            xs, zs, ys,
            color="red", alpha=1., lw=1
        )
        xv, yv, zv = vector[:-1, 1]
        xs, ys, zs = xv-x, yv-y, zv-z
        self.axes0.quiver(
            x, z, y,
            xs, zs, ys,
            color="green", alpha=1., lw=1
        )
        xv, yv, zv = vector[:-1, 2]
        xs, ys, zs = xv-x, yv-y, zv-z
        self.axes0.quiver(
            x, z, y,
            xs, zs, ys,
            color="blue", alpha=1., lw=1
        )

    # Seleção do objeto
    def _select_object(self, label):
        if label == "obj. model":
            self.sel = "mod"
        elif label == "obj. camera":
            self.sel = "cam"

    # Seleção da referência
    def _select_ref(self, label):
        if label == "ref. object":
            self.ref = "object"
        elif label == "ref. world":
            self.ref = "world"

    # Atualiza valores de translação e rotação
    def _get_sliders(self, val):
        xm = self.slider_x_mov.val
        ym = self.slider_y_mov.val
        zm = self.slider_z_mov.val
        xa = self.slider_x_ang.val
        ya = self.slider_y_ang.val
        za = self.slider_z_ang.val

        self.move_coords = (xm, ym, zm)
        self.rotate_coords = (xa, ya, za)

    # Atualiza valor de distância focal
    def _get_focal(self, val):
        self.cam_proj.focal = self.slider_focal.val
        self.update_plot()

    # Atualiza escala em x
    def _get_sx(self, txt):
        if txt.isdigit():
            self.cam_proj.sx = float(txt)
            self.update_plot()

    # Atualiza escala em y
    def _get_sy(self, txt):
        if txt.isdigit():
            self.cam_proj.sy = float(txt)
            self.update_plot()

    # Atualiza ponto principal em x
    def _get_centerx(self, txt):
        if txt.isdigit():
            self.cam_proj.cam_center[0] = float(txt)
            self.update_plot()

    # Atualiza ponto principal em y
    def _get_centery(self, txt):
        if txt.isdigit():
            self.cam_proj.cam_center[1] = float(txt)
            self.update_plot()

    # Reseta slide bars de translação e rotação
    def _reset_sliders(self, event):
        self.slider_x_mov.reset()
        self.slider_y_mov.reset()
        self.slider_z_mov.reset()
        self.slider_x_ang.reset()
        self.slider_y_ang.reset()
        self.slider_z_ang.reset()

    # Reseta tudo para estado inicial
    def _reset_all(self, event):
        self._reset_sliders(None)
        self.slider_focal.reset()
        self.mod = copy.deepcopy(self.mod_copy)
        self.cam = copy.deepcopy(self.cam_copy)
        self.move_coords = [0., 0., 0.]
        self.rotate_coords = [0., 0., 0.]
        self.cam_proj.cam_pos = np.array([0., 0., 0.])
        self.cam_proj.cam_rot = np.array([0., 0., 0.])

        # Retorna posição dos objetos na cena
        self.cam_proj.cam_pos -= [0., 40., -60.]
        self.cam.obj = self.transform_obj(self.cam.obj, [0., 40., -60.])
        self.cam.origin = self.transform_obj(self.cam.origin, [0., 40., -60.])
        self.cam.vector = self.transform_obj(self.cam.vector, [0., 40., -60.])
        self.mod.obj = self.transform_obj(self.mod.obj, [0., 0., 50.])
        self.mod.origin = self.transform_obj(self.mod.origin, [0., 0., 50.])
        self.mod.vector = self.transform_obj(self.mod.vector, [0., 0., 50.])

        self.update_plot()

    # Faz transformações assim que click do mouse é liberado
    def _onrelease(self, event):
        apply = False
        for ms, rs in zip(self.move_coords, self.rotate_coords):
            if ms != 0. or rs != 0:
                apply = True
                break
            else:
                continue

        if apply:
            move, rot = self.move_coords, self.rotate_coords
            # Quando objeto é o modelo
            if self.sel == "mod":
                self.mod.obj = self.transform_obj(self.mod.obj, move, rot)
                self.mod.origin = self.transform_obj(
                    self.mod.origin, move, rot)
                self.mod.vector = self.transform_obj(
                    self.mod.vector, move, rot)

            # Quando objeto é a câmera
            elif self.sel == "cam":
                self.cam_proj.cam_pos -= move
                self.cam_proj.cam_rot -= rot
                self.cam.obj = self.transform_obj(self.cam.obj, move, rot)
                self.cam.origin = self.transform_obj(
                    self.cam.origin, move, rot)
                self.cam.vector = self.transform_obj(
                    self.cam.vector, move, rot)

        self.update_plot()
        self._reset_sliders(None)

    # Transformação do objeto/pontos passados
    def transform_obj(self, obj, move=[0., 0., 0.], rotate=[0., 0., 0.]):
        # Altera referência conforme seleção
        origin = self.mod.origin if self.sel == "mod" else self.cam.origin

        # Leva até origem do mundo se ref é no objeto
        if self.ref == "object":
            obj = self.transforms.apply(
                obj=obj,
                move_coord=origin[:-1]*-1)

        # Aplica transformações
        obj = self.transforms.apply(
            obj=obj,
            move_coord=move,
            angles=rotate)

        # Tras da origem do mundo se ref é no objeto
        if self.ref == "object":
            obj = self.transforms.apply(
                obj=obj,
                move_coord=origin[:-1])

        return obj

    # Atualiza plotagens
    def update_plot(self):
        # Limpa plotagens anterioriores
        self.axes0.clear()
        self.axes1.clear()

        # Configura eixos
        self.axes0.set_xlabel('X')
        self.axes0.set_ylabel('Z')  # invertido
        self.axes0.set_zlabel('Y')  # invertido
        self.axes1.set_xlim([0, self.cam_proj.sx])
        self.axes1.set_ylim([0, self.cam_proj.sy])
        self.axes1.invert_yaxis()

        # Representa origens do mundo e objetos
        self._set_origin_circle()
        self._set_arrows_orientation()
        self._set_arrows_orientation(self.mod.origin[:-1], self.mod.vector)
        self._set_arrows_orientation(self.cam.origin[:-1], self.cam.vector)

        # Representa objetos
        mod = self.mod.obj
        self.axes0.plot(mod[0, :], mod[2, :], mod[1, :], 'm', lw=.5)
        cam = self.cam.obj
        self.axes0.plot(cam[0, :], cam[2, :], cam[1, :], 'k', lw=.5)

        # Representa projeção da câmera
        x = self.cam_proj.calc_projection(mod)
        center = self.cam_proj.cam_center
        self.axes1.plot(x[0], x[1], 'm', lw=.5)
        self.axes1.plot(center[0], center[1], 'b+', lw=.5)

        # Atualiza aspecto do objeto
        set_axes_equal(self.axes0)

        self.fig2.canvas.draw()

    # Plota imagem, necessário apenas no inicio
    def show(self):
        plt.show()


if __name__ == "__main__":

    # Pega argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--object', type=str, default=None, help='Caminho do objeto STL.')
    args = parser.parse_args()

    # Transformações e projeção da câmera
    cam_proj = CameraProjection()

    # Objetos 3D da cena, câmera e estrutura mesh
    cam = SceneObject()
    model = SceneObject()

    # Inicia plotagens e transformações, funciona com callbacks
    plots = ProjectionPlots(model, cam, cam_proj)
    plots.update_plot()
    plots.show()
