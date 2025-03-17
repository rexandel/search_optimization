from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import QTimer, Qt

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np

class Visualization3DWidget(QGLWidget):
    def __init__(self, parent=None):
        glutInit()
        super().__init__(parent)

        self.default_rotation_x = -90
        self.default_rotation_y = 0
        self.default_zoom_level = 35
        self.default_position_x = 0.0
        self.default_position_y = 0.0

        self.rotation_x = self.default_rotation_x
        self.rotation_y = self.default_rotation_y
        self.zoom_level = self.default_zoom_level
        self.position_x = self.default_position_x
        self.position_y = self.default_position_y

        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.is_rotating = False
        self.is_moving = False

        self.grid_size = 10
        self.grid_step = 1

        self.grid_visible = True
        self.axes_visible = True

        self.current_function = None
        self.objective_function_data = None
        self.display_lists = {}
        self.z_min = 0
        self.z_max = 0

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(16)

    def restore_default_view(self):
        self.rotation_x = self.default_rotation_x
        self.rotation_y = self.default_rotation_y
        self.zoom_level = self.default_zoom_level
        self.position_x = self.default_position_x
        self.position_y = self.default_position_y
        self.update()

    def initializeGL(self):
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        light_position = [10.0, 10.0, 10.0, 1.0]
        light_ambient = [0.2, 0.2, 0.2, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]

        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width() / self.height(), 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def resizeGL(self, width, height):
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        gluLookAt(0, 0, self.zoom_level, 0, 0, 0, 0, 1, 0)

        glTranslatef(self.position_x, self.position_y, 0)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_POLYGON_SMOOTH)
        glDisable(GL_LINE_SMOOTH)

        if self.grid_visible:
            self.render_grid()

        if self.axes_visible:
            self.render_axes()

        if self.current_function and self.objective_function_data:
            if 'function' not in self.display_lists:
                self.display_lists['function'] = self.create_function_display_list()
            glCallList(self.display_lists['function'])

    def create_function_display_list(self):
        display_list = glGenLists(1)
        glNewList(display_list, GL_COMPILE)

        if self.objective_function_data:
            for strip in self.objective_function_data:
                glBegin(GL_QUAD_STRIP)
                for vertex, color in strip:
                    glColor3f(*color)
                    glVertex3f(*vertex)
                glEnd()

        glEndList()
        return display_list

    def build_objective_function_data(self):
        if self.current_function is None:
            return

        x_values = np.linspace(-self.grid_size, self.grid_size, 200)
        y_values = np.linspace(-self.grid_size, self.grid_size, 200)

        z_values = np.zeros((len(x_values), len(y_values)))

        for i in range(len(x_values)):
            for j in range(len(y_values)):
                z_values[i, j] = self.current_function(x_values[i], y_values[j])

        self.z_min = np.min(z_values)
        self.z_max = np.max(z_values)

        shadow_strength = 0.6
        self.objective_function_data = []

        for i in range(len(x_values) - 1):
            strip = []
            for j in range(len(y_values)):
                x1 = x_values[i]
                x2 = x_values[i + 1]
                y = y_values[j]
                z1 = z_values[i, j]
                z2 = z_values[i + 1, j]

                z1_norm = (z1 - self.z_min) / (self.z_max - self.z_min) * 2 * self.grid_size - self.grid_size
                z2_norm = (z2 - self.z_min) / (self.z_max - self.z_min) * 2 * self.grid_size - self.grid_size

                z1_shadow = ((z1 - self.z_min) / (self.z_max - self.z_min)) ** 0.5
                z2_shadow = ((z2 - self.z_min) / (self.z_max - self.z_min)) ** 0.5
                shadow_intensity1 = 1.0 - shadow_strength * (1.0 - z1_shadow)
                shadow_intensity2 = 1.0 - shadow_strength * (1.0 - z2_shadow)

                color1 = (((x1 + self.grid_size) / (2 * self.grid_size)) * shadow_intensity1,
                          ((y + self.grid_size) / (2 * self.grid_size)) * shadow_intensity1,
                          0.7 * shadow_intensity1)
                color2 = (((x2 + self.grid_size) / (2 * self.grid_size)) * shadow_intensity2,
                          ((y + self.grid_size) / (2 * self.grid_size)) * shadow_intensity2,
                          0.7 * shadow_intensity2)

                strip.append(((x1, y, z1_norm), color1))
                strip.append(((x2, y, z2_norm), color2))
            self.objective_function_data.append(strip)

        if 'function' in self.display_lists:
            glDeleteLists(self.display_lists['function'], 1)
            self.display_lists.pop('function')

    def set_function(self, func):
        self.current_function = func
        if func is not None:
            self.build_objective_function_data()
        self.update()

    def render_axis_label(self, x, y, z, label, color=(0.0, 0.0, 0.0)):
        glDisable(GL_LIGHTING)
        glColor3f(*color)

        glPushMatrix()

        glTranslatef(x, y, z)

        glRotatef(-self.rotation_y, 0, 1, 0)
        glRotatef(-self.rotation_x, 1, 0, 0)

        size = 0.5
        if label == "X":
            self.render_x_symbol(0, 0, 0, size)
        elif label == "Y":
            self.render_y_symbol(0, 0, 0, size)
        elif label == "Z":
            self.render_z_symbol(0, 0, 0, size)

        glPopMatrix()

        glEnable(GL_LIGHTING)

    def render_x_symbol(self, x, y, z, size):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glEnd()

    def render_y_symbol(self, x, y, z, size):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x, y, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x, y, z)
        glVertex3f(x, y, z)
        glVertex3f(x, y - size / 2, z)
        glEnd()

    def render_z_symbol(self, x, y, z, size):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glEnd()

    def render_axes(self):
        glDisable(GL_LIGHTING)

        glLineWidth(2)
        glBegin(GL_LINES)

        glColor3f(1, 0, 0)
        glVertex3f(-self.grid_size, 0, 0)
        glVertex3f(self.grid_size, 0, 0)

        glColor3f(0, 1, 0)
        glVertex3f(0, -self.grid_size, 0)
        glVertex3f(0, self.grid_size, 0)

        glColor3f(0, 0, 1)
        glVertex3f(0, 0, -self.grid_size)
        glVertex3f(0, 0, self.grid_size)
        glEnd()

        offset = 0.7

        self.render_axis_label(self.grid_size + offset, 0, 0, "X", (0, 0, 0))

        self.render_axis_label(0, self.grid_size + offset, 0, "Y", (0, 0, 0))

        self.render_axis_label(0, 0, self.grid_size + offset, "Z", (0, 0, 0))

        glLineWidth(1)

        glEnable(GL_LIGHTING)

    def render_grid(self):
        glLineWidth(1)
        glColor3f(0.7, 0.7, 0.7)

        z_position = -self.grid_size

        for i in range(-self.grid_size, self.grid_size + 1, self.grid_step):
            glBegin(GL_LINES)
            glVertex3f(i, -self.grid_size, z_position)
            glVertex3f(i, self.grid_size, z_position)
            glEnd()

        for i in range(-self.grid_size, self.grid_size + 1, self.grid_step):
            glBegin(GL_LINES)
            glVertex3f(-self.grid_size, i, z_position)
            glVertex3f(self.grid_size, i, z_position)
            glEnd()

    def mousePressEvent(self, event):
        self.mouse_last_x = event.x()
        self.mouse_last_y = event.y()

        if event.modifiers() & Qt.ControlModifier:
            self.is_moving = True
        else:
            self.is_rotating = True

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom_level -= delta
        self.zoom_level = max(5, min(self.zoom_level, 50))
        self.update()

    def mouseMoveEvent(self, event):
        dx, dy = event.x() - self.mouse_last_x, event.y() - self.mouse_last_y

        if self.is_rotating:
            self.rotation_x += dy / 5
            self.rotation_y += dx / 5
        elif self.is_moving:
            movement_speed = 0.001 * self.zoom_level
            self.position_x += dx * movement_speed
            self.position_y -= dy * movement_speed

        self.mouse_last_x, self.mouse_last_y = event.x(), event.y()
        self.update()

    def mouseReleaseEvent(self, event):
        self.is_rotating = False
        self.is_moving = False
