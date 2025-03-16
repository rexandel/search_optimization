from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import uic

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import sys


class Optimization3D(QGLWidget):
    def __init__(self, parent=None):
        glutInit()
        super().__init__(parent)

        self.initial_angle_x = -90
        self.initial_angle_y = 0
        self.initial_zoom_level = 35
        self.initial_pan_x = 0.0
        self.initial_pan_y = 0.0

        self.angle_x = self.initial_angle_x
        self.angle_y = self.initial_angle_y
        self.zoom_level = self.initial_zoom_level
        self.pan_x = self.initial_pan_x
        self.pan_y = self.initial_pan_y

        self.last_x = 0
        self.last_y = 0
        self.is_dragging = False
        self.is_panning = False

        self.grid_size = 10
        self.grid_step = 1

        self.show_grid = True
        self.show_axes = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def reset_view(self):
        self.angle_x = self.initial_angle_x
        self.angle_y = self.initial_angle_y
        self.zoom_level = self.initial_zoom_level
        self.pan_x = self.initial_pan_x
        self.pan_y = self.initial_pan_y
        self.update()

    def initializeGL(self):
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

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

    def resizeGL(self, w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        gluLookAt(0, 0, self.zoom_level, 0, 0, 0, 0, 1, 0)

        glTranslatef(self.pan_x, self.pan_y, 0)

        glRotatef(self.angle_x, 1, 0, 0)
        glRotatef(self.angle_y, 0, 1, 0)

        if self.show_grid:
            self.draw_grid()

        if self.show_axes:
            self.draw_axes()

    def draw_label(self, x, y, z, label, color=(0.0, 0.0, 0.0)):
        glDisable(GL_LIGHTING)
        glColor3f(*color)

        glPushMatrix()

        glTranslatef(x, y, z)

        glRotatef(-self.angle_y, 0, 1, 0)
        glRotatef(-self.angle_x, 1, 0, 0)

        size = 0.5
        if label == "X":
            self.draw_x(0, 0, 0, size)
        elif label == "Y":
            self.draw_y(0, 0, 0, size)
        elif label == "Z":
            self.draw_z(0, 0, 0, size)

        glPopMatrix()

        glEnable(GL_LIGHTING)

    def draw_x(self, x, y, z, size):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glEnd()

    def draw_y(self, x, y, z, size):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x, y, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x, y, z)
        glVertex3f(x, y, z)
        glVertex3f(x, y - size / 2, z)
        glEnd()

    def draw_z(self, x, y, z, size):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(x - size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x + size / 2, y + size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x - size / 2, y - size / 2, z)
        glVertex3f(x + size / 2, y - size / 2, z)
        glEnd()

    def draw_axes(self):
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

        self.draw_label(self.grid_size + offset, 0, 0, "X", (0, 0, 0))

        self.draw_label(0, self.grid_size + offset, 0, "Y", (0, 0, 0))

        self.draw_label(0, 0, self.grid_size + offset, "Z", (0, 0, 0))

        glLineWidth(1)

        glEnable(GL_LIGHTING)

    def draw_grid(self):
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
        self.last_x = event.x()
        self.last_y = event.y()

        if event.modifiers() & Qt.ControlModifier:
            self.is_panning = True
        else:
            self.is_dragging = True

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom_level -= delta
        self.zoom_level = max(5, min(self.zoom_level, 50))
        self.update()

    def mouseMoveEvent(self, event):
        dx, dy = event.x() - self.last_x, event.y() - self.last_y

        if self.is_dragging:
            self.angle_x += dy / 5
            self.angle_y += dx / 5
        elif self.is_panning:
            pan_speed = 0.001 * self.zoom_level
            self.pan_x += dx * pan_speed
            self.pan_y -= dy * pan_speed

        self.last_x, self.last_y = event.x(), event.y()
        self.update()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.is_panning = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('main_window.ui', self)

        self.gl_container = Optimization3D(self.centralwidget)

        self.gl_container.setGeometry(self.openGLWidget.geometry())

        self.openGLWidget.setParent(None)
        self.openGLWidget = self.gl_container

        self.gridVisibility.stateChanged.connect(self.toggle_grid_visibility)
        self.axisVisibility.stateChanged.connect(self.toggle_axis_visibility)
        self.returnButton.clicked.connect(self.reset_view)

    def toggle_grid_visibility(self, state):
        self.gl_container.show_grid = bool(state)
        self.gl_container.update()

    def toggle_axis_visibility(self, state):
        self.gl_container.show_axes = bool(state)
        self.gl_container.update()

    def reset_view(self):
        self.gl_container.reset_view()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
