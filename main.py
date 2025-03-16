import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import QTimer
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
from PyQt5 import uic


class Optimization3D(QGLWidget):
    def __init__(self, parent=None):
        glutInit()
        super().__init__(parent)
        self.angle_x = -90
        self.angle_y = 0
        self.last_x = 0
        self.last_y = 0
        self.zoom_level = 35
        self.is_dragging = False

        self.grid_size = 10
        self.grid_step = 1

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

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

        glRotatef(self.angle_x, 1, 0, 0)
        glRotatef(self.angle_y, 0, 1, 0)

        self.draw_grid()

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
        self.is_dragging = True
        self.last_x = event.x()
        self.last_y = event.y()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom_level -= delta
        self.zoom_level = max(5, min(self.zoom_level, 50))
        self.update()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            dx, dy = event.x() - self.last_x, event.y() - self.last_y
            self.angle_x += dy / 5
            self.angle_y += dx / 5
            self.last_x, self.last_y = event.x(), event.y()
            self.update()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('main_window.ui', self)

        self.gl_container = Optimization3D(self.centralwidget)

        self.gl_container.setGeometry(self.openGLWidget.geometry())

        self.openGLWidget.setParent(None)
        self.openGLWidget = self.gl_container


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
