from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt

from windows import FunctionManagerWindow
from optimization_methods import GradientDescent
from utils import LogEmitter, FunctionManagerHelper

import threading


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('gui/ui/main.ui', self)

        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.append_log_message)
        self.log_emitter.html_log_signal.connect(self.append_html_log_message)

        self.function_manager_helper = FunctionManagerHelper('functions.json')
        self.function_manager_helper.populate_combo_box(self.selectFunctionComboBox)
        self.selectFunctionComboBox.currentIndexChanged.connect(self.on_function_selected)

        self.gradient_descent = None
        self.optimization_thread = None

        if len(self.function_manager_helper.get_function_names()) > 0:
            current_func = self.function_manager_helper.get_current_function()
            self.openGLWidget.set_function(current_func['function'])
        else:
            self.statusbar.showMessage("No functions available")

        self.gridVisibility.stateChanged.connect(self.toggle_grid_visibility)
        self.axisVisibility.stateChanged.connect(self.toggle_axes_visibility)
        self.returnButton.clicked.connect(self.reset_view_to_default)
        self.functionManagerButton.clicked.connect(self.show_function_manager_window)
        self.startButton.clicked.connect(self.on_start_button_clicked)
        self.stopButton.clicked.connect(self.on_stop_button_clicked)

        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.startButton.isEnabled():
                self.on_start_button_clicked()
        else:
            super().keyPressEvent(event)

    @QtCore.pyqtSlot(str)
    def append_log_message(self, message: str):
        self.logEventPlainTextEdit.appendPlainText(message)

    @QtCore.pyqtSlot(str)
    def append_html_log_message(self, html: str):
        cursor = self.logEventPlainTextEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertHtml(html)
        self.logEventPlainTextEdit.setTextCursor(cursor)
        self.logEventPlainTextEdit.ensureCursorVisible()

    def toggle_grid_visibility(self, state):
        self.openGLWidget.grid_visible = bool(state)
        self.openGLWidget.update()

    def toggle_axes_visibility(self, state):
        self.openGLWidget.axes_visible = bool(state)
        self.openGLWidget.update()

    def reset_view_to_default(self):
        self.openGLWidget.restore_default_view()

    def on_function_selected(self, index):
        self.function_manager_helper.set_current_function(index)
        current_func = self.function_manager_helper.get_current_function()

        if current_func:
            self.openGLWidget.set_function(current_func['function'])
            formula_text = current_func['formula']
            self.statusbar.showMessage(f"Selected function: {current_func['name']} - {formula_text}")
        else:
            self.openGLWidget.set_function(None)
            self.statusbar.showMessage("No function selected")

    def show_function_manager_window(self):
        function_manager_window = FunctionManagerWindow(self)
        function_manager_window.exec_()

    def on_start_button_clicked(self):
        data = {
            'x': self.xLineEdit.text(),
            'y': self.yLineEdit.text(),
            'first_eps': self.firstEpsLineEdit.text(),
            'sec_eps': self.secondEpsLineEdit.text(),
            'third_eps': self.thirdEpsLineEdit.text(),
            'initial_step': self.initialStepLineEdit.text(),
            'num_iter': self.numIterLineEdit.text()
        }

        try:
            x = float(data['x'])
            y = float(data['y'])
        except ValueError:
            self.statusbar.showMessage("Error: X and Y must be numeric values")
            return

        try:
            first_eps = float(data['first_eps'])
            sec_eps = float(data['sec_eps'])
            third_eps = float(data['third_eps'])

            if not (0 < first_eps < 1) or not (0 < sec_eps < 1) or not (0 < third_eps < 1):
                self.statusbar.showMessage("Error: All epsilon values must be between 0 and 1")
                return
        except ValueError:
            self.statusbar.showMessage("Error: Epsilon parameters must be floating-point numbers")
            return

        try:
            initial_step = float(data['initial_step'])
            if initial_step <= 0:
                self.statusbar.showMessage("Error: Initial step must be a positive integer")
                return
        except ValueError:
            self.statusbar.showMessage("Error: Initial step must be a positive integer")
            return

        try:
            num_iter = int(data['num_iter'])
            if num_iter <= 0:
                self.statusbar.showMessage("Error: Number of iterations must be a positive integer")
                return
        except ValueError:
            self.statusbar.showMessage("Error: Number of iterations must be a positive integer")
            return

        self.logEventPlainTextEdit.clear()
        self.tabWidget.setEnabled(False)
        self.selectFunctionComboBox.setEnabled(False)
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)

        params = {
            'point': (x, y),
            'epsilons': (first_eps, sec_eps, third_eps),
            'initial_step': initial_step,
            'max_iterations': num_iter,
            'function': self.function_manager_helper.get_current_function()['function']
        }

        self.gradient_descent = GradientDescent(params, self.log_emitter)
        self.gradient_descent.finished_signal.connect(self.on_optimization_finished)

        self.optimization_thread = threading.Thread(target=self.gradient_descent.run, daemon=True)
        self.optimization_thread.start()

        self.gradient_descent.update_signal.connect(self.openGLWidget.update_optimization_path)
        self.statusbar.showMessage("Optimization started")

    @QtCore.pyqtSlot()
    def on_optimization_finished(self):
        self.tabWidget.setEnabled(True)
        self.selectFunctionComboBox.setEnabled(True)
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.append_log_message("Optimization finished")

    def on_stop_button_clicked(self):
        if self.gradient_descent:
            self.gradient_descent.stop()
            self.stopButton.setEnabled(False)
            self.startButton.setEnabled(True)
