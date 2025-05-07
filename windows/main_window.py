from PyQt5.QtWidgets import QMainWindow, QLineEdit
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt

from windows import FunctionManagerWindow
from windows.work_log_window import WorkLogWindow

from optimization_methods import GradientDescent, LibrarySimplexMethod, MySimplexMethod
from utils import LogEmitter, FunctionManagerHelper

import numpy as np
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
        self.simplex_method = None
        self.optimization_thread = None

        if len(self.function_manager_helper.get_function_names()) > 0:
            self.on_function_selected(0)
        else:
            self.statusbar.showMessage("No functions available")

        self.current_tab_line_edits = []
        self.update_current_tab_line_edits()
        self.tabWidget.currentChanged.connect(self.on_tab_changed)

        self.gridVisibility.stateChanged.connect(self.toggle_grid_visibility)
        self.axisVisibility.stateChanged.connect(self.toggle_axes_visibility)
        self.returnButton.clicked.connect(self.reset_view_to_default)
        self.functionManagerButton.clicked.connect(self.show_function_manager_window)
        self.startButton.clicked.connect(self.on_start_button_clicked)
        self.stopButton.clicked.connect(self.on_stop_button_clicked)
        self.tabWidget.currentChanged.connect(self.on_tab_changed)
        self.clearButton.clicked.connect(self.clear_all_line_edits)
        self.viewInSeparateWindowButton.clicked.connect(self.open_work_log_in_separate_window)
        self.clearWorkLogButton.clicked.connect(self.clear_work_log)
        # self.viewButton.clicked.connect(self.view_function_graph)

        self.setFocusPolicy(Qt.StrongFocus)

    def clear_work_log(self):
        self.workLogPlainTextEdit.clear()

    def open_work_log_in_separate_window(self):
        log_text = self.workLogPlainTextEdit.toPlainText()
        self.work_log_window = WorkLogWindow(log_text)
        self.work_log_window.show()

    def clear_all_line_edits(self):
        for line_edit in self.current_tab_line_edits:
            line_edit.clear()

    def update_current_tab_line_edits(self):
        self.current_tab_line_edits.clear()
        current_tab = self.tabWidget.currentWidget()
        if current_tab:
            line_edits = current_tab.findChildren(QLineEdit)
            self.current_tab_line_edits = line_edits

    def on_tab_changed(self, index):
        self.update_current_tab_line_edits()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.startButton.isEnabled():
                self.on_start_button_clicked()
        else:
            super().keyPressEvent(event)

    @QtCore.pyqtSlot(str)
    def append_log_message(self, message: str):
        self.workLogPlainTextEdit.appendPlainText(message)

    @QtCore.pyqtSlot(str)
    def append_html_log_message(self, html: str):
        cursor = self.workLogPlainTextEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertHtml(html)
        self.workLogPlainTextEdit.setTextCursor(cursor)
        self.workLogPlainTextEdit.ensureCursorVisible()

    def toggle_grid_visibility(self, state):
        self.openGLWidget.grid_visible = bool(state)
        self.openGLWidget.update()

    def toggle_axes_visibility(self, state):
        self.openGLWidget.axes_visible = bool(state)
        self.openGLWidget.update()

    def reset_view_to_default(self):
        self.openGLWidget.restore_default_view()

    def on_function_selected(self, index):
        current_func = self.function_manager_helper.get_function_by_index(index)
        self.openGLWidget.update_optimization_path(np.array([]))
        if current_func:
            self.function_manager_helper.set_current_function(index)
            self.openGLWidget.clear_constraints()
            self.openGLWidget.set_function(current_func['function'])

            for constraint in current_func['constraints']:
                self.openGLWidget.add_constraint(constraint['function'])

            self.openGLWidget.build_objective_function_data()
            self.statusbar.showMessage(f"Selected function: {current_func['name']}")

    def show_function_manager_window(self):
        function_manager_window = FunctionManagerWindow(self)
        function_manager_window.exec_()

    def on_start_button_clicked(self):
        current_index = self.tabWidget.currentIndex()

        if current_index == 0:
            self.openGLWidget.update_optimization_path(np.array([]))
            if len(self.function_manager_helper.get_current_function()['constraints']) != 0:
                self.statusbar.showMessage("Attention: Selected function is not supported by GradientDescent class")
                return

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

            self.workLogPlainTextEdit.clear()
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

            self.openGLWidget.set_connect_optimization_points(True)
            self.gradient_descent = GradientDescent(params, self.log_emitter)
            self.gradient_descent.finished_signal.connect(self.on_optimization_finished)
            self.gradient_descent.update_signal.connect(self.openGLWidget.update_optimization_path)

            self.optimization_thread = threading.Thread(target=self.gradient_descent.run, daemon=True)
            self.optimization_thread.start()

            self.statusbar.showMessage("Optimization started")
        elif current_index == 1:
            self.workLogPlainTextEdit.clear()
            self.openGLWidget.update_optimization_path(np.array([]))

            if self.myMethodRadioButton.isChecked():
                if len(self.function_manager_helper.get_current_function()['constraints']) < 3:
                    self.statusbar.showMessage("Attention: Selected function is not supported by MySimplexMethod class")
                    return

                self.tabWidget.setEnabled(False)
                self.selectFunctionComboBox.setEnabled(False)
                self.startButton.setEnabled(False)
                self.stopButton.setEnabled(True)

                symbolic_result = self.function_manager_helper.to_symbolic_function(self.function_manager_helper.current_function_index)

                if symbolic_result['success']:
                    params = {
                        'function': symbolic_result['objective'],
                        'constraints': symbolic_result['constraints'],
                        'variables': symbolic_result['variables'],
                        'max_iterations': 100
                    }

                    self.openGLWidget.set_connect_optimization_points(False)
                    self.simplex_method = MySimplexMethod(params, self.log_emitter)
                    self.simplex_method.finished_signal.connect(self.on_optimization_finished)
                    self.simplex_method.update_signal.connect(self.openGLWidget.update_optimization_path)

                    self.optimization_thread = threading.Thread(target=self.simplex_method.run, daemon=True)
                    self.optimization_thread.start()

                    self.statusbar.showMessage("Simplex method optimization started")
                else:
                    self.statusbar.showMessage(symbolic_result['error'])

            elif self.libraryMethodRadioButton.isChecked():
                self.tabWidget.setEnabled(False)
                self.selectFunctionComboBox.setEnabled(False)
                self.startButton.setEnabled(False)
                self.stopButton.setEnabled(True)

                params = {
                    'function': self.function_manager_helper.get_current_function()['function'],
                    'constraints': self.function_manager_helper.get_current_function()['constraints']
                }

                self.optimizer = LibrarySimplexMethod(params, self.log_emitter)
                self.optimizer.run()

                self.openGLWidget.set_connect_optimization_points(False)
                self.simplex_method = LibrarySimplexMethod(params, self.log_emitter)
                self.simplex_method.finished_signal.connect(self.on_optimization_finished)
                self.simplex_method.update_signal.connect(self.openGLWidget.update_optimization_path)

                self.optimization_thread = threading.Thread(target=self.simplex_method.run, daemon=True)
                self.optimization_thread.start()

                self.statusbar.showMessage("Simplex method optimization started")


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
        elif self.simplex_method:
            self.simplex_method.stop()
            self.stopButton.setEnabled(False)
            self.startButton.setEnabled(True)

    # def view_function_graph(self):
    #     current_func = self.function_manager_helper.get_current_function()
    #
    #     if not current_func:
    #         self.statusbar.showMessage("No function selected!")
    #         return
    #
    #     try:
    #         x_min = -5
    #         x_max = 5
    #         y_min = -5
    #         y_max = 5
    #         segments_x = int(self.simplexSegmentsXLineEdit.text())
    #         segments_y = int(self.simplexSegmentsYLineEdit.text())
    #
    #         params = {
    #             'function': current_func['function'],
    #             'x_range': (x_min, x_max),
    #             'y_range': (y_min, y_max),
    #             'num_segments_x': segments_x,
    #             'num_segments_y': segments_y
    #         }
    #
    #         grid_points, linear_approx = OldSimplexMethod.piecewise_linear_approximation_2d(
    #             params['function'],
    #             params['x_range'],
    #             params['y_range'],
    #             params['num_segments_x'],
    #             params['num_segments_y']
    #         )
    #
    #         fig = plt.figure(figsize=(10, 8))
    #         ax = fig.add_subplot(111, projection='3d')
    #
    #         x_points, y_points = grid_points
    #         colors = plt.cm.plasma(np.linspace(0, 1, len(linear_approx)))
    #
    #         for i, (approx, color) in enumerate(zip(linear_approx, colors)):
    #             x1, x2 = approx['x_range']
    #             y1, y2 = approx['y_range']
    #
    #             x_seg = np.linspace(x1, x2, 10)
    #             y_seg = np.linspace(y1, y2, 10)
    #             x_seg_grid, y_seg_grid = np.meshgrid(x_seg, y_seg)
    #
    #             a0, a1, a2 = approx['coefs']
    #             z_seg_grid = a0 + a1 * x_seg_grid + a2 * y_seg_grid
    #
    #             surf = ax.plot_surface(x_seg_grid, y_seg_grid, z_seg_grid,
    #                                    color=color, alpha=0.7, edgecolor='black',
    #                                    linewidth=1.2, antialiased=True)
    #
    #             if i == 0:
    #                 surf._facecolors2d = surf._facecolor3d
    #                 surf._edgecolors2d = surf._edgecolor3d
    #
    #         for x in x_points:
    #             y_line = np.linspace(y_min, y_max, 2)
    #             z_line = np.zeros(2)
    #             ax.plot([x, x], [y_min, y_max], [0, 0], 'k-', linewidth=0.7, alpha=0.5)
    #
    #         for y in y_points:
    #             x_line = np.linspace(x_min, x_max, 2)
    #             z_line = np.zeros(2)
    #             ax.plot([x_min, x_max], [y, y], [0, 0], 'k-', linewidth=0.7, alpha=0.5)
    #
    #         ax.set_title(f"{current_func['name']} (Piecewise Linear Approximation)")
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')
    #         ax.set_zlabel('Z')
    #         plt.tight_layout()
    #         plt.show()
    #
    #     except ValueError:
    #         self.statusbar.showMessage("Error: Invalid segment values")
    #     except Exception as e:
    #         self.statusbar.showMessage(f"Error: {str(e)}")
