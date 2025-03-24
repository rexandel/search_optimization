from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic

from function_manager_helper import FunctionManagerHelper
from windows.function_manager_window import FunctionManagerWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('gui/ui/main.ui', self)

        self.function_manager_helper = FunctionManagerHelper('functions.json')
        self.function_manager_helper.populate_combo_box(self.selectFunctionComboBox)
        self.selectFunctionComboBox.currentIndexChanged.connect(self.on_function_selected)

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
            initial_step = int(data['initial_step'])
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
        self.gridGroupBox.setEnabled(False)
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)



        params = {
            'point': (x, y),
            'epsilons': (first_eps, sec_eps, third_eps),
            'initial_step': initial_step,
            'max_iterations': num_iter,
            'function': self.function_manager_helper.get_current_function()['function']
        }

        self.statusbar.showMessage("Optimization started")