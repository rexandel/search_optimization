from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic

from function_manager_helper import FunctionManagerHelper
from windows.function_manager_window import FunctionManagerWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('gui/ui/main.ui', self)

        self.function_manager = FunctionManagerHelper('functions.json')
        self.selectFunctionComboBox.clear()

        self.selectFunctionComboBox.addItems(self.function_manager.get_function_names())
        self.selectFunctionComboBox.currentIndexChanged.connect(self.on_function_selected)

        if self.function_manager.get_function_names():
            self.function_manager.set_current_function(0)

            current_func = self.function_manager.get_current_function()

            self.openGLWidget.set_function(current_func['function'])

            formula_text = current_func['formula']
            self.statusbar.showMessage(f"Selected function: {current_func['name']} - {formula_text}")

            self.selectFunctionComboBox.setCurrentIndex(0)
        else:
            self.statusbar.showMessage("No functions available")

        self.gridVisibility.stateChanged.connect(self.toggle_grid_visibility)
        self.axisVisibility.stateChanged.connect(self.toggle_axes_visibility)
        self.returnButton.clicked.connect(self.reset_view_to_default)
        self.functionManagerButton.clicked.connect(self.show_function_manager_window)

    def toggle_grid_visibility(self, state):
        self.openGLWidget.grid_visible = bool(state)
        self.openGLWidget.update()

    def toggle_axes_visibility(self, state):
        self.openGLWidget.axes_visible = bool(state)
        self.openGLWidget.update()

    def reset_view_to_default(self):
        self.openGLWidget.restore_default_view()

    def on_function_selected(self, index):
        self.function_manager.set_current_function(index)
        current_func = self.function_manager.get_current_function()

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
