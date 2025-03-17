from PyQt5.QtWidgets import QMainWindow

from PyQt5 import uic

from gui.widget import Visualization3DWidget
from utils.parser import FunctionManager

class Visualization3DApp(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('main_window.ui', self)

        self.visualization_widget = Visualization3DWidget(self.centralwidget)
        self.visualization_widget.setGeometry(self.openGLWidget.geometry())

        self.openGLWidget.setParent(None)
        self.openGLWidget = self.visualization_widget

        self.function_manager = FunctionManager('functions.json')
        self.selectFunctionComboBox.clear()

        self.selectFunctionComboBox.addItems(self.function_manager.get_function_names())
        self.selectFunctionComboBox.currentIndexChanged.connect(self.on_function_selected)

        if self.function_manager.get_function_names():
            self.function_manager.set_current_function(0)

            current_func = self.function_manager.get_current_function()

            self.visualization_widget.set_function(current_func['function'])

            formula_text = current_func['formula']
            self.statusbar.showMessage(f"Selected function: {current_func['name']} - {formula_text}")

            self.selectFunctionComboBox.setCurrentIndex(0)
        else:
            self.statusbar.showMessage("No functions available")

        self.gridVisibility.stateChanged.connect(self.toggle_grid_visibility)
        self.axisVisibility.stateChanged.connect(self.toggle_axes_visibility)
        self.returnButton.clicked.connect(self.reset_view_to_default)

    def toggle_grid_visibility(self, state):
        self.visualization_widget.grid_visible = bool(state)
        self.visualization_widget.update()

    def toggle_axes_visibility(self, state):
        self.visualization_widget.axes_visible = bool(state)
        self.visualization_widget.update()

    def reset_view_to_default(self):
        self.visualization_widget.restore_default_view()

    def on_function_selected(self, index):
        self.function_manager.set_current_function(index)
        current_func = self.function_manager.get_current_function()

        if current_func:
            self.visualization_widget.set_function(current_func['function'])
            formula_text = current_func['formula']
            self.statusbar.showMessage(f"Selected function: {current_func['name']} - {formula_text}")
        else:
            self.visualization_widget.set_function(None)
            self.statusbar.showMessage("No function selected")
