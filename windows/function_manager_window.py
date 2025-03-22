from PyQt5.QtWidgets import QDialog
from PyQt5 import uic

from function_manager_helper import FunctionManagerHelper

class FunctionManagerWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        uic.loadUi('gui/ui/function_manager.ui', self)

        self.function_manager_helper = FunctionManagerHelper('functions.json')
        self.function_manager_helper.populate_combo_box(self.selectFunctionComboBox)

        self.selectFunctionComboBox.currentIndexChanged.connect(self.on_function_selected)
        self.cancelButton.clicked.connect(self.close_button_event)

    def on_function_selected(self, index):
        self.function_manager_helper.set_current_function(index)
        current_func = self.function_manager_helper.get_current_function()

        if current_func:
            self.functionNameLineEdit.setText(current_func['name'])
            self.formulaLineEdit.setText(current_func['formula'])

    def close_button_event(self):
        self.close()