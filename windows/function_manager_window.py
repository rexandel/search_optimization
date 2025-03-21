from PyQt5.QtWidgets import QDialog
from PyQt5 import uic

class FunctionManagerWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        uic.loadUi('gui/ui/function_manager.ui', self)

