from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog


class WorkLogWindow(QDialog):
    def __init__(self, log_text=""):
        super().__init__()
        uic.loadUi('gui/ui/work_log.ui', self)
        self.logEventInSeparateWindowPlainTextEdit.clear()

        self.logEventInSeparateWindowPlainTextEdit.setPlainText(log_text)

    def closeEvent(self, event):
        event.accept()
