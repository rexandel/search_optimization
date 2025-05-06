from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog


class WorkLogWindow(QDialog):
    def __init__(self, log_text=""):
        super().__init__()
        uic.loadUi('gui/ui/work_log.ui', self)

        # Set the text from main window to the plain text edit
        self.logEventInSeparateWindowPlainTextEdit.setPlainText(log_text)

        # Make the window stay on top
        # self.setWindowFlags(QtWidgets.QWindow.StaysOnTopHint)

    def closeEvent(self, event):
        # Handle window close event
        event.accept()
