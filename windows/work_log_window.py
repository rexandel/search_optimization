from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QApplication

class WorkLogWindow(QDialog):
    def __init__(self, log_text=""):
        super().__init__()
        uic.loadUi('gui/ui/work_log.ui', self)
        self.workLogPlainTextEdit.clear()

        self.workLogPlainTextEdit.setPlainText(log_text)
        self.clearButton.clicked.connect(self.clear_work_log)
        self.copyButton.clicked.connect(self.copy_from_work_log)

    def clear_work_log(self):
        self.workLogPlainTextEdit.clear()

    def copy_from_work_log(self):
        text = self.workLogPlainTextEdit.toPlainText()
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def closeEvent(self, event):
        event.accept()
