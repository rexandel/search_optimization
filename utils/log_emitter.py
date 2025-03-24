from PyQt5.QtCore import pyqtSignal, QObject

class LogEmitter(QObject):
    log_signal = pyqtSignal(str)
    html_log_signal = pyqtSignal(str)
