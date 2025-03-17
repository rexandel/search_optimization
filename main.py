from PyQt5.QtWidgets import QApplication
import sys

from gui import Visualization3DApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Visualization3DApp()
    window.show()
    sys.exit(app.exec_())
