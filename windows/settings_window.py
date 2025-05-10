from PyQt5.QtWidgets import QDialog
import configparser
import os
from PyQt5 import uic

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('gui/ui/settings.ui', self)

        self.config_file = os.path.join('resources', 'config.ini')
        self.config = configparser.ConfigParser()

        self.load_settings()

        self.savePushButton.clicked.connect(self.save_settings)
        self.cancelPushButton.clicked.connect(self.close_window)
        self.axesVisibleCheckBox.toggled.connect(self.change_state_axis_tisks_numbers)

    def change_state_axis_tisks_numbers(self):
        if self.axesVisibleCheckBox.isChecked():
            self.axisTisksNumbersCheckBox.setEnabled(True)
        else:
            self.axisTisksNumbersCheckBox.setChecked(False)
            self.axisTisksNumbersCheckBox.setEnabled(False)

    def load_settings(self):
        if not os.path.exists(self.config_file):
            self.parent().statusbar.showMessage(f"Warning: {self.config_file} not found")
            return

        try:
            self.config.read(self.config_file)
            if 'Visualization' not in self.config:
                self.parent().statusbar.showMessage("Error: 'Visualization' section not found in config.ini")
                return

            viz = self.config['Visualization']
            self.dimensionXAxisLineEdit.setText(viz.get('grid_size_x', '10'))
            self.dimensionYAxisLineEdit.setText(viz.get('grid_size_y', '10'))
            self.dimensionZAxisLineEdit.setText(viz.get('grid_size_z', '10'))
            self.renderResolutionLineEdit.setText(viz.get('resolution', '250'))
            self.gridVisibleCheckBox.setChecked(viz.getboolean('grid_visible', True))
            self.axesVisibleCheckBox.setChecked(viz.getboolean('axes_visible', True))
            self.axisTisksNumbersCheckBox.setChecked(viz.getboolean('axis_ticks_and_numbers_visible', True))
        except Exception as e:
            self.parent().statusbar.showMessage(f"Error loading config.ini: {str(e)}")

    def save_settings(self):
        try:
            grid_size_x = int(self.dimensionXAxisLineEdit.text())
            grid_size_y = int(self.dimensionYAxisLineEdit.text())
            grid_size_z = int(self.dimensionZAxisLineEdit.text())
            resolution = int(self.renderResolutionLineEdit.text())

            if grid_size_x <= 0 or grid_size_y <= 0 or grid_size_z <= 0:
                self.parent().statusbar.showMessage("Error: Grid sizes must be positive integers")
                return
            if resolution <= 0:
                self.parent().statusbar.showMessage("Error: Resolution must be a positive integer")
                return

            if 'Visualization' not in self.config:
                self.config['Visualization'] = {}

            self.config['Visualization']['grid_size_x'] = str(grid_size_x)
            self.config['Visualization']['grid_size_y'] = str(grid_size_y)
            self.config['Visualization']['grid_size_z'] = str(grid_size_z)
            self.config['Visualization']['resolution'] = str(resolution)
            self.config['Visualization']['grid_visible'] = str(self.gridVisibleCheckBox.isChecked())
            self.config['Visualization']['axes_visible'] = str(self.axesVisibleCheckBox.isChecked())
            self.config['Visualization']['axis_ticks_and_numbers_visible'] = str(self.axisTisksNumbersCheckBox.isChecked())

            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)

            self.parent().statusbar.showMessage("Settings saved successfully")
            self.accept()
        except ValueError:
            self.parent().statusbar.showMessage("Error: All numeric fields must be valid integers")
        except Exception as e:
            self.parent().statusbar.showMessage(f"Error saving config.ini: {str(e)}")

    def close_window(self):
        self.close()