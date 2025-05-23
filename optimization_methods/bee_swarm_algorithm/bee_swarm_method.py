from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import time
from .bee import Bee

class BeeSwarmMethod(QObject):
    finished_signal = pyqtSignal()
    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, params_dict, log_emitter):
        super().__init__()
        self.number_of_scout_bees = params_dict['number_of_scout_bees']
        self.number_of_bees_sent_to_best_plots = params_dict['number_of_bees_sent_to_best_plots']
        self.number_of_bees_sent_to_other_plots = params_dict['number_of_bees_sent_to_other_plots']
        self.number_of_best_plots = params_dict['number_of_best_plots']
        self.number_of_other_selected_plots = params_dict['number_of_other_selected_plots']
        self.size_of_area = params_dict['size_of_area']

        self._x_bounds = params_dict['x_bounds']
        self._y_bounds = params_dict['y_bounds']
        self._function = params_dict['function']

        self._is_running = False
        self.log_emitter = log_emitter
        self.initial_delay = 0.3
        self.min_delay = 0.01

        self._swarm = self.initialize_swarm()

    def initialize_swarm(self):
        pass

    def run(self):
        pass

    def stop(self):
        pass

