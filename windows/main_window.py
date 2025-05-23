from PyQt5.QtWidgets import QMainWindow, QLineEdit, QApplication
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt

from windows import FunctionManagerWindow, WorkLogWindow, SettingsWindow

from optimization_methods import GradientDescentMethod, LibrarySimplexMethod, MySimplexMethod, GeneticAlgorithm, ParticleSwarmMethod, BeeSwarmMethod
from utils import LogEmitter, FunctionManagerHelper

import numpy as np
import threading
import configparser
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('gui/ui/main.ui', self)
        self._show_maximized()

        self.load_visualization_settings()

        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.append_log_message)
        self.log_emitter.html_log_signal.connect(self.append_html_log_message)

        self.function_manager_helper = FunctionManagerHelper('resources/functions.json')
        self.function_manager_helper.populate_combo_box(self.selectFunctionComboBox)
        self.selectFunctionComboBox.currentIndexChanged.connect(self.on_function_selected)

        self.gradient_descent = None
        self.simplex_method = None
        self.genetic_algorithm = None
        self.particle_swarm_algorithm = None
        self.bee_swarm_algorithm = None
        self.optimization_thread = None

        if len(self.function_manager_helper.get_function_names()) > 0:
            self.on_function_selected(0)
        else:
            self.statusbar.showMessage("No functions available")

        self.current_tab_line_edits = []
        self.update_current_tab_line_edits()
        self.tabWidget.currentChanged.connect(self.on_tab_changed)

        self.returnButton.clicked.connect(self.reset_view_to_default)
        self.functionManagerButton.clicked.connect(self.show_function_manager_window)
        self.startButton.clicked.connect(self.on_start_button_clicked)
        self.stopButton.clicked.connect(self.on_stop_button_clicked)
        self.tabWidget.currentChanged.connect(self.on_tab_changed)
        self.clearFieldsButton.clicked.connect(self.clear_all_line_edits)
        self.viewInSeparateWindowButton.clicked.connect(self.open_work_log_in_separate_window)
        self.clearWorkLogButton.clicked.connect(self.clear_work_log)
        self.clearDotsButton.clicked.connect(self.clear_optimization_path)
        self.settingsPushButton.clicked.connect(self.show_settings_window)

        self.truncationSelectionRadioButton.toggled.connect(self.truncation_selection_radio_button_toggled)
        self.truncationSelectionLineEdit.setEnabled(self.truncationSelectionRadioButton.isChecked())

        self.bolzmanSelectionRadioButton.toggled.connect(self.bolzman_selection_radio_button_toggled)
        self.bolzmanSelectionLineEdit.setEnabled(self.bolzmanSelectionRadioButton.isChecked())

        self.setFocusPolicy(Qt.StrongFocus)

    def show_settings_window(self):
        settings_window = SettingsWindow(self)
        settings_window.accepted.connect(self.load_visualization_settings)
        settings_window.exec_()

    def load_visualization_settings(self):
        config = configparser.ConfigParser()
        config_file = os.path.join('resources', 'config.ini')

        if not os.path.exists(config_file):
            self.statusbar.showMessage(f"Warning: {config_file} not found, using default settings")
            return

        try:
            config.read(config_file)
            if 'Visualization' not in config:
                self.statusbar.showMessage("Error: 'Visualization' section not found in config.ini")
                return

            viz = config['Visualization']

            try:
                self.openGLWidget.set_grid_size_x(int(viz.get('grid_size_x')))
            except ValueError:
                self.statusbar.showMessage("Error: Invalid grid_size_x in config.ini")

            try:
                self.openGLWidget.set_grid_size_y(int(viz.get('grid_size_y')))
            except ValueError:
                self.statusbar.showMessage("Error: Invalid grid_size_y in config.ini")

            try:
                self.openGLWidget.set_grid_size_z(int(viz.get('grid_size_z')))
            except ValueError:
                self.statusbar.showMessage("Error: Invalid grid_size_z in config.ini")

            try:
                self.openGLWidget.set_resolution(int(viz.get('resolution')))
            except ValueError:
                self.statusbar.showMessage("Error: Invalid resolution in config.ini")

            self.openGLWidget.set_grid_visible(viz.getboolean('grid_visible', True))
            self.openGLWidget.set_axes_visible(viz.getboolean('axes_visible', True))
            self.openGLWidget.set_axis_ticks_and_numbers_visible(viz.getboolean('axis_ticks_and_numbers_visible', True))

            self.statusbar.showMessage("Visualization settings loaded successfully")
        except Exception as e:
            self.statusbar.showMessage(f"Error loading config.ini: {str(e)}")

    def truncation_selection_radio_button_toggled(self):
        self.truncationSelectionLineEdit.setEnabled(self.truncationSelectionRadioButton.isChecked())

    def bolzman_selection_radio_button_toggled(self):
        self.bolzmanSelectionLineEdit.setEnabled(self.bolzmanSelectionRadioButton.isChecked())

    def on_start_button_clicked(self):
        current_index = self.tabWidget.currentIndex()

        if current_index == 0:
            if len(self.function_manager_helper.get_current_function()['constraints']) != 0:
                self.statusbar.showMessage("Attention: Selected function is not supported by gradient descent")
                return

            data = {
                'x': self.xLineEdit.text(),
                'y': self.yLineEdit.text(),
                'first_eps': self.firstEpsLineEdit.text(),
                'sec_eps': self.secondEpsLineEdit.text(),
                'third_eps': self.thirdEpsLineEdit.text(),
                'initial_step': self.initialStepLineEdit.text(),
                'num_iter': self.numIterLineEdit.text()
            }

            try:
                x = float(data['x'])
                y = float(data['y'])
            except ValueError:
                self.statusbar.showMessage("Error: X and Y must be numeric values")
                return

            try:
                first_eps = float(data['first_eps'])
                sec_eps = float(data['sec_eps'])
                third_eps = float(data['third_eps'])

                if not (0 < first_eps < 1) or not (0 < sec_eps < 1) or not (0 < third_eps < 1):
                    self.statusbar.showMessage("Error: All epsilon values must be between 0 and 1")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Epsilon parameters must be floating-point numbers")
                return

            try:
                initial_step = float(data['initial_step'])
                if initial_step <= 0:
                    self.statusbar.showMessage("Error: Initial step must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Initial step must be a positive integer")
                return

            try:
                num_iter = int(data['num_iter'])
                if num_iter <= 0:
                    self.statusbar.showMessage("Error: Number of iterations must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of iterations must be a positive integer")
                return

            self.workLogPlainTextEdit.clear()
            self.tabWidget.setEnabled(False)
            self.selectFunctionComboBox.setEnabled(False)
            self.startButton.setEnabled(False)
            self.clearDotsButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.openGLWidget.update_optimization_path(np.array([]))
            self.openGLWidget.set_connect_optimization_points(True)

            params = {
                'point': (x, y),
                'epsilons': (first_eps, sec_eps, third_eps),
                'initial_step': initial_step,
                'max_iterations': num_iter,
                'function': self.function_manager_helper.get_current_function()['function']
            }

            self.gradient_descent = GradientDescentMethod(params, self.log_emitter)
            self.gradient_descent.finished_signal.connect(self.on_optimization_finished)
            self.gradient_descent.update_signal.connect(self.openGLWidget.update_optimization_path)

            self.optimization_thread = threading.Thread(target=self.gradient_descent.run, daemon=True)
            self.optimization_thread.start()

            self.statusbar.showMessage("Optimization started")
        elif current_index == 1:
            if self.ourMethodRadioButton.isChecked():
                if len(self.function_manager_helper.get_current_function()['constraints']) < 3:
                    self.statusbar.showMessage("Attention: Selected function is not supported by my simplex method")
                    return

                symbolic_result = self.function_manager_helper.to_symbolic_function(self.function_manager_helper.current_function_index)

                if symbolic_result['success']:
                    params = {
                        'function': symbolic_result['objective'],
                        'constraints': symbolic_result['constraints'],
                        'variables': symbolic_result['variables'],
                        'max_iterations': 100
                    }

                    self.workLogPlainTextEdit.clear()
                    self.tabWidget.setEnabled(False)
                    self.selectFunctionComboBox.setEnabled(False)
                    self.startButton.setEnabled(False)
                    self.clearDotsButton.setEnabled(False)
                    self.stopButton.setEnabled(True)
                    self.openGLWidget.update_optimization_path(np.array([]))
                    self.openGLWidget.set_connect_optimization_points(False)
                    
                    self.simplex_method = MySimplexMethod(params, self.log_emitter)
                    self.simplex_method.finished_signal.connect(self.on_optimization_finished)
                    self.simplex_method.update_signal.connect(self.openGLWidget.update_optimization_path)

                    self.optimization_thread = threading.Thread(target=self.simplex_method.run, daemon=True)
                    self.optimization_thread.start()

                    self.statusbar.showMessage("Simplex method optimization started")
                else:
                    self.statusbar.showMessage(symbolic_result['error'])

            elif self.libraryMethodRadioButton.isChecked():
                params = {
                    'function': self.function_manager_helper.get_current_function()['function'],
                    'constraints': self.function_manager_helper.get_current_function()['constraints']
                }

                self.workLogPlainTextEdit.clear()
                self.tabWidget.setEnabled(False)
                self.selectFunctionComboBox.setEnabled(False)
                self.startButton.setEnabled(False)
                self.clearDotsButton.setEnabled(False)
                self.stopButton.setEnabled(True)
                self.openGLWidget.update_optimization_path(np.array([]))
                self.openGLWidget.set_connect_optimization_points(False)
                
                self.optimizer = LibrarySimplexMethod(params, self.log_emitter)
                self.optimizer.run()
                
                self.simplex_method = LibrarySimplexMethod(params, self.log_emitter)
                self.simplex_method.finished_signal.connect(self.on_optimization_finished)
                self.simplex_method.update_signal.connect(self.openGLWidget.update_optimization_path)

                self.optimization_thread = threading.Thread(target=self.simplex_method.run, daemon=True)
                self.optimization_thread.start()

                self.statusbar.showMessage("Simplex method optimization started")
        elif current_index == 2:
            if len(self.function_manager_helper.get_current_function()['constraints']) != 0:
                self.statusbar.showMessage("Attention: Selected function is not supported by genetic algorithm")
                return

            truncation_threshold = None
            bolzman_threshold = None

            data = {
                'population_size': self.populationSizeLineEdit.text(),
                'number_of_generations': self.numberOfGenerationsLineEdit.text(),
                'convergence_criterion': self.convergenceCriterionLineEdit.text(),

                'roulette_method_flag': self.rouletteMethodRadioButton.isChecked(),
                'tournament_method_flag': self.tournamentMethodRadioButton.isChecked(),

                'probability_of_recombination': self.probabilityOfRecombinationLineEdit.text(),
                'intermediate_recombination_flag': self.intermediateRecombinationRadioButton.isChecked(),
                'line_recombination_flag': self.lineRecombinationRadioButton.isChecked(),

                'probability_of_mutation': self.probabilityOfMutationLineEdit.text(),

                'truncation_threshold_flag': self.truncationSelectionRadioButton.isChecked(),
                'truncation_threshold': self.truncationSelectionLineEdit.text(),
                'bolzman_threshold_flag': self.bolzmanSelectionRadioButton.isChecked(),
                'bolzman_threshold': self.bolzmanSelectionLineEdit.text()
            }

            try:
                population_size = int(data['population_size'])
                if population_size <= 0:
                    self.statusbar.showMessage("Error: Population size must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Population size must be a positive number")
                return

            try:
                number_of_generations = int(data['number_of_generations'])
                if number_of_generations <= 0:
                    self.statusbar.showMessage("Error: Number of generations must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of generations must be a positive integer")
                return

            try:
                convergence_criterion = float(data['convergence_criterion'])

                if not 0 < convergence_criterion < 1:
                    self.statusbar.showMessage("Error: Convergence criterion value must be between 0 and 1")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Convergence criterion value must be floating-point number")
                return

            try:
                roulette_method_flag = bool(data['roulette_method_flag'])
            except ValueError:
                self.statusbar.showMessage("Unexpected error: Roulette method flag parsing is failed")
                return

            try:
                tournament_method_flag = bool(data['tournament_method_flag'])
            except ValueError:
                self.statusbar.showMessage("Unexpected error: Tournament method flag parsing is failed")
                return

            try:
                probability_of_recombination = float(data['probability_of_recombination'])

                if not 0 < probability_of_recombination < 1:
                    self.statusbar.showMessage("Error: Probability of recombination value must be between 0 and 1")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Probability of recombination value must be floating-point number")
                return

            try:
                probability_of_mutation = float(data['probability_of_mutation'])

                if not 0 < probability_of_mutation < 1:
                    self.statusbar.showMessage("Error: Probability of mutation value must be between 0 and 1")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Probability of mutation value must be floating-point number")
                return

            try:
                intermediate_recombination_flag = bool(data['intermediate_recombination_flag'])
            except ValueError:
                self.statusbar.showMessage("Unexpected error: Intermediate recombination flag parsing is failed")
                return

            try:
                line_recombination_flag = bool(data['line_recombination_flag'])
            except ValueError:
                self.statusbar.showMessage("Unexpected error: Line recombination flag parsing is failed")
                return

            try:
                truncation_threshold_flag = bool(data['truncation_threshold_flag'])
            except ValueError:
                self.statusbar.showMessage("Unexpected error: Truncation threshold flag parsing is failed")
                return

            if truncation_threshold_flag:
                try:
                    truncation_threshold = float(data['truncation_threshold'])

                    if truncation_threshold < 0 or truncation_threshold > population_size:
                        self.statusbar.showMessage("Error: Threshold size value must be between 0 and size of population")
                        return
                except ValueError:
                    self.statusbar.showMessage("Error: Threshold size value must be floating-point number")
                    return

            try:
                bolzman_threshold_flag = bool(data['bolzman_threshold_flag'])
            except ValueError:
                self.statusbar.showMessage("Unexpected error: Bolzman threshold flag parsing is failed")
                return

            if bolzman_threshold_flag:
                try:
                    bolzman_threshold = float(data['bolzman_threshold'])

                    if bolzman_threshold == 0:
                        self.statusbar.showMessage(
                            "Error: Temperature cannot be zero")
                        return
                except ValueError:
                    self.statusbar.showMessage("Error: Temperature must be a number")
                    return

            self.workLogPlainTextEdit.clear()
            self.tabWidget.setEnabled(False)
            self.selectFunctionComboBox.setEnabled(False)
            self.startButton.setEnabled(False)
            self.clearDotsButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.openGLWidget.update_optimization_path(np.array([]))
            self.openGLWidget.set_connect_optimization_points(False)

            params = {
                'population_size': population_size,
                'max_generations': number_of_generations,
                'std_threshold': convergence_criterion,
                'x_bounds': self.openGLWidget.get_x_axis_range(),
                'y_bounds': self.openGLWidget.get_y_axis_range(),
                'roulette_method_flag': roulette_method_flag,
                'tournament_method_flag': tournament_method_flag,
                'probability_of_recombination': probability_of_recombination,
                'intermediate_recombination_flag': intermediate_recombination_flag,
                'line_recombination_flag': line_recombination_flag,
                'probability_of_mutation': probability_of_mutation,
                'truncation_threshold_flag': truncation_threshold_flag,
                'truncation_threshold': truncation_threshold,
                'bolzman_threshold_flag': bolzman_threshold_flag,
                'bolzman_threshold': bolzman_threshold,
                'function': self.function_manager_helper.get_current_function()['function']
            }

            self.genetic_algorithm = GeneticAlgorithm(params, self.log_emitter)
            self.genetic_algorithm.finished_signal.connect(self.on_optimization_finished)
            self.genetic_algorithm.update_signal.connect(self.openGLWidget.update_optimization_path)

            self.optimization_thread = threading.Thread(target=self.genetic_algorithm.run, daemon=True)
            self.optimization_thread.start()

            self.statusbar.showMessage("Optimization started")
        elif current_index == 3:
            if len(self.function_manager_helper.get_current_function()['constraints']) != 0:
                self.statusbar.showMessage(
                    "Attention: Selected function is not supported by particle swarm optimization")
                return

            data = {
                'number_of_particles': self.numberOfParticlesLineEdit.text(),
                'cognitive_coefficient': self.cognitiveCoefficientLineEdit.text(),
                'social_coefficient': self.socialCoefficientLineEdit.text(),
                'inertial_weight_flag': self.inertialWeightCheckBox.isChecked(),
                'inertial_weight': self.inertialWeightLineEdit.text(),
                'normalization_flag': self.normalizationCheckBox.isChecked(),
                'normalization_coefficient': self.normalizationLineEdit.text(),
                'number_of_iterations': self.numberOfIterationsParticleLineEdit.text()
            }

            try:
                number_of_particles = int(data['number_of_particles'])
                if number_of_particles <= 0:
                    self.statusbar.showMessage("Error: Number of particles must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of particles must be a positive integer")
                return

            try:
                number_of_iterations = int(data['number_of_iterations'])
                if number_of_iterations <= 0:
                    self.statusbar.showMessage("Error: Number of iterations must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of iterations must be a positive integer")
                return

            try:
                cognitive_coefficient = float(data['cognitive_coefficient'])
                if cognitive_coefficient <= 0:
                    self.statusbar.showMessage("Error: Cognitive coefficient must be a positive number")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Cognitive coefficient must be a floating-point number")
                return

            try:
                social_coefficient = float(data['social_coefficient'])
                if social_coefficient <= 0:
                    self.statusbar.showMessage("Error: Social coefficient must be a positive number")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Social coefficient must be a floating-point number")
                return

            inertial_weight = None
            try:
                inertial_weight_flag = bool(data['inertial_weight_flag'])
                if inertial_weight_flag:
                    inertial_weight = float(data['inertial_weight'])
                    if not 0 < inertial_weight < 1:
                        self.statusbar.showMessage("Error: Inertial weight must be between 0 and 1 (exclusive)")
                        return
            except ValueError:
                self.statusbar.showMessage("Error: Inertial weight must be a floating-point number")
                return

            normalization_coefficient = None
            try:
                normalization_flag = bool(data['normalization_flag'])
                if normalization_flag:
                    if cognitive_coefficient + social_coefficient <= 4:
                        self.statusbar.showMessage(
                            "Error: Sum of cognitive and social coefficients must be greater than 4 when normalization is enabled")
                        return
                    normalization_coefficient = float(data['normalization_coefficient'])
                    if not 0 < normalization_coefficient < 1:
                        self.statusbar.showMessage(
                            "Error: Normalization coefficient must be between 0 and 1 (exclusive)")
                        return
            except ValueError:
                self.statusbar.showMessage("Error: Normalization coefficient must be a floating-point number")
                return

            self.workLogPlainTextEdit.clear()
            self.tabWidget.setEnabled(False)
            self.selectFunctionComboBox.setEnabled(False)
            self.startButton.setEnabled(False)
            self.clearDotsButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.openGLWidget.update_optimization_path(np.array([]))
            self.openGLWidget.set_connect_optimization_points(False)

            params = {
                'number_of_particles': number_of_particles,
                'max_iterations': number_of_iterations,
                'cognitive_coefficient': cognitive_coefficient,
                'social_coefficient': social_coefficient,
                'inertial_weight_flag': inertial_weight_flag,
                'inertial_weight': inertial_weight,
                'normalization_flag': normalization_flag,
                'normalization_coefficient': normalization_coefficient,
                'x_bounds': self.openGLWidget.get_x_axis_range(),
                'y_bounds': self.openGLWidget.get_y_axis_range(),
                'function': self.function_manager_helper.get_current_function()['function']
            }

            self.particle_swarm_algorithm = ParticleSwarmMethod(params, self.log_emitter)
            self.particle_swarm_algorithm.finished_signal.connect(self.on_optimization_finished)
            self.particle_swarm_algorithm.update_signal.connect(self.openGLWidget.update_optimization_path)

            self.optimization_thread = threading.Thread(target=self.particle_swarm_algorithm.run, daemon=True)
            self.optimization_thread.start()

            self.statusbar.showMessage("Optimization started")
        elif current_index == 4:
            if len(self.function_manager_helper.get_current_function()['constraints']) != 0:
                self.statusbar.showMessage(
                    "Attention: Selected function is not supported by particle swarm optimization")
                return

            data = {
                'number_of_iterations': self.numberOfIterationsBeeSwarmLineEdit.text(),
                'number_of_scout_bees': self.numberOfScoutBeesLineEdit.text(),
                'number_of_bees_sent_to_best_plots': self.numberOfBeesSentToBestPlotsLineEdit.text(),
                'number_of_bees_sent_to_other_plots': self.numberOfBeesSentToOtherPlotsLineEdit.text(),
                'number_of_best_plots': self.numberOfBestPlotsLineEdit.text(),
                'number_of_other_selected_plots': self.numberOfOtherSelectedPlotsLineEdit.text(),
                'size_of_area': self.sizeOfAreaLineEdit.text()
            }

            try:
                number_of_iterations = int(data['number_of_iterations'])
                if number_of_iterations <= 0:
                    self.statusbar.showMessage("Error: Number of iterations must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of iterations must be a positive integer")
                return

            try:
                number_of_scout_bees = int(data['number_of_scout_bees'])
                if number_of_scout_bees <= 0:
                    self.statusbar.showMessage("Error: Number of scout bees must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of scout bees must be a positive integer")
                return

            try:
                number_of_bees_sent_to_best_plots = int(data['number_of_bees_sent_to_best_plots'])
                if number_of_bees_sent_to_best_plots <= 0:
                    self.statusbar.showMessage("Error: Number of bees sent to best plots must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of bees sent to best plots must be a positive integer")
                return

            try:
                number_of_bees_sent_to_other_plots = int(data['number_of_bees_sent_to_other_plots'])
                if number_of_bees_sent_to_other_plots <= 0:
                    self.statusbar.showMessage("Error: Number of bees sent to other plots must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of bees sent to other plots must be a positive integer")
                return

            try:
                number_of_best_plots = int(data['number_of_best_plots'])
                if number_of_best_plots <= 0:
                    self.statusbar.showMessage("Error: Number of best plots must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of best plots must be a positive integer")
                return

            try:
                number_of_other_selected_plots = int(data['number_of_other_selected_plots'])
                if number_of_other_selected_plots <= 0:
                    self.statusbar.showMessage("Error: Number of other selected plots must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Number of other selected plots must be a positive integer")
                return

            try:
                size_of_area = int(data['size_of_area'])
                if size_of_area <= 0:
                    self.statusbar.showMessage("Error: Size of area must be a positive integer")
                    return
            except ValueError:
                self.statusbar.showMessage("Error: Size of area must be a positive integer")
                return

            self.workLogPlainTextEdit.clear()
            self.tabWidget.setEnabled(False)
            self.selectFunctionComboBox.setEnabled(False)
            self.startButton.setEnabled(False)
            self.clearDotsButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.openGLWidget.update_optimization_path(np.array([]))
            self.openGLWidget.set_connect_optimization_points(False)

            params = {
                'max_iterations': number_of_iterations,
                'number_of_scout_bees': number_of_scout_bees,
                'number_of_bees_sent_to_best_plots': number_of_bees_sent_to_best_plots,
                'number_of_bees_sent_to_other_plots': number_of_bees_sent_to_other_plots,
                'number_of_best_plots': number_of_best_plots,
                'number_of_other_selected_plots': number_of_other_selected_plots,
                'size_of_area': size_of_area,
                'x_bounds': self.openGLWidget.get_x_axis_range(),
                'y_bounds': self.openGLWidget.get_y_axis_range(),
                'function': self.function_manager_helper.get_current_function()['function']
            }

            self.bee_swarm_algorithm = BeeSwarmMethod(params, self.log_emitter)
            self.bee_swarm_algorithm.finished_signal.connect(self.on_optimization_finished)
            self.bee_swarm_algorithm.update_signal.connect(self.openGLWidget.update_optimization_path)

            self.optimization_thread = threading.Thread(target=self.bee_swarm_algorithm.run, daemon=True)
            self.optimization_thread.start()

            self.statusbar.showMessage("Optimization started")

    def _show_maximized(self):
        self.setWindowState(Qt.WindowMaximized)
        self.show()

    def _center(self):
        window_geometry = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        center_point = QApplication.desktop().screenGeometry(screen).center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())

    def clear_optimization_path(self):
        self.openGLWidget.update_optimization_path(np.array([]))

    def clear_work_log(self):
        self.workLogPlainTextEdit.clear()

    def open_work_log_in_separate_window(self):
        log_text = self.workLogPlainTextEdit.toPlainText()
        self.work_log_window = WorkLogWindow(log_text)
        self.work_log_window.show()

    def clear_all_line_edits(self):
        for line_edit in self.current_tab_line_edits:
            line_edit.clear()

    def update_current_tab_line_edits(self):
        self.current_tab_line_edits.clear()
        current_tab = self.tabWidget.currentWidget()
        if current_tab:
            line_edits = current_tab.findChildren(QLineEdit)
            self.current_tab_line_edits = line_edits

    def on_tab_changed(self, index):
        self.update_current_tab_line_edits()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.startButton.isEnabled():
                self.on_start_button_clicked()
        else:
            super().keyPressEvent(event)

    @QtCore.pyqtSlot(str)
    def append_log_message(self, message: str):
        self.workLogPlainTextEdit.appendPlainText(message)

    @QtCore.pyqtSlot(str)
    def append_html_log_message(self, html: str):
        cursor = self.workLogPlainTextEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertHtml(html)
        self.workLogPlainTextEdit.setTextCursor(cursor)
        self.workLogPlainTextEdit.ensureCursorVisible()

    def toggle_grid_visibility(self, state):
        self.openGLWidget.grid_visible = bool(state)
        self.openGLWidget.update()

    def toggle_axes_visibility(self, state):
        self.openGLWidget.axes_visible = bool(state)
        self.openGLWidget.update()

    def reset_view_to_default(self):
        self.openGLWidget.restore_default_view()

    def on_function_selected(self, index):
        current_func = self.function_manager_helper.get_function_by_index(index)
        self.openGLWidget.update_optimization_path(np.array([]))
        if current_func:
            self.function_manager_helper.set_current_function(index)
            self.openGLWidget.clear_constraints()
            self.openGLWidget.set_function(current_func['function'])

            for constraint in current_func['constraints']:
                self.openGLWidget.add_constraint(constraint['function'])

            self.openGLWidget.build_objective_function_data()
            self.statusbar.showMessage(f"Selected function: {current_func['name']}")

    def show_function_manager_window(self):
        function_manager_window = FunctionManagerWindow(self)
        function_manager_window.exec_()

    @QtCore.pyqtSlot()
    def on_optimization_finished(self):
        self.tabWidget.setEnabled(True)
        self.selectFunctionComboBox.setEnabled(True)
        self.startButton.setEnabled(True)
        self.clearDotsButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.append_log_message("Optimization finished")

    def on_stop_button_clicked(self):
        if self.gradient_descent:
            self.gradient_descent.stop()
            self.stopButton.setEnabled(False)
            self.clearDotsButton.setEnabled(True)
            self.startButton.setEnabled(True)
        elif self.simplex_method:
            self.simplex_method.stop()
            self.stopButton.setEnabled(False)
            self.clearDotsButton.setEnabled(True)
            self.startButton.setEnabled(True)
        elif self.genetic_algorithm:
            self.genetic_algorithm.stop()
            self.stopButton.setEnabled(False)
            self.clearDotsButton.setEnabled(True)
            self.startButton.setEnabled(True)
        elif self.particle_swarm_algorithm:
            self.particle_swarm_algorithm.stop()
            self.stopButton.setEnabled(False)
            self.clearDotsButton.setEnabled(True)
            self.startButton.setEnabled(True)
        elif self.bee_swarm_algorithm:
            self.bee_swarm_algorithm.stop()
            self.stopButton.setEnabled(False)
            self.clearDotsButton.setEnabled(True)
            self.startButton.setEnabled(True)
