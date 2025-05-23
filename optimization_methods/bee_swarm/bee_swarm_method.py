from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import time
from .bee import Bee

class BeeSwarmMethod(QObject):
    finished_signal = pyqtSignal()
    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, params_dict, log_emitter):
        super().__init__()
        self._number_of_scout_bees = params_dict['number_of_scout_bees']
        self._number_of_bees_sent_to_best_plots = params_dict['number_of_bees_sent_to_best_plots']
        self._number_of_bees_sent_to_other_plots = params_dict['number_of_bees_sent_to_other_plots']
        self._number_of_best_plots = params_dict['number_of_best_plots']
        self._number_of_other_selected_plots = params_dict['number_of_other_selected_plots']
        self._size_of_area = params_dict['size_of_area']
        self._max_iterations = params_dict['max_iterations']

        self._x_bounds = params_dict['x_bounds']
        self._y_bounds = params_dict['y_bounds']
        self._function = params_dict['function']

        self._is_running = False
        self.log_emitter = log_emitter
        self.initial_delay = 0.3
        self.min_delay = 0.01

        self._swarm = self.initialize_swarm()
        self._best_fitness = float('inf')
        self._best_position = None

        self._scout_bees = []
        self._bees_in_best_plots = []
        self._bees_in_other_plots = []

    def initialize_swarm(self):
        bee_swarm = []
        bees_sent_to_best_plots = self._number_of_bees_sent_to_best_plots * self._number_of_best_plots
        bees_sent_to_other_plots = self._number_of_bees_sent_to_other_plots * self._number_of_other_selected_plots
        bee_count = self._number_of_scout_bees + bees_sent_to_best_plots + bees_sent_to_other_plots

        for _ in range(bee_count):
            bee_swarm.append(Bee(self))
        return bee_swarm

    def find_best_position(self):
        current_best_fitness_in_swarm = float('inf')
        current_best_position_in_swarm = None
        has_valid_bee = False

        for bee in self._swarm:
            if bee.fitness is not None and bee.fitness < current_best_fitness_in_swarm:
                current_best_fitness_in_swarm = bee.fitness
                current_best_position_in_swarm = bee.position.copy()
                has_valid_bee = True
        
        if has_valid_bee and current_best_fitness_in_swarm < self._best_fitness:
            self._best_fitness = current_best_fitness_in_swarm
            self._best_position = current_best_position_in_swarm
            self.log_emitter.log_signal.emit(f"Global best updated: Function Value: {self._best_fitness:.4f}, Position: ({self._best_position[0]:.2f}, {self._best_position[1]:.2f})")

    def run(self):
        self._is_running = True
        self.log_emitter.log_signal.emit("🔹 Bee Swarm Optimization started...")

        self.log_emitter.log_signal.emit("------------------------------------")
        self.log_emitter.log_signal.emit("Selected parameters:")
        self.log_emitter.log_signal.emit(f"  Number of scout bees: {self._number_of_scout_bees}")
        self.log_emitter.log_signal.emit(f"  Bees sent to best plots: {self._number_of_bees_sent_to_best_plots}")
        self.log_emitter.log_signal.emit(f"  Bees sent to other plots: {self._number_of_bees_sent_to_other_plots}")
        self.log_emitter.log_signal.emit(f"  Number of best plots: {self._number_of_best_plots}")
        self.log_emitter.log_signal.emit(f"  Number of other selected plots: {self._number_of_other_selected_plots}")
        self.log_emitter.log_signal.emit(f"  Max iterations: {self._max_iterations}")
        self.log_emitter.log_signal.emit(f"  X bounds: ({self._x_bounds[0]:.6f}, {self._x_bounds[1]:.6f})")
        self.log_emitter.log_signal.emit(f"  Y bounds: ({self._y_bounds[0]:.6f}, {self._y_bounds[1]:.6f})")
        self.log_emitter.log_signal.emit(f"  Area size: {self._size_of_area:.6f}")
        self.log_emitter.log_signal.emit("------------------------------------")
        self.log_emitter.log_signal.emit(f"Initial swarm created with {len(self._swarm)} bees in hive.")

        try:
            iteration = 0
            while self._is_running and iteration < self._max_iterations:
                self.log_emitter.log_signal.emit(f"--- Iteration {iteration + 1} ---")
                self.log_emitter.log_signal.emit("Working with bee scouts...")
                available_for_scouting = [b for b in self._swarm if b not in self._scout_bees and b not in self._bees_in_best_plots and b not in self._bees_in_other_plots]
                num_to_deploy_scouts = min(self._number_of_scout_bees, len(available_for_scouting))
                
                current_scouts = available_for_scouting[:num_to_deploy_scouts]
                self._scout_bees.extend(current_scouts)

                for bee in current_scouts:
                    bee.move_to_random_point()

                if current_scouts:
                    scout_positions = np.array([bee.position for bee in current_scouts if bee.position is not None])
                    if scout_positions.size > 0:
                        self.update_signal.emit(scout_positions)
                        self.log_emitter.log_signal.emit(f"Deployed {len(current_scouts)} scouts. They are now exploring...")
                    else:
                        self.log_emitter.log_signal.emit(f"Deployed {len(current_scouts)} scouts, but no valid positions to show yet.")

                    scouting_pause_duration = max(self.min_delay, self.initial_delay / 2)
                    time.sleep(scouting_pause_duration)
                    self.log_emitter.log_signal.emit(f"Scouts have returned after {scouting_pause_duration:.2f}s of exploration.")
                    
                    # scout_bees_info = "\n ".join(str(bee) for bee in current_scouts)
                    # self.log_emitter.log_signal.emit(f"Returned scout bees details:\n{scout_bees_info}")
                else:
                    self.log_emitter.log_signal.emit("No available bees to deploy as scouts this iteration.")

                active_positions = np.array([b.position for b in self._swarm if b.position is not None])
                if active_positions.size > 0:
                    self.update_signal.emit(active_positions)
                
                self._swarm.sort(key=lambda bee: bee.fitness if bee.fitness is not None else float('inf'))
                self.log_emitter.log_signal.emit("Swarm sorted by fitness (bees with no fitness are last).")

                self.find_best_position()
                self.log_emitter.log_signal.emit("")

                available_for_best_plots = [b for b in self._swarm if b not in self._scout_bees and b not in self._bees_in_best_plots and b not in self._bees_in_other_plots]

                self.log_emitter.log_signal.emit("Working with best sites...")
                num_bees_to_send_to_best = self._number_of_bees_sent_to_best_plots * self._number_of_best_plots
                actual_sent_to_best = 0

                for i, best_site_bee in enumerate(self._swarm[:self._number_of_best_plots]):
                    if best_site_bee.position is None:
                        continue
                    self.log_emitter.log_signal.emit(f" Processing best site {i+1} (found by {best_site_bee})")
                    for _ in range(self._number_of_bees_sent_to_best_plots):
                        if not available_for_best_plots or actual_sent_to_best >= num_bees_to_send_to_best:
                            break
                        candidate_bee = available_for_best_plots.pop(np.random.randint(len(available_for_best_plots)))
                        candidate_bee.move_to_random_nearby_point(best_site_bee.position)
                        self._bees_in_best_plots.append(candidate_bee)
                        actual_sent_to_best += 1
                    if not available_for_best_plots or actual_sent_to_best >= num_bees_to_send_to_best:
                        break
                if actual_sent_to_best > 0:
                     self.log_emitter.log_signal.emit(f"Sent {actual_sent_to_best} bees to best plots.")

                self.log_emitter.log_signal.emit("")
                available_for_other_plots = [b for b in self._swarm if b not in self._scout_bees and b not in self._bees_in_best_plots and b not in self._bees_in_other_plots]
                num_bees_to_send_to_other = self._number_of_bees_sent_to_other_plots * self._number_of_other_selected_plots
                actual_sent_to_other = 0

                self.log_emitter.log_signal.emit("Working with other sites...")
                start_index_other_plots = self._number_of_best_plots
                end_index_other_plots = self._number_of_best_plots + self._number_of_other_selected_plots

                for i, other_site_bee in enumerate(self._swarm[start_index_other_plots:end_index_other_plots]):
                    if other_site_bee.position is None:
                        continue
                    self.log_emitter.log_signal.emit(f"Processing other site {i+1} (found by {other_site_bee})")
                    for _ in range(self._number_of_bees_sent_to_other_plots):
                        if not available_for_other_plots or actual_sent_to_other >= num_bees_to_send_to_other:
                            break
                        candidate_bee = available_for_other_plots.pop(np.random.randint(len(available_for_other_plots)))
                        candidate_bee.move_to_random_nearby_point(other_site_bee.position)
                        self._bees_in_other_plots.append(candidate_bee)
                        actual_sent_to_other += 1
                    if not available_for_other_plots or actual_sent_to_other >= num_bees_to_send_to_other:
                        break
                if actual_sent_to_other > 0:
                    self.log_emitter.log_signal.emit(f"Sent {actual_sent_to_other} bees to other plots.")

                self.find_best_position()
                self.log_emitter.log_signal.emit("")

                active_positions = np.array([b.position for b in self._swarm if b.position is not None])
                if active_positions.size > 0:
                    self.update_signal.emit(active_positions)

                self.log_emitter.log_signal.emit("Working with best fitness bees...")
                bees_with_fitness = [b for b in self._swarm if b.fitness is not None]
                if bees_with_fitness:
                    num_top_bees = max(1, int(len(bees_with_fitness) * 0.1))
                    top_bees = sorted(bees_with_fitness, key=lambda bee: bee.fitness)[:num_top_bees]
                    top_bees_info = "\n".join(
                        f"  Bee {i + 1}: Position: ({bee.position[0]:.2f}, {bee.position[1]:.2f}), Function Value: {bee.fitness:.2f}"
                        for i, bee in enumerate(top_bees)
                    )
                    self.log_emitter.log_signal.emit(f"Top {len(top_bees)} bees (with fitness):\n{top_bees_info}")
                else:
                    self.log_emitter.log_signal.emit("No bees with fitness to display top 10%.")
                
                # Форматированный вывод глобальной лучшей позиции и значения функции
                fitness_str = f"{self._best_fitness:.4f}" if self._best_fitness != float('inf') else "N/A"
                if self._best_position is not None:
                    position_str = f"({self._best_position[0]:.4f}, {self._best_position[1]:.4f})"
                else:
                    position_str = "N/A"
                self.log_emitter.log_signal.emit(f"Global best after iteration {iteration + 1}: Function Value: {fitness_str}, Position: {position_str}")

                self._scout_bees.clear()
                self._bees_in_best_plots.clear()
                self._bees_in_other_plots.clear()
                self.log_emitter.log_signal.emit("All active bees returned to hive (roles cleared).\n")

                iteration += 1
                time.sleep(max(self.min_delay, self.initial_delay * (0.95 ** iteration)))

            final_fitness_log = f"{self._best_fitness:.4f}" if self._best_fitness != float('inf') else "Not found"
            final_position_log = f"({self._best_position[0]:.2f}, {self._best_position[1]:.2f})" if self._best_position is not None else "Not found"
            final_message = (
                "🎉 Bee Swarm Optimization finished!\n"
                f"🏁 Best Position: {final_position_log}\n"
                f"📊 Best Fitness: {final_fitness_log}\n"
                f"Total iterations: {iteration}"
            )
            self.log_emitter.log_signal.emit(final_message)

        except Exception as e:
            self.log_emitter.log_signal.emit(f"❌ Error in Bee Swarm: {str(e)} at iteration {iteration if 'iteration' in locals() else 'N/A'}")
            import traceback
            self.log_emitter.log_signal.emit(traceback.format_exc())
        finally:
            self._is_running = False
            self.finished_signal.emit()

    def stop(self):
        pass

    @property
    def number_of_scout_bees(self):
        return self._number_of_scout_bees

    @property
    def number_of_bees_sent_to_best_plots(self):
        return self._number_of_bees_sent_to_best_plots

    @property
    def number_of_bees_sent_to_other_plots(self):
        return self._number_of_bees_sent_to_other_plots

    @property
    def number_of_best_plots(self):
        return self._number_of_best_plots

    @property
    def number_of_other_selected_plots(self):
        return self._number_of_other_selected_plots

    @property
    def size_of_area(self):
        return self._size_of_area

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def x_bounds(self):
        return self._x_bounds

    @property
    def y_bounds(self):
        return self._y_bounds

    @property
    def function(self):
        return self._function

    @property
    def is_running(self):
        return self._is_running

    @property
    def swarm(self):
        return self._swarm

    @property
    def best_fitness(self):
        return self._best_fitness

    @property
    def best_position(self):
        return self._best_position

    @property
    def bees_in_best_plots(self):
        return self._bees_in_best_plots

    @property
    def bees_in_other_plots(self):
        return self._bees_in_other_plots

    @number_of_scout_bees.setter
    def number_of_scout_bees(self, value):
        if isinstance(value, int) and value > 0:
            self._number_of_scout_bees = value
        else:
            raise ValueError("Number of scout bees must be a positive integer")

    @number_of_bees_sent_to_best_plots.setter
    def number_of_bees_sent_to_best_plots(self, value):
        if isinstance(value, int) and value > 0:
            self._number_of_bees_sent_to_best_plots = value
        else:
            raise ValueError("Number of bees sent to best plots must be a positive integer")

    @number_of_bees_sent_to_other_plots.setter
    def number_of_bees_sent_to_other_plots(self, value):
        if isinstance(value, int) and value > 0:
            self._number_of_bees_sent_to_other_plots = value
        else:
            raise ValueError("Number of bees sent to other plots must be a positive integer")

    @number_of_best_plots.setter
    def number_of_best_plots(self, value):
        if isinstance(value, int) and value > 0:
            self._number_of_best_plots = value
        else:
            raise ValueError("Number of best plots must be a positive integer")

    @number_of_other_selected_plots.setter
    def number_of_other_selected_plots(self, value):
        if isinstance(value, int) and value > 0:
            self._number_of_other_selected_plots = value
        else:
            raise ValueError("Number of other selected plots must be a positive integer")

    @size_of_area.setter
    def size_of_area(self, value):
        if isinstance(value, (int, float)) and value > 0:
            self._size_of_area = value
        else:
            raise ValueError("Size of area must be a positive number")

    @max_iterations.setter
    def max_iterations(self, value):
        if isinstance(value, int) and value > 0:
            self._max_iterations = value
        else:
            raise ValueError("Max iterations must be a positive integer")

    @x_bounds.setter
    def x_bounds(self, value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            self._x_bounds = value
        else:
            raise ValueError("X bounds must be a list or tuple of length 2")

    @y_bounds.setter
    def y_bounds(self, value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            self._y_bounds = value
        else:
            raise ValueError("Y bounds must be a list or tuple of length 2")

    @function.setter
    def function(self, value):
        if callable(value):
            self._function = value
        else:
            raise ValueError("Function must be callable")

    @is_running.setter
    def is_running(self, value):
        if isinstance(value, bool):
            self._is_running = value
        else:
            raise ValueError("is_running must be a boolean")

    @swarm.setter
    def swarm(self, value):
        if isinstance(value, list) and all(isinstance(bee, Bee) for bee in value):
            self._swarm = value
        else:
            raise ValueError("Swarm must be a list of Bee objects")

    @best_fitness.setter
    def best_fitness(self, value):
        if isinstance(value, (int, float)):
            self._best_fitness = value
        else:
            raise ValueError("Best fitness must be a number")

    @best_position.setter
    def best_position(self, value):
        if isinstance(value, np.ndarray) and value.shape == (2,):
            self._best_position = value
        else:
            raise ValueError("Best position must be a numpy array of shape (2,)")

    @bees_in_best_plots.setter
    def bees_in_best_plots(self, value):
        if isinstance(value, list) and all(isinstance(bee, Bee) for bee in value):
            self._bees_in_best_plots = value
        else:
            raise ValueError("Bees in best plots must be a list of Bee objects")

    @bees_in_other_plots.setter
    def bees_in_other_plots(self, value):
        if isinstance(value, list) and all(isinstance(bee, Bee) for bee in value):
            self._bees_in_other_plots = value
        else:
            raise ValueError("Bees in other plots must be a list of Bee objects")
