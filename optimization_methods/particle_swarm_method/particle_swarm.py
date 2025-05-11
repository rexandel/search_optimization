import numpy as np
import time
from PyQt5.QtCore import QObject, pyqtSignal
from .particle import Particle


class ParticleSwarmMethod(QObject):
    finished_signal = pyqtSignal()
    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, params_dict, log_emitter):
        super().__init__()
        self._number_of_particles = params_dict['number_of_particles']
        self._max_iterations = params_dict['max_iterations']
        self._inertial_weight = params_dict['inertial_weight']
        self._inertial_weight_flag = params_dict['inertial_weight_flag']
        self._cognitive_coefficient = params_dict['cognitive_coefficient']
        self._social_coefficient = params_dict['social_coefficient']
        self._x_bounds = params_dict['x_bounds']
        self._y_bounds = params_dict['y_bounds']
        self._function = params_dict['function']

        self._is_running = False
        self.log_emitter = log_emitter
        self.initial_delay = 0.3
        self.min_delay = 0.01

        self._swarm = self.initialize_swarm()
        self._best_global_position = None
        self._best_global_fitness = None
        self.initialize_best_position()

    def initialize_swarm(self):
        particle_swarm = []
        for _ in range(self._number_of_particles):
            particle_swarm.append(Particle(self))
        return np.array(particle_swarm)

    def initialize_best_position(self):
        self._best_global_position = self._swarm[0].position.copy()
        self._best_global_fitness = self._swarm[0].fitness
        self.find_best_position()

    def find_best_position(self):
        for particle in self._swarm[1:]:
            if particle.best_local_fitness < self._best_global_fitness:
                self._best_global_fitness = particle.best_local_fitness
                self._best_global_position = particle.best_local_position.copy()

    def run(self):
        self._is_running = True
        self.log_emitter.log_signal.emit("🔹 Particle Swarm Optimization started...")

        self.log_emitter.log_signal.emit("------------------------------------")
        self.log_emitter.log_signal.emit("Selected parameters:")
        self.log_emitter.log_signal.emit(f"  Number of particles: {self._number_of_particles}")
        self.log_emitter.log_signal.emit(f"  Max iterations: {self._max_iterations}")
        self.log_emitter.log_signal.emit(f"  Inertial weight: {self._inertial_weight:.6f}")
        self.log_emitter.log_signal.emit(f"  Inertial weight flag: {self._inertial_weight_flag}")
        self.log_emitter.log_signal.emit(f"  Cognitive coefficient: {self._cognitive_coefficient:.6f}")
        self.log_emitter.log_signal.emit(f"  Social coefficient: {self._social_coefficient:.6f}")
        self.log_emitter.log_signal.emit(f"  X bounds: ({self._x_bounds[0]:.6f}, {self._x_bounds[1]:.6f})")
        self.log_emitter.log_signal.emit(f"  Y bounds: ({self._y_bounds[0]:.6f}, {self._y_bounds[1]:.6f})")
        self.log_emitter.log_signal.emit("------------------------------------")

        try:
            # Initialize points with initial swarm positions
            initial_points = np.array([particle.position.copy() for particle in self._swarm])
            self.update_signal.emit(initial_points)

            # PSO parameters
            c1 = self._cognitive_coefficient
            c2 = self._social_coefficient
            w = self._inertial_weight

            for iteration in range(self._max_iterations):
                if not self._is_running:
                    break

                # w = self._inertial_weight * (1 - iteration / self._max_iterations)

                for particle in self._swarm:
                    r1 = np.random.random(2)
                    r2 = np.random.random(2)
                    cognitive = c1 * r1 * (particle.best_local_position - particle.position)
                    social = c2 * r2 * (self._best_global_position - particle.position)

                    if self._inertial_weight_flag:
                        particle.velocity = w * particle.velocity + cognitive + social
                    else:
                        particle.velocity = particle.velocity + cognitive + social

                    # Restrict velocity to prevent excessive movements
                    max_vx = 0.1 * (self._x_bounds[1] - self._x_bounds[0])
                    max_vy = 0.1 * (self._y_bounds[1] - self._y_bounds[0])
                    particle.velocity[0] = np.clip(particle.velocity[0], -max_vx, max_vx)
                    particle.velocity[1] = np.clip(particle.velocity[1], -max_vy, max_vy)

                    particle.position = particle.position + particle.velocity

                    x_min, x_max = self._x_bounds
                    y_min, y_max = self._y_bounds
                    particle.position[0] = np.clip(particle.position[0], x_min, x_max)
                    particle.position[1] = np.clip(particle.position[1], y_min, y_max)

                    particle.fitness = particle.calculate_current_fitness()

                    if particle.fitness < particle.best_local_fitness:
                        particle.best_local_fitness = particle.fitness
                        particle.best_local_position = particle.position.copy()

                    self.find_best_position()

                current_points = np.array([particle.position.copy() for particle in self._swarm])

                message = (
                    f"Iteration {iteration + 1}:\n"
                    f"📍 Best Global Position: ({self._best_global_position[0]:.6f}, {self._best_global_position[1]:.6f})\n"
                    f"📉 Best Global Fitness: {self._best_global_fitness:.6f}\n"
                    f"🔄 Inertial Weight: {w:.6f}\n"
                    f"------------------------------------"
                )
                self.log_emitter.log_signal.emit(message)

                self.update_signal.emit(current_points)

                delay = max(self.min_delay, self.initial_delay * (0.95 ** iteration))
                time.sleep(delay)

            final_message = (
                "🎉 Particle Swarm Optimization finished!\n"
                f"🏁 Best Global Position: ({self._best_global_position[0]:.6f}, {self._best_global_position[1]:.6f})\n"
                f"📊 Best Global Fitness: {self._best_global_fitness:.6f}"
            )
            self.log_emitter.log_signal.emit(final_message)

        except Exception as e:
            self.log_emitter.log_signal.emit(f"❌ Error: {str(e)}")
        finally:
            self.update_signal.emit(np.array([particle.position.copy() for particle in self._swarm]))
            self._is_running = False
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.log_emitter.log_signal.emit("⏹ Particle Swarm Optimization stopped by user")

    # Getters
    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def swarm(self):
        return self._swarm

    @property
    def function(self):
        return self._function

    @property
    def number_of_particles(self):
        return self._number_of_particles

    @property
    def inertial_weight(self):
        return self._inertial_weight

    @property
    def inertial_weight_flag(self):
        return self._inertial_weight_flag

    @property
    def cognitive_coefficient(self):
        return self._cognitive_coefficient

    @property
    def social_coefficient(self):
        return self._social_coefficient

    @property
    def x_bounds(self):
        return self._x_bounds

    @property
    def y_bounds(self):
        return self._y_bounds

    @property
    def best_global_position(self):
        return self._best_global_position

    @property
    def best_global_fitness(self):
        return self._best_global_fitness

    @property
    def is_running(self):
        return self._is_running

    @property
    def log_emitter(self):
        return self._log_emitter

    @property
    def initial_delay(self):
        return self._initial_delay

    @property
    def min_delay(self):
        return self._min_delay

    # Setters
    @max_iterations.setter
    def max_iterations(self, value):
        if value is None or value <= 0:
            raise ValueError("Max iterations must be a positive integer")
        self._max_iterations = value

    @swarm.setter
    def swarm(self, value):
        if value is None:
            raise ValueError("Swarm cannot be None")
        self._swarm = value

    @function.setter
    def function(self, value):
        if value is None:
            raise ValueError("Function cannot be None")
        self._function = value

    @number_of_particles.setter
    def number_of_particles(self, value):
        if value is None or value <= 0:
            raise ValueError("Number of particles must be a positive integer")
        self._number_of_particles = value

    @inertial_weight.setter
    def inertial_weight(self, value):
        if value is None or value <= 0:
            raise ValueError("Inertial weight must be positive")
        self._inertial_weight = value

    @inertial_weight_flag.setter
    def inertial_weight_flag(self, value):
        if value is None:
            raise ValueError("Inertial weight flag cannot be None")
        self._inertial_weight_flag = value

    @cognitive_coefficient.setter
    def cognitive_coefficient(self, value):
        if value is None or value <= 0:
            raise ValueError("Cognitive coefficient must be positive")
        self._cognitive_coefficient = value

    @social_coefficient.setter
    def social_coefficient(self, value):
        if value is None or value <= 0:
            raise ValueError("Social coefficient must be positive")
        self._social_coefficient = value

    @x_bounds.setter
    def x_bounds(self, value):
        if value is None or len(value) != 2 or value[0] >= value[1]:
            raise ValueError("X bounds must be a tuple of (min, max) where min < max")
        self._x_bounds = value

    @y_bounds.setter
    def y_bounds(self, value):
        if value is None or len(value) != 2 or value[0] >= value[1]:
            raise ValueError("Y bounds must be a tuple of (min, max) where min < max")
        self._y_bounds = value

    @best_global_position.setter
    def best_global_position(self, value):
        if value is not None and not isinstance(value, (np.ndarray, list)) or len(value) != 2:
            raise ValueError("Best global position must be a 2D array or list")
        self._best_global_position = value

    @best_global_fitness.setter
    def best_global_fitness(self, value):
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Best global fitness must be a number")
        self._best_global_fitness = value

    @is_running.setter
    def is_running(self, value):
        if not isinstance(value, bool):
            raise ValueError("is_running must be a boolean")
        self._is_running = value

    @log_emitter.setter
    def log_emitter(self, value):
        if value is None:
            raise ValueError("Log emitter cannot be None")
        self._log_emitter = value

    @initial_delay.setter
    def initial_delay(self, value):
        if value is None or value <= 0:
            raise ValueError("Initial delay must be positive")
        self._initial_delay = value

    @min_delay.setter
    def min_delay(self, value):
        if value is None or value <= 0:
            raise ValueError("Min delay must be positive")
        self._min_delay = value
