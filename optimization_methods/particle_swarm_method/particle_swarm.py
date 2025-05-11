import numpy as np

from particle import Particle


class ParticleSwarm:
    def __init__(self, params_dict):
        self._number_of_particles = params_dict['number_of_particles']
        self._max_iterations = params_dict['max_iterations']
        self._inertial_weight = params_dict['inertial_weight']
        self._inertial_weight_flag = params_dict['inertial_weight_flag']
        self._cognitive_coefficient = params_dict['cognitive_coefficient']
        self._social_coefficient = params_dict['social_coefficient']
        self._x_bounds = params_dict['x_bounds']
        self._y_bounds = params_dict['y_bounds']
        self._function = params_dict['function']
        self._swarm = self.initialize_swarm()
        self._best_global_position = None
        self._best_global_fitness = None
        # Initialize global best
        self.initialize_best_position()

    def initialize_swarm(self):
        particle_swarm = []
        for _ in range(self._number_of_particles):
            particle_swarm.append(Particle(self))
        return np.array(particle_swarm)

    def initialize_best_position(self):
        # Initialize with the first particle's values
        self._best_global_position = self._swarm[0].position.copy()
        self._best_global_fitness = self._swarm[0].fitness

    def find_best_position(self):
        for particle in self._swarm:
            if particle.fitness < self._best_global_fitness:
                self._best_global_fitness = particle.fitness
                self._best_global_position = particle.position.copy()

    def run(self):
        # PSO parameters
        c1 = self._cognitive_coefficient  # Cognitive parameter
        c2 = self._social_coefficient  # Social parameter
        w = self._inertial_weight  # Initial inertial weight

        for iteration in range(self._max_iterations):
            # Update inertial weight if flag is True (linear decrease)
            if self._inertial_weight_flag:
                w = self._inertial_weight * (1 - iteration / self._max_iterations)

            for particle in self._swarm:
                # Update velocity
                r1 = np.random.random(2)  # Random vector for cognitive component
                r2 = np.random.random(2)  # Random vector for social component
                cognitive = c1 * r1 * (particle.best_local_position - particle.position)
                social = c2 * r2 * (self._best_global_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive + social

                # Update position
                particle.position = particle.position + particle.velocity

                # Clip position to stay within bounds
                x_min, x_max = self._x_bounds
                y_min, y_max = self._y_bounds
                particle.position[0] = np.clip(particle.position[0], x_min, x_max)
                particle.position[1] = np.clip(particle.position[1], y_min, y_max)

                # Update fitness
                particle.fitness = particle.calculate_current_fitness()

                # Update local best
                if particle.fitness < particle.best_local_fitness:
                    particle.best_local_fitness = particle.fitness
                    particle.best_local_position = particle.position.copy()

            # Recalculate global best at the end of each iteration
            self.find_best_position()

        return self._best_global_position, self._best_global_fitness

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
        if value is None:
            raise ValueError("Inertial weight cannot be None")
        if value <= 0:
            raise ValueError("Inertial weight must be positive")
        self._inertial_weight = value

    @inertial_weight_flag.setter
    def inertial_weight_flag(self, value):
        if value is None:
            raise ValueError("Inertial weight flag cannot be None")
        self._inertial_weight_flag = value

    @cognitive_coefficient.setter
    def cognitive_coefficient(self, value):
        if value is None:
            raise ValueError("Cognitive coefficient cannot be None")
        if value <= 0:
            raise ValueError("Cognitive coefficient must be positive")
        self._cognitive_coefficient = value

    @social_coefficient.setter
    def social_coefficient(self, value):
        if value is None:
            raise ValueError("Social coefficient cannot be None")
        if value <= 0:
            raise ValueError("Social coefficient must be positive")
        self._social_coefficient = value

    @x_bounds.setter
    def x_bounds(self, value):
        if value is None:
            raise ValueError("X bounds cannot be None")
        if len(value) != 2 or value[0] >= value[1]:
            raise ValueError("X bounds must be a tuple of (min, max) where min < max")
        self._x_bounds = value

    @y_bounds.setter
    def y_bounds(self, value):
        if value is None:
            raise ValueError("Y bounds cannot be None")
        if len(value) != 2 or value[0] >= value[1]:
            raise ValueError("Y bounds must be a tuple of (min, max) where min < max")
        self._y_bounds = value

    @best_global_position.setter
    def best_global_position(self, value):
        if value is not None and not isinstance(value, (np.ndarray, list)) and len(value) != 2:
            raise ValueError("Best global position must be a 2D array or list or None")
        self._best_global_position = value

    @best_global_fitness.setter
    def best_global_fitness(self, value):
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Best global fitness must be a number or None")
        self._best_global_fitness = value
