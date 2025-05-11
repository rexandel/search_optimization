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
        self.find_best_position()

    def initialize_swarm(self):
        particle_swarm = []
        for _ in range(self._number_of_particles):
            particle_swarm.append(Particle(self))
        return np.array(particle_swarm)

    def find_best_position(self):
        # Initialize with the first particle's values
        self._best_global_position = self._swarm[0].position.copy()
        self._best_global_fitness = self._swarm[0].fitness

        # Iterate through the swarm to find the best
        for particle in self._swarm[1:]:
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

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        if value is None or value <= 0:
            raise ValueError("Max iterations must be a positive integer")
        self._max_iterations = value

    @property
    def swarm(self):
        return self._swarm

    @swarm.setter
    def swarm(self, value):
        if value is None:
            raise ValueError("Swarm cannot be None")
        self._swarm = value

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        if value is None:
            raise ValueError("Function cannot be None")
        self._function = value

    @property
    def number_of_particles(self):
        return self._number_of_particles

    @number_of_particles.setter
    def number_of_particles(self, value):
        if value is None or value <= 0:
            raise ValueError("Number of particles must be a positive integer")
        self._number_of_particles = value

    @property
    def inertial_weight(self):
        return self._inertial_weight

    @inertial_weight.setter
    def inertial_weight(self, value):
        if value is None:
            raise ValueError("Inertial weight cannot be None")
        self._inertial_weight = value

    @property
    def inertial_weight_flag(self):
        return self._inertial_weight_flag

    @inertial_weight_flag.setter
    def inertial_weight_flag(self, value):
        self._inertial_weight_flag = value

    @property
    def x_bounds(self):
        return self._x_bounds

    @x_bounds.setter
    def x_bounds(self, value):
        if value is None:
            raise ValueError("X bounds cannot be None")
        self._x_bounds = value

    @property
    def y_bounds(self):
        return self._y_bounds

    @y_bounds.setter
    def y_bounds(self, value):
        if value is None:
            raise ValueError("Y bounds cannot be None")
        self._y_bounds = value