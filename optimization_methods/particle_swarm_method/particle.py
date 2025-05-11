import numpy as np

class Particle:
    def __init__(self, particle_swarm):
        self._particle_swarm = particle_swarm
        self._position = self.initialize_position()
        self._fitness = self.calculate_current_fitness()
        self._velocity = self.initialize_velocity()
        self._best_local_position = self.position.copy()
        self._best_local_fitness = self.fitness

    def initialize_position(self):
        x_min, x_max = self._particle_swarm.x_bounds
        y_min, y_max = self._particle_swarm.y_bounds

        center_x = np.random.uniform(x_min + 0.2 * (x_max - x_min), x_max - 0.2 * (x_max - x_min))
        center_y = np.random.uniform(y_min + 0.2 * (y_max - y_min), y_max - 0.2 * (y_max - y_min))

        spread = 0.1 * min(x_max - x_min, y_max - y_min)

        x = np.random.normal(center_x, spread)
        y = np.random.normal(center_y, spread)

        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)

        return np.array([x, y])

    def initialize_velocity(self):
        x_min, x_max = self._particle_swarm.x_bounds
        y_min, y_max = self._particle_swarm.y_bounds

        max_vx = 0.1 * (x_max - x_min)
        max_vy = 0.1 * (y_max - y_min)

        vx = np.random.uniform(-max_vx, max_vx)
        vy = np.random.uniform(-max_vy, max_vy)

        return np.array([vx, vy])

    def calculate_current_fitness(self):
        return self._particle_swarm.function(self.position[0], self.position[1])

    def __str__(self):
        return (f"Particle:\n"
                f"  Position: ({self._position[0]:.4f}, {self._position[1]:.4f})\n"
                f"  Velocity: ({self._velocity[0]:.4f}, {self._velocity[1]:.4f})\n"
                f"  Fitness: {self._fitness:.4f}\n"
                f"  Best Local Position: ({self._best_local_position[0]:.4f}, {self._best_local_position[1]:.4f})\n"
                f"  Best Local Fitness: {self._best_local_fitness:.4f}")

    # Getters
    @property
    def particle_swarm(self):
        return self._particle_swarm

    @property
    def position(self):
        return self._position

    @property
    def fitness(self):
        return self._fitness

    @property
    def velocity(self):
        return self._velocity

    @property
    def best_local_position(self):
        return self._best_local_position

    @property
    def best_local_fitness(self):
        return self._best_local_fitness

    # Setters
    @particle_swarm.setter
    def particle_swarm(self, value):
        self._particle_swarm = value

    @position.setter
    def position(self, value):
        self._position = value

    @fitness.setter
    def fitness(self, value):
        self._fitness = value

    @velocity.setter
    def velocity(self, value):
        self._velocity = value

    @best_local_position.setter
    def best_local_position(self, value):
        self._best_local_position = value

    @best_local_fitness.setter
    def best_local_fitness(self, value):
        self._best_local_fitness = value
