import numpy as np

class Bee:
    def __init__(self, swarm):
        self._bee_swarm = swarm
        self._position = self.initialize_position()
        self._fitness = self.calculate_current_fitness()

    def initialize_position(self):
        x_min, x_max = self._bee_swarm.x_bounds
        y_min, y_max = self._bee_swarm.y_bounds

        center_x = np.random.uniform(x_min + 0.1 * (x_max - x_min), x_max - 0.1 * (x_max - x_min))
        center_y = np.random.uniform(y_min + 0.1 * (y_max - y_min), y_max - 0.1 * (y_max - y_min))

        spread = 0.1 * min(x_max - x_min, y_max - y_min)

        x = np.random.normal(center_x, spread)
        y = np.random.normal(center_y, spread)

        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)

        return np.array([x, y])

    def calculate_current_fitness(self):
        return self._bee_swarm.function(self._position[0], self._position[1])

    # Getters
    @property
    def position(self):
        return self._position

    @property
    def fitness(self):
        return self._fitness

    @property
    def bee_swarm(self):
        return self._bee_swarm

    # Setters
    @position.setter
    def position(self, new_position):
        if isinstance(new_position, np.ndarray) and new_position.shape == (2,):
            self._position = new_position
            self._fitness = self.calculate_current_fitness()
        else:
            raise ValueError("Position must be a numpy array of shape (2,)")

    @fitness.setter
    def fitness(self, new_fitness):
        if isinstance(new_fitness, (int, float)):
            self._fitness = new_fitness
        else:
            raise ValueError("Fitness must be a numeric value")

    def __str__(self):
        return (f"Bee:\n"
                f"  Position: ({self._position[0]:.4f}, {self._position[1]:.4f})\n"
                f"  Fitness: {self._fitness:.4f}")

