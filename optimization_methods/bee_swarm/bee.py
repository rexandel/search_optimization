import numpy as np

class Bee:
    def __init__(self, swarm):
        self._bee_swarm = swarm
        self._position = None
        self._fitness = None

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
        if self._position is None:
            return None
        return self._bee_swarm.function(self._position[0], self._position[1])

    def move_to_random_nearby_point(self, target_point):
        if not isinstance(target_point, np.ndarray) or target_point.shape != (2,):
            raise ValueError("Target point must be a numpy array of shape (2,)")

        area_size = self._bee_swarm.size_of_area

        offset = np.random.uniform(-area_size, area_size, size=2)
        new_position = target_point + offset

        x_min, x_max = self._bee_swarm.x_bounds
        y_min, y_max = self._bee_swarm.y_bounds

        new_position[0] = np.clip(new_position[0], x_min, x_max)
        new_position[1] = np.clip(new_position[1], y_min, y_max)

        self.position = new_position
    
    def move_to_random_point(self):
        x_min, x_max = self._bee_swarm.x_bounds
        y_min, y_max = self._bee_swarm.y_bounds

        center_x = np.random.uniform(x_min + 0.1 * (x_max - x_min), x_max - 0.1 * (x_max - x_min))
        center_y = np.random.uniform(y_min + 0.1 * (y_max - y_min), y_max - 0.1 * (y_max - y_min))

        spread = 0.1 * min(x_max - x_min, y_max - y_min)

        x = np.random.normal(center_x, spread)
        y = np.random.normal(center_y, spread)

        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)
        
        new_position = np.array([x, y])
        self.position = new_position

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
        if new_position is None:
            self._position = None
            self._fitness = None  # Если позиция сбрасывается, фитнес тоже
            return

        if isinstance(new_position, np.ndarray) and new_position.shape == (2,):
            self._position = new_position
            self._fitness = self.calculate_current_fitness()
        else:
            raise ValueError("Position must be a numpy array of shape (2,) or None")

    @fitness.setter
    def fitness(self, new_fitness):
        if isinstance(new_fitness, (int, float)):
            self._fitness = new_fitness
        else:
            raise ValueError("Fitness must be a numeric value")

    def __str__(self):
        if self._position is not None and self._fitness is not None:
            return (f"Bee: Position: ({self._position[0]:.2f}, {self._position[1]:.2f}), Function Value: {self._fitness:.2f}")
        elif self._position is not None:
            return (f"Bee: Position: ({self._position[0]:.2f}, {self._position[1]:.2f}), Function Value: N/A")
        else:
            return "Bee: In hive (no position/fitness)"
