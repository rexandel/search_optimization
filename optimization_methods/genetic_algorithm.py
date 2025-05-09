from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import time


class GeneticAlgorithm(QObject):
    finished_signal = pyqtSignal()
    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, params_dict, log_emitter):
        super().__init__()
        self._population_size = params_dict['population_size']
        self._max_generations = params_dict['max_generations']
        self._std_threshold = params_dict['std_threshold']
        self._x_bounds = params_dict['x_bounds']
        self._y_bounds = params_dict['y_bounds']
        self._probability_of_recombination = params_dict['probability_of_recombination']
        self._probability_of_mutation = params_dict['probability_of_mutation']
        self._function = params_dict['function']

        self._roulette_method_flag = params_dict['roulette_method_flag']
        self._tournament_method_flag = params_dict['tournament_method_flag']

        self._intermediate_recombination_flag = params_dict['intermediate_recombination_flag']
        self._line_recombination_flag = params_dict['line_recombination_flag']

        self._truncation_threshold = params_dict['truncation_threshold']
        self._truncation_threshold_flag = params_dict['truncation_threshold_flag']

        self._bolzman_threshold = params_dict['bolzman_threshold']
        self._bolzman_threshold_flag = params_dict['bolzman_threshold_flag']

        self._is_running = False
        self.points = []
        self.log_emitter = log_emitter
        self.initial_delay = 0.1
        self.min_delay = 0.01

    def _initialize_population(self):
        population = np.zeros((self._population_size, 2))
        population[:, 0] = np.random.uniform(self._x_bounds[0], self._x_bounds[1], self._population_size)
        population[:, 1] = np.random.uniform(self._y_bounds[0], self._y_bounds[1], self._population_size)
        return population

    def _roulette_method(self, population):
        population_size = len(population)
        fitness = np.array([self._function(x, y) for x, y in population])
        fitness = 1 / (fitness + 1e-10)
        total_fitness = np.sum(fitness)

        if total_fitness == 0:
            probabilities = np.ones(population_size) / population_size
        else:
            probabilities = fitness / total_fitness

        probabilities = np.nan_to_num(probabilities, nan=0.0)
        parent_indices = np.random.choice(population_size, size=2, p=probabilities)
        parents = population[parent_indices]
        return parents

    def _tournament_method(self, population, tournament_size=2):
        population_size = len(population)
        intermediate_pool = []

        # Выполняем N турниров для формирования промежуточного массива
        for _ in range(population_size):
            # Случайно выбираем t особей для турнира
            tournament_indices = np.random.choice(population_size, size=tournament_size, replace=False)
            # Вычисляем значения целевой функции для выбранных особей
            tournament_fitness = np.array([self._function(population[idx][0], population[idx][1])
                                           for idx in tournament_indices])
            # Находим индекс лучшей особи (с минимальным значением функции)
            best_idx = tournament_indices[np.argmin(tournament_fitness)]
            # Добавляем лучшую особь в промежуточный массив
            intermediate_pool.append(population[best_idx])

        # Преобразуем промежуточный массив в numpy массив
        intermediate_pool = np.array(intermediate_pool)
        # Случайно выбираем двух родителей из промежуточного массива
        parent_indices = np.random.choice(population_size, size=2, replace=False)
        parents = intermediate_pool[parent_indices]
        return parents

    def _intermediate_recombination(self, parents, coefficient=0.25):
        first_parent, second_parent = parents
        num_genes = len(first_parent)
        first_descendant = np.zeros(num_genes)
        second_descendant = np.zeros(num_genes)

        for gen_index in range(num_genes):
            alpha_for_first_parent = np.random.uniform(-coefficient, 1 + coefficient)
            alpha_for_second_parent = np.random.uniform(-coefficient, 1 + coefficient)
            first_descendant[gen_index] = first_parent[gen_index] + alpha_for_first_parent * (
                    second_parent[gen_index] - first_parent[gen_index])
            second_descendant[gen_index] = first_parent[gen_index] + alpha_for_second_parent * (
                    second_parent[gen_index] - first_parent[gen_index])
        return np.array([first_descendant, second_descendant])

    def _line_recombination(self, parents, coefficient=0.25):
        first_parent, second_parent = parents
        num_genes = len(first_parent)
        first_descendant = np.zeros(num_genes)
        second_descendant = np.zeros(num_genes)

        alpha_for_first_parent = np.random.uniform(-coefficient, 1 + coefficient)
        alpha_for_second_parent = np.random.uniform(-coefficient, 1 + coefficient)

        for gen_index in range(num_genes):
            first_descendant[gen_index] = first_parent[gen_index] + alpha_for_first_parent * (
                    second_parent[gen_index] - first_parent[gen_index])
            second_descendant[gen_index] = first_parent[gen_index] + alpha_for_second_parent * (
                    second_parent[gen_index] - first_parent[gen_index])
        return np.array([first_descendant, second_descendant])

    def _real_valued_mutation(self, descendant, m=20):
        mutated_descendant = descendant.copy()
        bounds = [self._x_bounds, self._y_bounds]

        for gen_index in range(len(descendant)):
            search_space_size = bounds[gen_index][1] - bounds[gen_index][0]
            alpha = 0.5 * search_space_size
            delta = 0
            for index in range(1, m + 1):
                alpha_value = 1 if np.random.random() < 1 / m else 0
                delta += alpha_value * (2 ** (-index))
            sign = 1 if np.random.random() < 0.5 else -1
            mutated_descendant[gen_index] = descendant[gen_index] + sign * alpha * delta
            mutated_descendant[gen_index] = np.clip(mutated_descendant[gen_index], bounds[gen_index][0],
                                                    bounds[gen_index][1])
        return mutated_descendant

    def _truncation_selection(self, population, descendants):
        combined_population = np.vstack((population, descendants))
        fitness = np.array([self._function(x, y) for x, y in combined_population])

        # Функция np.argsort() возвращает массив индексов, которые указывают,
        # в каком порядке взять элементы, чтобы они были отсортированы по возрастанию.
        # В данном случае в качестве аргумента выступает массив значений функций
        sorted_indices = np.argsort(fitness)

        # Вычисляем количество особей, среди которых будет производиться отбор в новую популяцию
        if self._truncation_threshold <= 1:
            num_select = int(len(combined_population) * self._truncation_threshold)
        else:
            num_select = int(self._truncation_threshold)

        # Гарантируем, что размер промежуточной популяции будет хотя бы 1, или равен размеру популяции
        num_select = max(1, min(num_select, len(combined_population)))

        # Выбираем только тех особей, которые находятся в первых num_select
        selected_indices = sorted_indices[:num_select]

        # Выбираем случайным образом новых особей с необходимыми индексами
        new_population_indices = (np.random.choice(selected_indices, size=self._population_size, replace=True))
        new_population = combined_population[new_population_indices]

        return new_population

    def _bolzman_selection(self, population, descendants):
        combined_population = np.vstack((population, descendants))
        population_size = len(combined_population)

        fitness = np.array([self._function(x, y) for x, y in combined_population])

        new_population = []
        temperature = self._bolzman_threshold

        while len(new_population) < self._population_size:
            # Случайно выбираем двух особей i и j
            i, j = np.random.choice(population_size, size=2, replace=False)

            # Вычисляем вероятность p по формуле Болцмана
            delta_function = fitness[i] - fitness[j]
            probability = 1 / (1 + np.exp(delta_function / temperature))

            # Генерируем случайное число r из (0, 1)
            random_value = np.random.random()

            # Выбираем особь: i, если p > r, иначе j
            if probability > random_value:
                selected_index = i
            else:
                selected_index = j

            new_population.append(combined_population[selected_index])

        new_population = np.array(new_population)
        return new_population

    def _check_convergence(self, population):
        std_x = np.std(population[:, 0])
        std_y = np.std(population[:, 1])
        return std_x < self._std_threshold and std_y < self._std_threshold

    def run(self):
        self._is_running = True
        self.log_emitter.log_signal.emit("🔹 Genetic Algorithm started...")

        self.log_emitter.log_signal.emit("------------------------------------\n")
        self.log_emitter.log_signal.emit(" Selected parameters:")

        if self._roulette_method_flag:
            message = (
                f"  Parents selection method is Roulette Method"
            )
            self.log_emitter.log_signal.emit(message)
        elif self._tournament_method_flag:
            message = (
                f"  Parents selection method is Tournament Method"
            )
            self.log_emitter.log_signal.emit(message)

        if self._intermediate_recombination_flag:
            message = (
                f"  Recombination method is Intermediate Recombination"
            )
            self.log_emitter.log_signal.emit(message)
        elif self._line_recombination_flag:
            message = (
                f"  Recombination method is Line Recombination"
            )
            self.log_emitter.log_signal.emit(message)

        if self._truncation_threshold_flag:
            message = (
                f"  Selection method is Truncation Selection"
            )
            self.log_emitter.log_signal.emit(message)
        elif self._bolzman_threshold_flag:
            message = (
                f"  Selection method is Bolzman Selection"
            )
            self.log_emitter.log_signal.emit(message)

        self.log_emitter.log_signal.emit("------------------------------------\n")

        try:
            population = self._initialize_population()
            self.points = [population.copy()]

            for generation in range(self._max_generations):
                if not self._is_running:
                    break

                descendants = []
                while len(descendants) < self._population_size:

                    if self._roulette_method_flag:
                        parents = self._roulette_method(population)
                    elif self._tournament_method_flag:
                        parents = self._tournament_method(population)

                    if np.random.rand() < self._probability_of_recombination:
                        if self._intermediate_recombination_flag:
                            new_descendants = self._intermediate_recombination(parents)
                        elif self._line_recombination_flag:
                            new_descendants = self._line_recombination(parents)
                    else:
                        new_descendants = parents

                    for descendant in new_descendants:
                        if np.random.rand() < self._probability_of_mutation:
                            mutated_descendant = self._real_valued_mutation(descendant)
                            descendants.append(mutated_descendant)
                        else:
                            descendants.append(descendant)

                descendants = np.array(descendants[:self._population_size])

                if self._truncation_threshold_flag:
                    population = self._truncation_selection(population, descendants)
                elif self._bolzman_threshold_flag:
                    population = self._bolzman_selection(population, descendants)

                self.points.append(population.copy())

                best_idx = np.argmin([self._function(x, y) for x, y in population])
                best_point = population[best_idx]
                best_value = self._function(best_point[0], best_point[1])

                message = (
                    f"Generation {generation + 1}:\n"
                    f"📍 Best Point: ({best_point[0]:.6f}, {best_point[1]:.6f})\n"
                    f"📉 Best Function value: {best_value:.6f}\n"
                    f"------------------------------------\n"
                )
                self.log_emitter.log_signal.emit(message)

                if self._check_convergence(population):
                    self.log_emitter.log_signal.emit(
                        f"✅ Convergence achieved at generation {generation + 1}"
                    )
                    break

                self.update_signal.emit(population.copy())
                delay = max(self.min_delay, self.initial_delay * (0.95 ** generation))
                time.sleep(delay)

            self.points = np.array(self.points)
            final_message = (
                "🎉 Genetic Algorithm finished!\n"
                f"🏁 Best point: ({best_point[0]:.6f}, {best_point[1]:.6f})\n"
                f"📊 Best value: {best_value:.6f}"
            )
            self.log_emitter.log_signal.emit(final_message)

        except Exception as e:
            self.log_emitter.log_signal.emit(f"❌ Error: {str(e)}")
        finally:
            self.update_signal.emit(population)
            self._is_running = False
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.log_emitter.log_signal.emit("⏹ Genetic Algorithm stopped by user")

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Population size must be positive integer")
        self._population_size = value

    @property
    def max_generations(self):
        return self._max_generations

    @max_generations.setter
    def max_generations(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Max generations must be positive integer")
        self._max_generations = value

    @property
    def std_threshold(self):
        return self._std_threshold

    @std_threshold.setter
    def std_threshold(self, value):
        if not isinstance(value, float) or value <= 0:
            raise ValueError("Standard deviation threshold must be positive float")
        self._std_threshold = value

    @property
    def x_bounds(self):
        return self._x_bounds

    @x_bounds.setter
    def x_bounds(self, value):
        if not isinstance(value, (list, tuple)) or len(value) != 2 or value[0] >= value[1]:
            raise ValueError("X bounds must be a list/tuple of two numbers where lower < upper")
        self._x_bounds = value

    @property
    def y_bounds(self):
        return self._y_bounds

    @y_bounds.setter
    def y_bounds(self, value):
        if not isinstance(value, (list, tuple)) or len(value) != 2 or value[0] >= value[1]:
            raise ValueError("Y bounds must be a list/tuple of two numbers where lower < upper")
        self._y_bounds = value

    @property
    def probability_of_recombination(self):
        return self._probability_of_recombination

    @probability_of_recombination.setter
    def probability_of_recombination(self, value):
        if not isinstance(value, float) or not (0 <= value <= 1):
            raise ValueError("Recombination probability must be float between 0 and 1")
        self._probability_of_recombination = value

    @property
    def probability_of_mutation(self):
        return self._probability_of_mutation

    @probability_of_mutation.setter
    def probability_of_mutation(self, value):
        if not isinstance(value, float) or not (0 <= value <= 1):
            raise ValueError("Mutation probability must be float between 0 and 1")
        self._probability_of_mutation = value

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        if value is None:
            raise ValueError("Function cannot be None")
        self._function = value

    @property
    def truncation_threshold(self):
        return self._truncation_threshold

    @truncation_threshold.setter
    def truncation_threshold(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Truncation threshold must be positive")
        self._truncation_threshold = value

    @property
    def bolzman_threshold(self):
        return self._bolzman_threshold

    @bolzman_threshold.setter
    def bolzman_threshold(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Bolzman threshold must be positive")
        self._bolzman_threshold = value

    @property
    def tournament_size(self):
        return self._tournament_size

    @tournament_size.setter
    def tournament_size(self, value):
        if not isinstance(value, int) or value < 2 or value > self._population_size:
            raise ValueError("Tournament size must be an integer >= 2 and <= population size")
        self._tournament_size = value

    @property
    def roulette_method_flag(self):
        return self._roulette_method_flag

    @roulette_method_flag.setter
    def roulette_method_flag(self, value):
        if not isinstance(value, bool):
            raise ValueError("Roulette method flag must be a boolean")
        self._roulette_method_flag = value

    @property
    def tournament_method_flag(self):
        return self._tournament_method_flag

    @tournament_method_flag.setter
    def tournament_method_flag(self, value):
        if not isinstance(value, bool):
            raise ValueError("Tournament method flag must be a boolean")
        self._tournament_method_flag = value

    @property
    def intermediate_recombination_flag(self):
        return self._intermediate_recombination_flag

    @intermediate_recombination_flag.setter
    def intermediate_recombination_flag(self, value):
        if not isinstance(value, bool):
            raise ValueError("Intermediate recombination flag must be a boolean")
        self._intermediate_recombination_flag = value

    @property
    def line_recombination_flag(self):
        return self._line_recombination_flag

    @line_recombination_flag.setter
    def line_recombination_flag(self, value):
        if not isinstance(value, bool):
            raise ValueError("Line recombination flag must be a boolean")
        self._line_recombination_flag = value

    @property
    def truncation_threshold_flag(self):
        return self._truncation_threshold_flag

    @truncation_threshold_flag.setter
    def truncation_threshold_flag(self, value):
        if not isinstance(value, bool):
            raise ValueError("Truncation threshold flag must be a boolean")
        self._truncation_threshold_flag = value

    @property
    def bolzman_threshold_flag(self):
        return self._bolzman_threshold_flag

    @bolzman_threshold_flag.setter
    def bolzman_threshold_flag(self, value):
        if not isinstance(value, bool):
            raise ValueError("Bolzman threshold flag must be a boolean")
        self._bolzman_threshold_flag = value