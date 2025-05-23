import numpy as np


def rosenbrock_function(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def initialize_population(population_size, x_bounds, y_bounds):
    population = np.zeros((population_size, 2))
    population[:, 0] = np.random.uniform(x_bounds[0], x_bounds[1], population_size)
    population[:, 1] = np.random.uniform(y_bounds[0], y_bounds[1], population_size)
    return population


def roulette_method(population):
    population_size = len(population)

    fitness = np.array([rosenbrock_function(x, y) for x, y in population])
    fitness = 1 / (fitness + 1e-10)

    total_fitness = np.sum(fitness)

    if total_fitness == 0:
        probabilities = np.ones(population_size) / population_size
    else:
        probabilities = fitness / total_fitness

    for index in range(len(probabilities)):
        if np.isnan(probabilities[index]):
            probabilities[index] = 0

    parent_indices = np.random.choice(
        population_size,
        size=2,
        p=probabilities
    )

    parents = population[parent_indices]
    return parents


def intermediate_recombination(parents, coefficient=0.25):
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


def real_valued_mutation(descendant, bounds=[[-5, 5], [-5, 5]], m=20):
    mutated_descendant = descendant.copy()

    for gen_index in range(len(descendant)):
        # Вычисляем alpha = 0.5 * размер поискового пространства
        search_space_size = bounds[gen_index][1] - bounds[gen_index][0]  # Например, 5 - (-5) = 10
        alpha = 0.5 * search_space_size

        # Формируем величину delta по формуле
        delta = 0
        for index in range(1, m + 1):
            if np.random.random() < 1 / m:
                alpha_value = 1
            else:
                alpha_value = 0
            delta += alpha_value * (2 ** (-index))

        # Выбираем знак (+ или -) с равной вероятностью
        if np.random.random() < 0.5:
            sign = 1
        else:
            sign = -1

        # Применяем мутацию: новый_ген = старый_ген + sign * alpha * beta
        mutated_descendant[gen_index] = descendant[gen_index] + sign * alpha * delta

        # Ограничиваем значения в пределах поискового пространства
        mutated_descendant[gen_index] = np.clip(mutated_descendant[gen_index], bounds[gen_index][0], bounds[gen_index][1])

    return mutated_descendant


def check_convergence(population, std_threshold=0.005):
    std_x = np.std(population[:, 0])
    std_y = np.std(population[:, 1])
    return std_x < std_threshold and std_y < std_threshold


def genetic_algorithm(population, probability_of_recombination, probability_of_mutation, number_of_generations=1000):
    for generation in range(number_of_generations):
        new_population = []

        while len(new_population) < len(population):
            parents = roulette_method(population)

            if np.random.rand() < probability_of_recombination:
                descendants = intermediate_recombination(parents)
            else:
                descendants = parents

            for descendant in descendants:
                if np.random.rand() < probability_of_mutation:
                    mutated_descendant = real_valued_mutation(descendant)
                    new_population.append(mutated_descendant)
                else:
                    new_population.append(descendant)

        population = np.array(new_population)

        if check_convergence(population):
            print(f"Convergence is achieved on generation {generation + 1}")
            break

    return population


def main():
    x_bounds = [-5, 5]
    y_bounds = [-5, 5]
    population_size = 100
    probability_of_recombination = 0.8
    probability_of_mutation = 0.1

    population = initialize_population(population_size, x_bounds, y_bounds)
    print("Initial Population:")
    for i in range(population_size):
        x, y = population[i]
        print(f"Individual (Point) {i + 1}: x = {x:.4f}, y = {y:.4f}, f(x,y) = {rosenbrock_function(x, y):.4f}")

    print()

    new_population = genetic_algorithm(population, probability_of_recombination, probability_of_mutation)
    print("New Population:")
    for i in range(population_size):
        x, y = new_population[i]
        print(f"Individual (Point) {i + 1}: x = {x:.4f}, y = {y:.4f}, f(x,y) = {rosenbrock_function(x, y):.4f}")


if __name__ == "__main__":
    main()
