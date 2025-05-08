import numpy as np


def rosenbrock_function(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def initialize_population(population_size, x_bounds, y_bounds):
    population = np.zeros((population_size, 2))
    population[:, 0] = np.random.uniform(x_bounds[0], x_bounds[1], population_size)
    population[:, 1] = np.random.uniform(y_bounds[0], y_bounds[1], population_size)
    return population


def select_parents(population):
    fitness = np.array([rosenbrock_function(x, y) for x, y in population])
    fitness = 1 / (fitness + 1e-10)
    # В задаче поиска минимума функции при использовании оператора рулетки
    # Пригодность будет определяться как 1 / значение функции в этой точке
    # Это необходимо делать, поскольку перед нами стоит задача минимизации функции
    # А значит в случае, если значение функции будет достаточно мало
    # То fitness, то есть приспособленность особи будет наоборот велико, а значит эта особь лучше других

    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness

    parent_indices = np.random.choice(
        len(population),
        size=2,
        p=probabilities
    )

    parents = population[parent_indices]
    return parents


def intermediate_recombination(parents, coefficient=0.25):
    first_parent, second_parent = parents
    num_genes = len(first_parent)

    first_descendant = np.zeros(2)
    second_descendant = np.zeros(2)

    for gen_index in range(num_genes):
        alpha_for_first_parent = np.random.uniform(-coefficient, 1 + coefficient)
        alpha_for_second_parent = np.random.uniform(-coefficient, 1 + coefficient)

        first_descendant[gen_index] = first_parent[gen_index] + alpha_for_first_parent * (second_parent[gen_index] - first_parent[gen_index])
        second_descendant[gen_index] = first_parent[gen_index] + alpha_for_second_parent * (second_parent[gen_index] - first_parent[gen_index])

    return np.array([first_descendant, second_descendant])


def genetic_algorithm(population, number_of_generations=100):
    parents = select_parents(population)

    descendants = intermediate_recombination(parents)

    print(f"Parents: \n{parents}")
    print()
    print(f"Descendants: \n{descendants}")


def main():
    x_bounds = [-5, 5]
    y_bounds = [-5, 5]
    population_size = 100

    population = initialize_population(population_size, x_bounds, y_bounds)
    genetic_algorithm(population)


if __name__ == "__main__":
    main()