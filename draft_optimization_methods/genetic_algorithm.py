import numpy as np


def rosenbrock_function(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def initialize_population(pop_size, x_bounds, y_bounds):
    population = np.zeros((pop_size, 2))
    population[:, 0] = np.random.uniform(x_bounds[0], x_bounds[1], pop_size)
    population[:, 1] = np.random.uniform(y_bounds[0], y_bounds[1], pop_size)
    return population


def genetic_algorithm():
    pass


def main():
    x_bounds = [-5, 5]
    y_bounds = [-5, 5]
    population_size = 100

    population = initialize_population(population_size, x_bounds, y_bounds)

    for i in range(5):
        x, y = population[i]
        print(f"Individual (Point) {i + 1}: x = {x:.4f}, y = {y:.4f}, f(x,y) = {rosenbrock_function(x, y):.4f}")


if __name__ == "__main__":
    main()