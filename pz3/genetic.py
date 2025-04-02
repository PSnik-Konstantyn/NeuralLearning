import numpy as np
import matplotlib.pyplot as plt
import random

from math_functions import objective_function


def initialize_population(size, bounds):
    return [random.uniform(bounds[0], bounds[1]) for _ in range(size)]


def evaluate_population(population, objective_fn):
    return [objective_fn(x) for x in population]


def select_parents(population, fitness, num_parents):
    probabilities = np.exp(fitness - np.max(fitness))
    probabilities /= probabilities.sum()
    selected = np.random.choice(population, size=num_parents, p=probabilities, replace=False)
    return selected.tolist()


def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        p1, p2 = random.sample(parents, 2)
        child = (p1 + p2) / 2
        offspring.append(child)
    return offspring


def mutate(offspring, mutation_rate, bounds):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i] += random.uniform(-0.1, 0.1)
            offspring[i] = np.clip(offspring[i], bounds[0], bounds[1])
    return offspring


def genetic_algorithm(bounds, objective_fn, maximize=True, pop_size=120, generations=100, mutation_rate=0.1,
                      elitism=0.1, patience=3):
    population = initialize_population(pop_size, bounds)
    best_solution = None
    best_value = -np.inf if maximize else np.inf
    elite_size = int(pop_size * elitism)
    no_improve_count = 0

    for generation in range(generations):
        fitness = np.array(evaluate_population(population, objective_fn))

        if not maximize:
            sorted_indices = np.argsort(fitness)
        else:
            sorted_indices = np.argsort(fitness)[::-1]

        elites = [population[i] for i in sorted_indices[:elite_size]]

        if (maximize and fitness[sorted_indices[0]] > best_value) or (
                not maximize and fitness[sorted_indices[0]] < best_value):
            best_value = fitness[sorted_indices[0]]
            best_solution = population[sorted_indices[0]]
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f'Алгоритм зупинився на поколінні {generation + 1} через відсутність покращень.')
            break

        parents = select_parents(population, fitness, num_parents=pop_size // 2)
        offspring = crossover(parents, offspring_size=pop_size - len(elites))
        offspring = mutate(offspring, mutation_rate, bounds)
        population = elites + offspring

        print(f'Покоління {generation + 1}: найкраще значення = {best_value:.6f}')

    return best_solution, best_value


bounds = [1, 4]
max_x, max_y = genetic_algorithm(bounds, objective_function, maximize=True)
print("\n__________________\n")
min_x, min_y = genetic_algorithm(bounds, objective_function, maximize=False)

x_vals = np.linspace(bounds[0], bounds[1], 500)
y_vals = objective_function(x_vals)
plt.plot(x_vals, y_vals, label='Y(x)')
plt.scatter([max_x], [max_y], color='red', label='Max')
plt.scatter([min_x], [min_y], color='blue', label='Min')
plt.xlabel('x')
plt.ylabel('Y(x)')
plt.legend()
plt.title('Оптимізація функції Y(x) генетичним алгоритмом')
plt.show()

print(f'Максимум: Y({max_x:.4f}) = {max_y:.4f}')
print(f'Мінімум: Y({min_x:.4f}) = {min_y:.4f}')
