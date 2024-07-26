import math
import subprocess
import numpy as np
import pygad
from itertools import product
import random
import matplotlib.pyplot as plt

# Constants for parameters and population size
LENGTH_OF_Z = 110.84
LENGTH_OF_RHO = 60.96
TOTAL_GRANULARITY = 700

MIN_Z_CELLS = 8
MIN_RHO_CELLS = 8
MIN_PHI_CELLS = 8
MAX_Z_CELLS = round(np.sqrt(TOTAL_GRANULARITY))
MAX_RHO_CELLS = round(np.sqrt(TOTAL_GRANULARITY))
MAX_PHI_CELLS = round(np.sqrt(TOTAL_GRANULARITY))

#Error window
PERCENTAGE = 0.15  

NUM_SOLUTIONS = 20
NUM_GENERATIONS = 30
NUM_PARENTS_MATING = NUM_SOLUTIONS - 1

# Fitness function to approximate
def fitness_func_approx(ga_instance, solution, solution_idx):
    nbOfZCells, nbOfRhoCells, nbOfPhiCells = solution

    if (MIN_Z_CELLS <= nbOfZCells <= MAX_Z_CELLS) and \
       (MIN_RHO_CELLS <= nbOfRhoCells <= MAX_RHO_CELLS) and \
       (MIN_PHI_CELLS <= nbOfPhiCells <= MAX_PHI_CELLS) and \
       (TOTAL_GRANULARITY * (1.0 - PERCENTAGE) <= nbOfZCells * nbOfRhoCells * nbOfPhiCells <= TOTAL_GRANULARITY * (1.0 + PERCENTAGE)):
        try:
            result = subprocess.run(['python3', './fitness_granularity.py',
                                     '--nbOfZCells', str(int(nbOfZCells)),
                                     '--sizeOfZCells', str((LENGTH_OF_Z / nbOfZCells)),
                                     '--nbOfRhoCells', str(int(nbOfRhoCells)),
                                     '--sizeOfRhoCells', str((LENGTH_OF_RHO / nbOfRhoCells)),
                                     '--nbOfPhiCells', str(int(nbOfPhiCells))],
                                    capture_output=True, text=True, check=True)

            if result.returncode == 0:
                output_lines = result.stdout.split('\n')
                average_empty_cells_line = output_lines[-2]
                average_empty_cells = float(average_empty_cells_line.split(':')[-1].strip())

                fitness = -average_empty_cells
                print("Fitness:", fitness, "Combination:", int(nbOfZCells), ", ", int(nbOfRhoCells), ", ", int(nbOfPhiCells))
            else:
                print("Error running script")
                fitness = -1e6
        except Exception as e:
            print("Exception occurred:", e)
            fitness = -1e6
    else:
        fitness = -1e6
        print("Combination outisde Interval!")

    return fitness

# Function to find three factors that multiply to a given number
def find_three_factors_with_fitness(original_int):
    min_fitness = -1e6
    best_combination = None
    for combo in product(range(1, original_int), repeat=3):  # Use repeat=3 to get combinations of length 3
        prod = combo[0] * combo[1] * combo[2]
        if prod == original_int:
            nbOfZCells, nbOfRhoCells, nbOfPhiCells = combo
            fitness = fitness_func_exact(nbOfZCells, nbOfRhoCells, nbOfPhiCells)
            print("Fitness(out):", fitness, "Combination:", nbOfZCells, ", ", nbOfRhoCells, ", ", nbOfPhiCells)
            if fitness > min_fitness:
                print("in")
                min_fitness = fitness
                best_combination = combo
    return best_combination

# Fitness function for exact values
def fitness_func_exact(nbOfZCells, nbOfRhoCells, nbOfPhiCells):
    if (MIN_Z_CELLS <= nbOfZCells <= MAX_Z_CELLS) and \
       (MIN_RHO_CELLS <= nbOfRhoCells <= MAX_RHO_CELLS) and \
       (MIN_PHI_CELLS <= nbOfPhiCells <= MAX_PHI_CELLS) and \
       (TOTAL_GRANULARITY * (1.0 - PERCENTAGE) <= nbOfZCells * nbOfRhoCells * nbOfPhiCells <= TOTAL_GRANULARITY * (1.0 + PERCENTAGE)):
        try:
            result = subprocess.run(['python3', './fitness_granularity.py',
                                    '--nbOfZCells', str(int(nbOfZCells)),
                                    '--sizeOfZCells', str((LENGTH_OF_Z / nbOfZCells)),
                                    '--nbOfRhoCells', str(int(nbOfRhoCells)),
                                    '--sizeOfRhoCells', str((LENGTH_OF_RHO / nbOfRhoCells)),
                                    '--nbOfPhiCells', str(int(nbOfPhiCells))],
                                    capture_output=True, text=True, check=True)

            if result.returncode == 0:
                output_lines = result.stdout.split('\n')
                average_empty_cells_line = output_lines[-2]
                average_empty_cells = float(average_empty_cells_line.split(':')[-1].strip())

                fitness = -1 * average_empty_cells
                print("Fitness:", fitness, "Combination:", int(nbOfZCells), ", ", int(nbOfRhoCells), ", ", int(nbOfPhiCells))

            else:
                print("Error running script")
                fitness = -1e6
        except Exception as e:
            print("Exception occurred:", e)
            fitness = -1e6
    else:
        fitness = -1e6
        print("Combination outisde Interval!")
    return fitness

# Callback function for generation change
def on_generation(ga_instance):
    print("Generation {}/{} - Best Fitness: {}".format(ga_instance.generations_completed, ga_instance.num_generations, ga_instance.best_solution()[1]))

# Function to find the third variable from two variables and a product
def find_third_variable(target, z_min, z_max, rho_min, rho_max, phi_min, phi_max):
    counter=0
    while True:
        counter=counter+1
        if counter>1000:
            return 10, 10, 10
        variables = random.sample(["Z", "Rho", "Phi"], 2)
        if "Z" in variables:
            z = random.randint(z_min, z_max)
        else:
            z = None
        if "Rho" in variables:
            rho = random.randint(rho_min, rho_max)
        else:
            rho = None
        if "Phi" in variables:
            phi = random.randint(phi_min, phi_max)
        else:
            phi = None

        if z is not None and rho is not None:
            phi = target // (z * rho)
        elif z is not None and phi is not None:
            rho = target // (z * phi)
        elif rho is not None and phi is not None:
            z = target // (rho * phi)

        if (z_min <= z <= z_max) and (rho_min <= rho <= rho_max) and (phi_min <= phi <= phi_max):
            return z, rho, phi

# Custom mutation function
def custom_mutation(offsprings, ga_instance):
    offsprings_mutated = []
    fitnesses = np.sort(np.array(ga_instance.last_generation_fitness))[::-1]
    fitnesses = fitnesses[:-1]
    fitnesses = fitnesses - np.min(fitnesses)
    parents = ga_instance.last_generation_parents
    generation = ga_instance.generations_completed
    idx = 0

    while idx < len(offsprings):
        solution = offsprings[idx]
        mutated_solution = solution.copy()
        z, rho, phi = mutated_solution
        counter = 0
        factor = fitnesses[idx] / ((generation + 1) * np.max(fitnesses)) + 1
        factor = 1 if math.isnan(factor) else factor

        while True:
            variable_index = random.randrange(3)
            variable_to_mutate = mutated_solution[variable_index]
            mutated_value = variable_to_mutate + random.randint(round(-10 * factor), round(10 * factor))

            if variable_index == 0:
                product = mutated_value * rho * phi
            elif variable_index == 1:
                product = z * mutated_value * phi
            else:
                product = z * rho * mutated_value

            if 1.0 - PERCENTAGE <= product / TOTAL_GRANULARITY <= 1.0 + PERCENTAGE:
                if MIN_Z_CELLS <= mutated_value <= MAX_Z_CELLS and variable_index == 0:
                    mutated_solution[0] = mutated_value
                    offsprings_mutated.append(mutated_solution.copy())
                    break
                elif MIN_RHO_CELLS <= mutated_value <= MAX_RHO_CELLS and variable_index == 1:
                    mutated_solution[1] = mutated_value
                    offsprings_mutated.append(mutated_solution.copy())
                    break
                elif MIN_PHI_CELLS <= mutated_value <= MAX_PHI_CELLS and variable_index == 2:
                    mutated_solution[2] = mutated_value
                    offsprings_mutated.append(mutated_solution.copy())
                    break
            counter += 1
            if counter > 100:
                offsprings_mutated.append(mutated_solution.copy())
                break

        idx += 1

    return np.array(offsprings_mutated)

def main():
    approximate_algo = False  # True: Original granularity approximated | False: Original granularity conserved

    if approximate_algo:  # Original granularity Approximated
        initial_population = []

        while len(initial_population) < NUM_SOLUTIONS:
            z, rho, phi = find_third_variable(TOTAL_GRANULARITY, MIN_Z_CELLS, MAX_Z_CELLS, MIN_RHO_CELLS, MAX_RHO_CELLS, MIN_PHI_CELLS, MAX_PHI_CELLS)
            initial_population.append([z, rho, phi])

        param_ranges = [{'low': MIN_Z_CELLS, 'high': MAX_Z_CELLS, 'step': 1},
                        {'low': MIN_RHO_CELLS, 'high': MAX_RHO_CELLS, 'step': 1},
                        {'low': MIN_PHI_CELLS, 'high': MAX_PHI_CELLS, 'step': 1}]

        ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                               num_parents_mating=NUM_PARENTS_MATING,
                               fitness_func=fitness_func_approx,
                               num_genes=len(param_ranges),
                               initial_population=initial_population,
                               on_generation=on_generation,
                               parent_selection_type="sss",
                               save_solutions=True,
                               crossover_type=None,
                               mutation_type=custom_mutation)

        ga_instance.run()

        best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution()

        ga_instance.plot_fitness()
        plt.savefig('best_fitness_plot_Granularity.png')

        ga_instance.plot_genes()
        plt.savefig('genes_plot_Granularity.png')

        print("Best Solution:", best_solution)
        print("Best Fitness:", best_solution_fitness)
    else:
        print("Best Combination: ",find_three_factors_with_fitness(TOTAL_GRANULARITY))  # Original granularity conserved

if __name__ == "__main__":
    main()

#Best Solution: [ 5. 17.  5.]Best Fitness: -348.9421052631579 at 500 Gran