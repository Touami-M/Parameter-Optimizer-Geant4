import subprocess
import numpy as np
import pygad
import matplotlib.pyplot as plt
import random

# Constants for parameter ranges and population size
MIN_Z = 1
MAX_Z = 300
MIN_RHO = 0.1
MAX_RHO = 300
NUM_SOLUTIONS = 50
NUM_GENERATIONS = 30
NUM_PARENTS_MATING = 20
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.1

# Fitness function to optimize
def fitness_func(ga_instance, solution, solution_idx):
    # Extract parameters from solution
    lengthOfZCells, lengthOfRhoCells = solution

    # Run subprocess with parameters
    try:
        result = subprocess.run(['python3', './fitness_dimensions.py',
                                 '--nbOfZCells', str(1), 
                                 '--sizeOfZCells', str(lengthOfZCells), 
                                 '--nbOfRhoCells', str(1), 
                                 '--sizeOfRhoCells', str(lengthOfRhoCells), 
                                 '--nbOfPhiCells', str(1)], 
                                capture_output=True, text=True, check=True)
        if result.returncode == 0:
            output_lines = result.stdout.split('\n')
            average_energy_loss_line = output_lines[-2]
            average_energy_loss = float(average_energy_loss_line.split(':')[-1].strip())

            # Scale energy loss relative to "2%"
            if average_energy_loss > 2.0:
                average_energy_loss_scaled = 2 * (np.exp(1.25 * average_energy_loss - 2.25)) - np.exp(+0.25)
            else:
                average_energy_loss_scaled = 2 * (np.exp(-1.25 * average_energy_loss + 1.75)) - np.exp(-0.75)

            fitness = -average_energy_loss_scaled
            # Penalize based on constraints
            fitness -= 2 * ((lengthOfZCells - MIN_Z) / (MAX_Z - MIN_Z) + (lengthOfRhoCells - MIN_RHO) / (MAX_RHO - MIN_RHO))
        else:
            print("Error running script")
            fitness = -1e6
        print("Fitness:", fitness, "Energy Loss:", average_energy_loss, "%, Length of Z Cells:", lengthOfZCells, ", Length of Rho Cells:", lengthOfRhoCells)
        return fitness
    except Exception as e:
        print("Exception occurred:", e)
        return -1e6

# Function to print when a new generation is reached
def on_generation(ga_instance):
    print("Generation {}/{} - Best Fitness: {}".format(ga_instance.generations_completed, ga_instance.num_generations, ga_instance.best_solution()[1]))

def main():
    # Define parameter ranges
    param_ranges = [{'low': MIN_Z, 'high': MAX_Z, 'step': 0.01}, {'low': MIN_RHO, 'high': MAX_RHO, 'step': 0.01}]

    # Initialize population
    initial_population = [
        [MIN_Z, MIN_RHO],
        [MIN_Z, MAX_RHO],
        [MAX_Z, MIN_RHO],
        [MAX_Z, MAX_RHO],
        [MIN_Z, (MIN_RHO + MAX_RHO) / 2],
        [MAX_Z, (MIN_RHO + MAX_RHO) / 2],
        [(MIN_Z + MAX_Z) / 2, MIN_RHO],
        [(MIN_Z + MAX_Z) / 2, MAX_RHO]
    ]

    # Extend population to desired size
    while len(initial_population) < NUM_SOLUTIONS:
        z = random.uniform(MIN_Z, MAX_Z)
        rho = random.uniform(MIN_RHO, MAX_RHO)
        initial_population.append([z, rho])

    # Create genetic algorithm instance
    ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                           num_parents_mating=NUM_PARENTS_MATING,
                           fitness_func=fitness_func,
                           sol_per_pop=NUM_SOLUTIONS,
                           num_genes=len(param_ranges),
                           gene_space=param_ranges,
                           initial_population=initial_population,
                           on_generation=on_generation,
                           parent_selection_type="sss",
                           crossover_probability=CROSSOVER_PROBABILITY,
                           mutation_probability=MUTATION_PROBABILITY,
                           save_solutions=True)

    # Run optimization
    ga_instance.run()

    # Get best solution
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()

    # Plot best fitness
    ga_instance.plot_fitness()
    plt.savefig('best_fitness_plot_ZRho.png')

    # Plot genes
    ga_instance.plot_genes()
    plt.savefig('genes_plot_ZRho.png')

    # Print best solution and fitness
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_solution_fitness)

if __name__ == "__main__":
    main()
