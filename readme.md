# Parameter Optimization 
The 4 scripts titled `fitness_dimensions.py`, `ga_dimensions.py` and `fitness_granularity.py`, `ga_granularity` are used to find optimal simulation parameters for desired constraints (energy loss, and number of empty cells). 

The genetic algorithm scripts (GA) will use their respective fitness scripts to calculate the fitness function, this means both files can be repurposed for other 

The scripts were designed to run with the Par04 example [1], though can also run with any example of similar workflow. Ensure that the paths in the fitness files correspond to those of the example folder.

-[1]: https://gitlab.cern.ch/fastsim/par04

**How to use the scripts:** 
- Copy the four scripts into the **build** directory of the example (do not forget to give permission for reading, writing and executing)
- Modify the fitness files with the desired simulation settings, and your target constraints (more details below).
- Run the genetic algorithm scipts (GA): `python ./ga_dimensions.py` or `python ./ga_granularity.py`.
- The algorithm will print the best fitness value and the associated solution, and two plots will be saved as png, fitness evolution and invidual gene evolution.
- For more detailed analysis, fitness scripts can be ran individually, ensure you pass simulation parameters as arguments, e.g: `python ./fitness_dimensions.py --nbOfZCells 45 --sizeOfZCells 1.9 --nbOfRhoCells 16 --sizeOfRhoCells 2.3 --nbOfPhiCells 16`
- The fitness scripts return the normalized (not scaled!) constraint value for the respective script, as well as the plot for the data distribution, and its PDF plots after normalization.

## Dimensions (Z and Rho):
The first phase concerns finding the optimal lengths for Z and Rho, the height and radius of the cylindrical scoring mesh, the scripts in question are titled `fitness_dimensions.py`, `ga_dimensions.py`. The genetic algorithm aims to find the **smallest** values for Z and Rho that yield the closest target energy loss (the not captured by the scoring mesh).
The parameters available in the fitness file are:
- Incident energy: in ENERGY and ENERGY_UNIT.
- Number of events per run: in EVENTS
- File paths for: common settings and examplePar04 in FILE_PATH and EXAMPLE_PAR04_PATH (if change is needed).

Normalization and plotting are done in the function `calculate_energy_loss()`. The generated .root and .h5 files are deleted after each run of the script.
The parameters available in the genetic algorithm script are:
- Minimum length for Z (in mm): MIN_Z
- Maximum length for Z (in mm): MAX_Z
- Minimum length for Rho (in mm): MIN_RHO
- Maximum length for Rho (in mm): MAX_RHO
- Number of solutions per generation: NUM_SOLUTIONS
- Number of parents per generation: NUM_PARENTS_MATING
- Number of generations: NUM_GENERATIONS
- Probability of crossover: CROSSOVER_PROBABILITY
- Probability of mutation: MUTATION_PROBABILITY

Other parameters can be found in the initialisation of the PyGAD instance in `pygad.GA()`, see PyGAD documentation [2] for further customization and explanation of algorithm parameters.
Current target energy is set to 2%, this can be modified in the `fitness_func()`, along with exponential scaling and normalization of lengths Z and Rho. Penalization of the two constraints can also be modified through the final formula for fitness calculation.
**Note** that the algorithm uses custom initial populations, for this case it explores all the min, max and mean values for Z and Rho, and fills the rest of the population with randomized values from set intervals.
-[2]: https://pygad.readthedocs.io/en/latest/

## Granularity (Nb_Z, Nb_Rho, Nb_Phi):
The second phase concerns finding the optimal number of cells in each dimension of the cylindrical scoring mesh, the scripts in question are titled `fitness_granularity.py`, `ga_granularity.py`. The genetic algorithm aims to find the best combination for these three dimensions, that:
- Yields (or is close enough to) a given total granularity of the cylinder.
- Returns the lowest number of empty cells (empty cells are defined as cells containing less than 5KeV of energy deposition).
 
The parameters available in the fitness file are:
- Incident energy: in ENERGY and ENERGY_UNIT.
- Number of events per run: in EVENTS
- File paths for: common settings and examplePar04 in FILE_PATH and EXAMPLE_PAR04_PATH (if change is needed).

Normalization and plotting are done in the function `calculate_energy_loss()`, as well as the definition of definition of empty cells to <5KeV. The generated .root and .h5 files are deleted after each run of the script.
The parameters available in the genetic algorithm script are:
- Mesh height (Z): LENGTH_OF_Z
- Mesh radius (Rho): LENGTH_OF_RHO
- Targeted granularity: TOTAL_GRANULARITY
- Minimum number of Z cells: MIN_Z_CELLS
- Maximum number of Z cells: MAX_Z_CELLS
- Minimum number of Rho cells: MIN_RHO_CELLS
- Maximum number of Rho cells: MAX_RHO_CELLS
- Minimum number of Phi cells: MIN_PHI_CELLS
- Maximum number of Phi cells: MAX_PHI_CELLS
- Granularity error window (explained below): PERCENTAGE
- Number of solutions per generation: NUM_SOLUTIONS
- Number of parents per generation: NUM_PARENTS_MATING
- Number of generations: NUM_GENERATIONS

Other parameters can be found in the initialisation of the PyGAD instance in `pygad.GA()`, see PyGAD documentation [2] for further customization and explanation of algorithm parameters.

**Note** that this script (in the GA file) contains two possible possible algorithms, set through the boolean `approximate_algo`, if set to FALSE, the algorithm used is a simple brute force that explores all possible combinations of Nb_Z, Nb_Rho, Nb_Phi whose product equals **exactly** the total granularity.
If set to TRUE, the script uses a genetic algorithm, however, the product of the three variables will be approximate to the target granularity, an error window can be set (defaulted to 10%).

No scaling is used as the normalized number of empty cells is the only variable to optimize, though if you wish to add additional constraints, you can do so in the `fitness_func_approx()` function.

**Note** that the algorithm uses custom initial populations, they are generated using the `find_third_variable()` function, to ensure that for all combinations each variables (gene) falls into its respective min-max interval, and that the product is within the error window of the granularity.
Additionally, a custom mutation function is used in `custom_mutation`, to also ensure these two constrains are respected. Mutation of variables is dynamically changed based on the current generation, and the fitness of the solution in question.