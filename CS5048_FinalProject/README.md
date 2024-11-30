# CS5048_FinalProject

# Genetic Algorithm for Optimization

This repository contains a Python implementation of a genetic algorithm for optimization gene selection for clustering in scRNA-seq data. The algorithm evolves solutions through generations and outputs the results to a CSV file, along with an optional convergence plot.


## Requirements
To run rhis code you will need to install:
- Python 3.x
- R 4.x
- Required Python packages:
  - `pandas`
  - `matplotlib`
  - `click`
  - `sklearn`
- Required R packages:
  - `Seurat`
  - `SingleCellExperiment`
  - `dplyr`
  - `patchwork`
  - `graphics`
  - `ggplot2`


## Usage
 To run the program, you can use the following command:

```bash
python CS5048_FinalProject -g MAX_GEN -p POPU_SIZE -Pc CROSSOVER_RATE -Pm MUTATION_RATE -Mp MUTATION_POINTS -Cp CROSS_POINTS -m MARKERS_FILE_PATH -l LABELS_FILE_PATH -n NUM_RUNS DATASET_FILE_PATH
```
You can also get help of what each argument means by running the following command:

```bash
python CS5048_FinalProject --help
```

# Output
For each run, the program generates:
- A csv file with the results of the best individual for generation. It includes the individual (the genes), the solution score (NMI), and the length of the individual (number of genes).
- Optionally a Convergence plot for each run.

If you want to test the algorithm with a different dataset, you will have to put the dataset in a rds file with the name "muraro.rds" into the folder SOURCE.

If you don't want to run the clustering algorithm in R, you can use the k-means algorithm, which is commented in the script GeneticAlgorithm.py.
