from GeneticAlgorithm import GeneticAlgorithm
import time
import click
import os
import pandas as pd

@click.command()
@click.option("-g", "--max-gen", type=int, required=True, help="Specify the maximum number of generations for the program to run.")
@click.option("-p", "--popu-size", type=int, required=True, help="Specify the population size (number of individuals).")
@click.option("-Mp", "--mutation-points", type=int, default=1, help="Specify the number of bits fliped per individual during mutation.   [DEFAULT = 1]")
@click.option("-Cp", "--cross-points", type=int, default=1, help="Specify the number of crossover points for the parents.   [DEFAULT = 1]")
@click.option("-m", "--markers", type=click.Path(exists=True), help="Select the file containing the marker genes in order to include them always.")
@click.argument("dataset", type=click.Path(exists=True))
def executeProgram(max_gen, popu_size, mutation_points, cross_points, markers, dataset):
    # Load the marker genes
    marker_genes = []
    if markers != None:
        with open(os.path.abspath(markers)) as markerGenesFile:
            for line in markerGenesFile:
                cleanLine = line.replace("\n", "")
                if cleanLine != "":
                    marker_genes.append(cleanLine)
    
    # Load the dataset
    source = os.path.abspath(dataset)
    dataset = pd.read_csv(source, index_col=0)


    dataset = dataset.iloc[:10, :]


    # Execute the Evolutionary algorithm
    GenAlg = GeneticAlgorithm(
        popSize         =   popu_size, 
        markerGenes     =   marker_genes, 
        dataset         =   dataset, 
        mutateBits      =   mutation_points, 
        crossPoints     =   cross_points
    )
    solution = GenAlg.run(max_gen)

if __name__ == "__main__":
    executeProgram()