from GeneticAlgorithm import GeneticAlgorithm
import time
import click
import os
import pandas as pd

@click.command()
@click.option("-g", "--max-gen", type=int, required=True, help="Specify the maximum number of generations for the program to run.")
@click.option("-p", "--popu-size", type=int, required=True, help="Specify the population size (number of individuals).")
@click.option("-m", "--markers", type=click.Path(exists=True), help="Select the file containing the marker genes in order to include them always.")
@click.argument("dataset", type=click.Path(exists=True))
def executeProgram(max_gen, popu_size, markers, dataset):
    # Load the marker genes
    marker_genes = []
    if markers != None:
        with open(os.path.abspath(markers)) as markerGenesFile:
            for line in markerGenesFile:
                cleanLine = line.replace("\n", "")
                if cleanLine != "":
                    marker_genes.append(cleanLine)
    
    # Execute the Evolutionary algorithm
    source = os.path.abspath(dataset)
    dataset = pd.read_csv(source, index_col=0)

    GenAlg = GeneticAlgorithm(popu_size, marker_genes, dataset)
    solution = GenAlg.run(max_gen)

if __name__ == "__main__":
    executeProgram()