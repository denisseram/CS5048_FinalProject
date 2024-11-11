from GeneticAlgorithm import GeneticAlgorithm
import time
import click
import os
import pandas as pd

projectPath = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("-g", "--max-gen", type=int, required=True, help="Specify the maximum number of generations for the program to run.")
@click.option("-p", "--popu-size", type=int, required=True, help="Specify the population size (number of individuals).")
@click.option("-m", "--markers", is_flag=True, default=False, help="Do not include marker genes in the individuals (keep them always active).")
@click.argument("dataset", type=click.Path(exists=True))
def executeProgram(max_gen, popu_size, markers, dataset):
    # Load the marker genes
    markerGenes = []
    with open(os.path.join(projectPath, "SOURCE", "all_markers.txt")) as markerGenesFile:
        for line in markerGenesFile:
            cleanLine = line.replace("\n", "")
            if cleanLine != "":
                markerGenes.append(cleanLine)
    marker_genes = markerGenes if markers else []
    
    # Execute the Evolutionary algorithm
    source = os.path.abspath(dataset)
    dataset = pd.read_csv(source, index_col=0)

    GenAlg = GeneticAlgorithm(popu_size, marker_genes, dataset)
    solution = GenAlg.run(max_gen)

if __name__ == "__main__":
    executeProgram()