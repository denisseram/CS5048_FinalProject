from GeneticAlgorithm import GeneticAlgorithm
import click
import os
import pandas as pd
import matplotlib.pyplot as plt


@click.command()
@click.option("-g", "--max-gen", type=int, required=True, help="Specify the maximum number of generations for the program to run.")
@click.option("-p", "--popu-size", type=int, required=True, help="Specify the population size (number of individuals).")
@click.option("-Pc", "--crossover-rate", type=float, default=0.8, help="Specify the probability of a pair of individuals to create offspring.   [DEFAULT = 0.8]")
@click.option("-Pm", "--mutation-rate", type=float, default=0.1, help="Specify the probability of an individual to suffer mutation.   [DEFAULT = 0.1]")
@click.option("-Mp", "--mutation-points", type=int, default=1, help="Specify the number of bits fliped per individual during mutation.   [DEFAULT = 1]")
@click.option("-Cp", "--cross-points", type=int, default=1, help="Specify the number of crossover points for the parents.   [DEFAULT = 1]")
@click.option("-m", "--markers", type=click.Path(exists=True), help="Select the file containing the marker genes in order to include them always.")
@click.option("-l", "--labels", type=click.Path(exists=True), help="Select the file containing the true labels.")
@click.option("-n", "--num-runs", type=int, default=1, help="Specify how many times to run the algorithm. [DEFAULT = 1]")  
@click.argument("dataset", type=click.Path(exists=True))
def executeProgram(max_gen, popu_size, crossover_rate, mutation_rate, mutation_points, cross_points, markers, labels, num_runs, dataset):
    # Load the dataset
    source = os.path.abspath(dataset)
    dataset = pd.read_csv(source, index_col=0)
    labels_source = os.path.abspath(labels)
    labels_df = pd.read_csv(labels_source)

    # Load the marker genes
    marker_genes = []
    if markers != None:
        with open(os.path.abspath(markers)) as markerGenesFile:
            for line in markerGenesFile:
                cleanLine = line.replace("\n", "")
                if cleanLine != "":
                    # Only keep marker genes that appear in the dataset
                    if cleanLine in dataset.index:
                        marker_genes.append(cleanLine)

    for run in range(1, num_runs + 1):
        print(f"Running execution {run}/{num_runs}...")

        # Execute the Evolutionary algorithm
        GenAlg = GeneticAlgorithm(
            popSize=popu_size,
            markerGenes=marker_genes,
            dataset=dataset,
            mutateBits=mutation_points,
            crossPoints=cross_points,
            Pm=mutation_rate,
            Pc=crossover_rate,
            labels_df=labels_df
        )
        individuals, solutions = GenAlg.run(max_gen)

        # Collect results
        results = []
        for individual, solution in zip(individuals, solutions):
            results.append({
                "Individual": individual,  # Assuming individual is iterable
                "Solution Score": solution,
                "Length": len(individual)
            })

        results_df = pd.DataFrame(results)
        output_path = os.path.join(os.getcwd(), f"genetic_algorithm_results_run_{run}.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        # Optional: Print results to console
        print("RESULTS")
        print(results_df)

        # Convergence plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_gen + 1), results_df["Solution Score"], marker="o", linestyle="-", color="b")
        plt.title(f"Convergence Plot: Best NMI per Generation (Run {run})")
        plt.xlabel("Generation")
        plt.ylabel("Best NMI")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"convergence_plot_run_{run}.png")
        print(f"Convergence plot saved as 'convergence_plot_run_{run}.png'")
        plt.show()
    
if __name__ == "__main__":
    executeProgram()