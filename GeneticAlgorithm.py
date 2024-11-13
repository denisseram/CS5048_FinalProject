import numpy as np
import math
import multiprocessing

class GeneticAlgorithm():

    def __init__(self, popSize, markerGenes, dataset, mutateBits, crossPoints, Pm = 0.1, Pc = 0.9):
        # Problem's properties
        self._markerGenes = np.array(markerGenes)
        self._encodedGenes = np.array([gene for gene in list(dataset.index) if gene not in self._markerGenes])
        self._dataset = dataset

        # GA's Hyper-parameters
        self._popSize = popSize
        self._indSize = len(self._encodedGenes)
        self._mutateProb = Pm
        self._crossProb = Pc
        self._mutateBits = mutateBits
        self._crossPoints = crossPoints

    def run(self, maxGenerations):
        # Lists to store the historical best values per generation
        bestSolutions = []
        bestIndividuals = []

        # Create the initial population at random
        population = np.random.randint(0, 2, size=(self._popSize, self._indSize))

        for generation in range(1, maxGenerations+1):

            # Evaluate the fitness of the current population
            fitness = self._evaluation(population)

            # Select the parent pairs that will be used for crossover
            parents = self._selection(population, fitness)

            # Generate offspring through crossover
            offspring = self._crossover(parents)

            # Select only the best individuals for the next generation (μ + λ)
            if len(offspring) > 0:
                fitnessOffspring = self._evaluation(offspring)
                combinedFitness = np.concatenate([fitness, fitnessOffspring])
                combinedPopulation = np.vstack((population, offspring))
                indexes = np.argsort(combinedFitness)[::-1]
                newPopulation = combinedPopulation[indexes][:self._popSize]
            else:
                newPopulation = population

            # Update historical values
            bestIndex = np.argmax(fitness)
            bestSolutions.append(fitness[bestIndex])
            bestIndividuals.append(self._decode(population[bestIndex]))
            
            # Mutate the selected individuals to create the new population
            population = self._mutation(newPopulation)

            # Stopping Criterion (if the standard deviation of the last 5 generations is less than threshold 5)
            if generation % 10 == 0:
                if np.std(np.array(bestSolutions[generation-10:])) < 0.05:
                    return bestIndividuals, bestSolutions

        return bestIndividuals, bestSolutions

    def _mutation(self, population):
        # Create an empty new population
        newPopulation = np.empty_like(population)

        # Mutate each individual, if the probability is greater than the threshold
        for i, individual in enumerate(population):
            probability = np.random.rand()
            if probability < self._mutateProb:

                # If n > 1, selects n at random and flips their value (mutates them)
                for _ in range(self._mutateBits):
                    gene = np.random.randint(self._indSize)
                    individual[gene] = not individual[gene]

            newPopulation[i] = individual

        return newPopulation

    def _selection(self, population, fitness):
        # Combine the individuals and their scores for Binary tournament selection
        evaluatedIndividuals = list(zip(population, fitness))

        parents = np.empty_like(population)
        for i in range(len(population)):

            # Shuffle individuals
            shuffledPop = evaluatedIndividuals.copy()
            np.random.shuffle(shuffledPop)

            # Get two random individuals
            randChoice1 = np.random.randint(0, len(shuffledPop))
            randChoice2 = np.random.randint(0, len(shuffledPop))

            candidate1, fitnessCand1 = shuffledPop[randChoice1]
            candidate2, fitnessCand2 = shuffledPop[randChoice2]

            # Make them compete based on their fitness and select the fittest one
            parents[i] = candidate1 if fitnessCand1 > fitnessCand2 else candidate2

        return parents

    def _crossover(self, parents):
        offspring = []

        # Cross individuals pairwise, if the probability is greater than the threshold
        for i in range(0, len(parents), 2):
            probability = np.random.rand()
            if probability < self._crossProb:

                # Creates an array with n distinct points to perform the crossover
                crossIndexes = np.random.choice(np.arange(2, self._indSize-2, 3), size=self._crossPoints, replace=False)
                crossIndexes = np.insert(crossIndexes, 0, 0)
                crossIndexes = np.append(crossIndexes, self._indSize)
                crossIndexes = np.sort(crossIndexes)

                # Performs the crossover by alternating the slices exchanged
                child1 = parents[i].copy()
                child2 = parents[i + 1].copy()

                for j in range(len(crossIndexes) - 1):
                    # Only exchange odd segments (alternate)
                    if j % 2 != 0:
                        start = crossIndexes[j]
                        end = crossIndexes[j + 1]

                        child1[start:end] = parents[i + 1][start:end]
                        child2[start:end] = parents[i][start:end]

                offspring.extend([child1, child2])

        return np.array(offspring)

    def _evaluation(self, population):
        # For each individual in the population, perform clustering and get the fitness
        fitness = np.empty(len(population))
        for i, individual in enumerate(population):
            fitness[i] = self._fitnessFunction(individual)
            
        return fitness

    def _decode(self, individual):
        phenotype = self._encodedGenes[individual == 1]
        phenotype = np.concatenate([phenotype, self._markerGenes])

        return phenotype

    # This is the one you have to modify, Den.
    # As you can see, I've already added the phenotype decoding, now you just have to 
    # do your magic and perform the clustering and get the required metric...
    def _fitnessFunction(self, individual):

        # Decoding the genes from bits to the actual names
        genesToCluster = self._decode(individual)

        # Filter the dataset to keep only the selected genes
        dataset = self._dataset.loc[genesToCluster]
        
        # #####################################################################
        #
        #   CLUSTERING AND METRIC GOES HERE
        #
        # REMOVE:  Temporary dummy fitness (The count of 1's in the chromosome)
        return np.sum(individual)
