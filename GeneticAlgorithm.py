import numpy as np
import math
import multiprocessing

class GeneticAlgorithm():

    def __init__(self, popSize, markerGenes, dataset, Pm = 0.1, Pc = 0.9):
        # Problem's properties
        self._markerGenes = markerGenes 
        self._encodedGenes = [gene for gene in list(dataset.index) if gene not in self._markerGenes]
        self._dataset = dataset

        # GA's Hyper-parameters
        self._popSize = popSize
        self._indSize = len(self._encodedGenes)
        self._mutateProb = Pm
        self._crossProb = Pc

    def run(self, maxGenerations):
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
            fitnessOffspring = self._evaluate(offspring)
            combinedFitness = population
            combinedPopulation = fitness




            # Mutate the selected individuals to create the new population
            population = self._mutation(population)


            if generation % 10 == 0:
                pass

        return population





        # Initialize population
        self.initialize_population()
        self._Pm = 1/len(self._population[0])
        best_solutions = []

        for generation in range(1, num_generations+1):
            
            # Evaluation
            fit_list = self._evaluation(self._getPopulation())

            # Selection
            selected_individuals = self._selection(fit_list)

            # Crossover
            children = self._crossover(selected_individuals)

            # Mutation
            mutated_population = self._mutation(children)

            # Update population
            self._population = list(mutated_population)

            # print best fitness in the generation
            min_index = fit_list.index(min(fit_list))

            # Store the best fitness and its corresponding decoded point
            best_fitness = fit_list[min_index]
            #print(f"Este es el mejor fitness {best_fitness}, generación {generation}")
            best_solutions.append(best_fitness)


            # Stopping Criterion (if the standard deviation of the last 5 generations is less than threshold 5)
            if generation % 10 == 0:
                if np.std(np.array(best_solutions[generation-10:])) < 0.05:
                    return best_solutions

        return best_solutions










    def _mutation(self, population, nBits = 1):
        # Create an empty new population
        newPopulation = np.empty_like(population)

        # Mutate each individual, if the probability is greater than the threshold
        for i, individual in enumerate(population):
            probability = np.random.rand()
            if probability < self._mutateProb:

                # If nBits > 1, selects nBits at random and flips their value (mutates them)
                for _ in range(nBits):
                    gene = np.random.randint(len(individual))
                    individual[gene] = not individual[gene]

            newPopulation[i] = individual

        return newPopulation

    def _selection(self, population, fitness):
        # Combine the individuals and their scores for Binary tournament selection
        evaluatedIndividuals = np.array(list(zip(population, fitness)))

        parents = np.empty_like(population)
        for i in range(len(population)):

            # Shuffle individuals
            shuffledPop = np.copy(evaluatedIndividuals)
            np.random.shuffle(shuffledPop)

            # Get two random individuals
            randChoice1 = np.random.randint(0, len(shuffledPop))
            randChoice2 = np.random.randint(0, len(shuffledPop))

            candidate1, fitnessCand1 = shuffledPop[randChoice1]
            candidate2, fitnessCand2 = shuffledPop[randChoice2]

            # Make them compete based on their fitness and select the fittest one
            parents[i] = candidate1 if fitnessCand1 > fitnessCand2 else candidate2

        return parents

    def _crossover(self, parents, crossPoints = 1):
        offspring = []

        for i in range(0, len(parents), 2):
            probability = np.random.rand()
            if probability < self._mutateProb:


    def _evaluation(self, population):
        # For each individual in the population, perform clustering and get the fitness
        fitness = np.empty_like(population)
        for i, individual in enumerate(population):
            individualFitness = self._fitnessFunction(individual)
            fitness[i] = individualFitness
            
        return fitness

    # This is the one you have to modify, Den.
    # As you can see, I've already added the phenotype decoding, now you just have to 
    # do your magic and perform the clustering and get the required metric...
    def _fitnessFunction(self, individual):

        genesToCluster = np.array(self._encodedGenes)
        dataset = self.dataset[genesToCluster]
        print(dataset)
        

        # Temporary dummy fitness (The one with most 1s is most fit)
        return np.sum(individual)




"""
    def run(self, num_generations):
        # Initialize population
        self.initialize_population()
        self._Pm = 1/len(self._population[0])
        best_solutions = []

        for generation in range(1, num_generations+1):
            
            # Evaluation
            fit_list = self._evaluation(self._getPopulation())

            # Selection
            selected_individuals = self._selection(fit_list)

            # Crossover
            children = self._crossover(selected_individuals)

            # Mutation
            mutated_population = self._mutation(children)

            # Update population
            self._population = list(mutated_population)

            # print best fitness in the generation
            min_index = fit_list.index(min(fit_list))

            # Store the best fitness and its corresponding decoded point
            best_fitness = fit_list[min_index]
            #print(f"Este es el mejor fitness {best_fitness}, generación {generation}")
            best_solutions.append(best_fitness)


            # Stopping Criterion (if the standard deviation of the last 5 generations is less than threshold 5)
            if generation % 10 == 0:
                if np.std(np.array(best_solutions[generation-10:])) < 0.05:
                    return best_solutions

        return best_solutions

    def _crossover(self, parents):

        #empty lists for selected individual to crossover and childrens
        individuals_cross = []
        index_cross = []
        children =[]

        #select individuals random to perform the crossover
        for i in range(0, len(parents), 2):
            r = random.random()
            if r < self._Pc:
                individuals_cross.append(parents[i])
                individuals_cross.append(parents[i+1])
                index_cross.append(i)
                index_cross.append(i+1)


        # Make the crossover
        if len(individuals_cross) > 1:
            for i in range(0, len(individuals_cross)-1,2):
                parent1 = parents[i]
                parent2 = parents[i+1]

                # select the position for crossover
                position_cross = random.randint(1, len(parents[0])-1)

                #perform single point crossover
                chromosome1 = parent1[:position_cross] + parent2[position_cross:]
                chromosome2 = parent2[:position_cross] + parent1[position_cross:]  
            
                children.append(chromosome1)
                children.append(chromosome2)

            cont = 0
            
            for i in range(len(children)):
                parents[index_cross[i]]=children[cont]
                cont +=1


        return parents

"""