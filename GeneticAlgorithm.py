import numpy as np
import random
import math
import multiprocessing

class GeneticAlgorithm():

    def __init__(self, popSize, markerGenes, dataset, Pm = 0.1, Pc = 0.9):
        self._popSize = popSize
        self._mutateProb = Pm
        self._crossProb = Pc
        self._markerGenes = markerGenes 
        self._otherGenes = list(markerGenes.index)
        self._indSize = dataset.shape[0]
        self._dataset = dataset

    def _createPopulation(self):
        if self._markerGenes != []:
            return np.random.randint(0, 2, size=(self._popSize, self._indSize))
        else:
            return np.random.randint(0, 2, size=(self._popSize, self._indSize))

    def run(self, maxGenerations):
        population = self._createPopulation()

        for generation in range(1, maxGenerations+1):


            population = self._mutation(population)

        return population

    def _mutation(self, population, nBits = 1):
        newPopulation = np.empty_like(population)

        for i, individual in enumerate(population):
            probability = random.random()
            if probability < self._mutateProb:

                for _ in range(nBits):
                    gene = random.randint(len(individual))
                    individual[gene] = not individual[gene]

            newPopulation[i] = individual

        return newPopulation

    def _selection(self):
        pass

    def _crossover(self, parent1, parent2, crossPoints = 1):
        pass

    def _evaluate(self, individual):
        pass

    # This is the one you have to modify, Den.
    # As you can see, I've already added the phenotype decoding, now you just have to 
    # do your magic and perform the clustering and ge the required metric...
    def _fitnessFunction(self, individual):

        genesToCluster = []
        
        if self._markerGenes != None:
            pass


        # Temporary dummy fitness (The one with most 1s is most fit)
        return np.sum(individual)




"""
class GeneticAlgorithm(ABC):

    def __init__(self, lowerBound, upperBound, varNum, func, popu_size, Pc=0.9, Pm = None ):
        self._x_max = upperBound
        self._x_min = lowerBound
        self._func = func
        self._population = []
        self._vars = varNum 
        self._popu_size = popu_size
        self._Pc = Pc           # Probability of crossover
        self._Pm = None          # Probability of mutation

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

    def _evaluation(self, population):
        fit_list = []
        for i in population:
            current_value = self._func(i)
            fit_list.append(current_value)
            
        return fit_list
    
#####################################################################
################## Child class for Binary Encoding ##################
#####################################################################

class BinaryGA(GeneticAlgorithm):

    def __init__(self, lowerBound, upperBound, varNum, func, popu_size, Pc= 0.9, Pm=0.1):
        super().__init__(lowerBound, upperBound, varNum, func, popu_size, Pc, Pm)
        self._num_bits = self.number_bits()

    def initialize_population(self):
        # Initialize each individual in the population
        num_bits = self.number_bits()
        for _ in range(self._popu_size):
            chromosome = ''.join(''.join(str(random.randint(0, 1)) for _ in range(num_bits)) for _ in range(self._vars))
            self._population.append(chromosome)

    
    def decoder(self):
        decode_population = []
        for individual in self._population:
            values = []
            pos = 0
            # Decode each variable
            for _ in range(self._vars):
                genome = individual[pos:pos+self._num_bits]
                decimal_value = int(genome, 2)

                # Scale the decimal value to its original range
                real_value = self._x_min + decimal_value * ((self._x_max - self._x_min) / (2**self._num_bits - 1))
                values.append(round(real_value, 4))
                pos += self._num_bits
            decode_population.append(values)
        return decode_population



    #Roulette wheel
    def _selection(self, fit_list): 
        #Initialize the total fitness (f) and cumulative probability (q) to 0
        f=0
        q=0

        # Lists to store cumulative probabilities a probabilities of selection
        cumu_probability = []
        probability = []

        #Step 2 calculate the total fitness (f)
        #revert fitness values
        for i in fit_list:
            i_new = max(fit_list) - i + 1e-6
            f += i_new
        
        # Step 3 calculate the probability for each element
        #In case that f is equal to zero all the individuals will have the same probability
        for i in fit_list:
            if f == 0:
                new_prob = 1/len(fit_list)
            else:
                new_prob = (max(fit_list) - i + 1e-6)/f
            probability.append(new_prob)
        
        #Step 4 calculate the cumulative probability
        for i in probability:
            q += i  
            cumu_probability.append(q)

        # step 5 get a pseudo-random number between 0 and 1


        selected_individuals =[]
        for i in range(len(self._population)):
            r = random.random()

            # Find the first individual whose cumulative probability is greater than or equal to r
            for k in range(len(self._population)):
                if r <= cumu_probability[k]:
                    selected_individuals.append(self._population[k])
                    break  
                 
        return selected_individuals

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

      
    def _mutation(self, population):
        #Pm = 1/(len(population[0]))
        for i in range(len(population)):
            r = random.random()
            if r < self._Pm:
                chromosome = population[i]
                gene_number = random.randint(0, len(chromosome)-1)
                gene = chromosome[gene_number]

                if gene == '0':
                    gene = '1'
                else:
                    gene = '0'
                chromosome_mutated = chromosome[:gene_number] + gene + chromosome[(gene_number+1):]
                population[i] = chromosome_mutated
"""