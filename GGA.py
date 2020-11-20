import numpy as np
import threading
import time


"""

Usefull link:

https://www.geeksforgeeks.org/crossover-in-genetic-algorithm/


"""





"""

Element class, contain a possible solution



"""
class Element:

    def __init__(self, idd, geracao, genome):
        self.idd = idd
        self.geracao = geracao
        self.genome = genome
        self.score = None


    def __repr__(self):
        return "(id={},geracao={},score={:.20f})".format(self.idd, self.geracao,self.score)



class GeneticAlgorithm:


    """
    
    Create the GA main class and prepare the parameters that will be used
    Many of this parameters can be adjusted using the setters

    """
    def __init__(self):
        self.population = []
        self.historic = []
        self.mutation_rate = 0
        self.population_size = 50
        self.iteration_limit = 5
        self.elements_created = 0
        self.crossover_type = 0
        self.best_element_total = None
        self.random_elements_generation = 0.2
        self.max_possible_score = float('inf')
        self.iteration_counter = 0
        self.stop_criteria_type = 0
        self.probs_type = 0
        self.use_threads = False
        self.crossover_rate = 0.5
        self.cut_half_population = False

        self.replicate_best = 0


    def get_config(self):
        o = {}

        o["mutation_rate"] = self.mutation_rate
        o["population_size"] = self.population_size
        o["iteration_limit"] = self.iteration_limit
        o["elements_created"] = self.elements_created
        o["random_elements_generation"] = self.random_elements_generation
        o["crossover_type"] = self.crossover_type
        o["max_possible_score"] = self.max_possible_score
        o["iteration_counter"] = self.iteration_counter
        o["stop_criteria_type"] = self.stop_criteria_type
        o["probs_type"] = self.probs_type
        o["crossover_rate"] = self.crossover_rate
        o["cut_half_population"] = self.cut_half_population
        o["replicate_best"] = self.replicate_best

        return o


    """
    
    Main loop

    """
    def run(self):

        print("[GA] Evolution Started")

        self.time_start = time.time()

        if len(self.population)==0:
            self.create_initial_population()

        while self.check_stop():
            self.calculate_score()
            self.population.sort(key=lambda x: x.score, reverse=True)

            if self.best_element_total==None:
                self.best_element_total = self.population[0]

            if self.population[0].score > self.best_element_total.score:
                self.best_element_total = self.population[0]

            self.do_log()
            print("[GA]  Generation: {}, Best geral: {:.2f}, Best Actual: {:.2f}, Mean: {:.2f}".format(self.iteration_counter,self.best_element_total.score,self.historic[-1]["best"],self.historic[-1]["avg"]))


            if self.cut_half_population:
                self.population = self.population[0:len(self.population)//2]

            self.iteration_counter +=1

            self.new_population()





        self.time_end = time.time()

        print("[GA] Evolucao finalizada. Tempo total: {:.2f} segundos".format(self.time_end-self.time_start))
        return self.best_element_total


    """

    Create new population

    """
    def new_population(self):

        probs = self.get_probs()

        newPop = []

        best_replicator = int(self.population_size*self.replicate_best)

        for i in range(best_replicator):
            newPop.append(self.population[i])

        random_generator = int(self.population_size*self.random_elements_generation)

        for i in range(random_generator):
            newPop.append(Element(self.elements_created, 0, self.random_genome()))
            self.elements_created += 1


        while len(newPop)<self.population_size:
            parents = np.random.choice(self.population,size=2,p=probs)

            if parents[0].score<parents[1].score: # garantee that the parent is always better
                parents = parents[::-1] # reverse array

            new_element = Element(self.elements_created, self.iteration_counter, self.crossover(parents[0].genome, parents[1].genome))

            new_element.genome = self.active_mutate(new_element.genome)
            newPop.append(new_element)
            self.elements_created += 1


        self.population = newPop

    """

    Return array with chance to select each element
    
    """
    def get_probs(self):
        if self.probs_type == 0:
            return self.probs_roulette()
        elif self.probs_type == 1:
            return self.probs_equal()


    """

    Return the same chance for every element

    """
    def probs_equal(self):
        return [1/len(self.population)]*len(self.population)


    """
    
    Elements with higher scores have higher chances to be selected

    """
    def probs_roulette(self):
        probs = [0]*len(self.population)
        for i in range(len(probs)):
            probs[i] = self.population[i].score
        div = sum(probs)

        if div>0:
            for i in range(len(probs)):
                probs[i] /= div
        else: # if there is no solution, return equal chance
            probs = self.probs_equal()
        return probs


    def do_log(self):

            score_geracao_medio = 0
            score_geracao_max = float('-inf')
            score_geracao_min = float('inf')
            for i in range(len(self.population)):
                score_geracao_medio += self.population[i].score
                score_geracao_min = min(score_geracao_min, self.population[i].score)
                score_geracao_max = max(score_geracao_max, self.population[i].score)
            score_geracao_medio /= len(self.population)

            todos_genomes = []
            for i in range(len(self.population)):
                todos_genomes.append(self.population[i].genome.tolist())


            self.historic.append({"geracao":self.iteration_counter,
                "max":score_geracao_max,
                "min":score_geracao_min,
                "avg":score_geracao_medio,
                "best":self.best_element_total.score,
                "best_genome":self.best_element_total.genome.tolist(),
                "todos_genomes":todos_genomes})
               



    """

    Check stop criteria

    """
    def check_stop(self):
        ret = None
        if self.stop_criteria_type==0:
            ret = self.stop_criteria_double()
        elif self.stop_criteria_type==1:
            ret = self.stop_criteria_iteration()
        elif self.stop_criteria_type==2:
            ret = self.stop_criteria_score()


        return ret



    def stop_criteria_double(self):
        s = self.population[0].score
        if s==None:
            s = 0
        return self.iteration_counter<self.iteration_limit or s>=self.max_possible_score



    def stop_criteria_iteration(self):
        return self.iteration_counter<self.iteration_limit

    def stop_criteria_score(self):
        s = self.population[0].score
        if s==None:
            s = 0
        return s>=self.max_possible_score

    def set_replicate_best(self, e):
        if e<0 or e>1:
            raise Exception("Value must be between 0 and 1.")
        self.replicate_best = e

    def set_random_elements_generation(self, e):
        self.random_elements_generation = e

    def set_probs_type(self, e):
        self.probs_type = e

    def set_cut_half_population(self, e):
        self.cut_half_population = e

    def set_max_score(self, e):
        self.max_possible_score = e

    def set_iteration_limit(self, e):
        self.iteration_limit = e

    def set_population_size(self, e):
        self.population_size = e

    def set_mutation_rate(self, e):
        self.mutation_rate = e

    def set_evaluate(self, e):
        self.evaluate = e

    def set_custom_crossover(self, e):
        self.custom_crossover = e

    def set_random_genome(self, e):
        self.random_genome = e

    def set_mutate(self, e):
        self.mutate = e

    def set_stop_criteria_type(self, e):
        self.stop_criteria_type = e

    def threads(self, e):
        self.use_threads = e

    # Create random initial population
    def create_initial_population(self):
        for _ in range(self.population_size):
            self.population.append(Element(self.elements_created, 0, self.random_genome()))
            self.elements_created += 1

    def set_crossover_type(self, e):
        self.crossover_type = e

    def set_crossover_rate(self, e):
        self.crossover_rate = e

    def crossover(self, genA, genB):
        if self.crossover_type==0:
            return self.crossover_uniform(genA, genB)
        elif self.crossover_type==1:
            return self.crossover_single_point(genA, genB)
        elif self.crossover_type==2:
            return self.crossover_two_point(genA, genB)
        elif self.crossover_type==3:
            return self.crossover_rate_selection(genA, genB)
        elif self.crossover_type==4:
            return self.custom_crossover(genA, genB)

    def crossover_rate_selection(self, genA, genB):
        new = np.array([],dtype=int)
        for i in range(len(genA)):
            if np.random.random()<self.crossover_rate:
                new = np.append(new, genA[i])
            else:
                new = np.append(new, genB[i])
        return new

    def crossover_uniform(self, genA, genB):
        new = np.array([],dtype=int)
        for i in range(len(genA)):
            if np.random.random()<0.5:
                new = np.append(new, genA[i])
            else:
                new = np.append(new, genB[i])
        return new

    def crossover_single_point(self, genA, genB):
        p = np.random.randint(low=1,high=len(genA)-1) # starts with low=1 to not copy entire element
        return np.append(genA[0:p],genB[p:])

    def crossover_two_point(self, genA, genB):
        c1 = c2 = np.random.randint(low=0,high=len(genA))
        while c2==c1:
            c2 = np.random.randint(low=0,high=len(genA))

        if c1>c2:
            c1, c2 = c2,c1

        new = np.append(np.append(genA[0:c1],genB[c1:c2]),genA[c2:])

        return new


    def calculate_score(self):
        # This function may be incomplet yet.. need a few more tests
        if self.use_threads:

            threads_running = []
            for e in self.population:
                x = threading.Thread(target=self.thread_evaluate, args=(e,))
                x.start()
                threads_running.append(x)

            for i in range(len(threads_running)):
                threads_running[i].join()

        else:
            for e in self.population:
                e.score = self.evaluate(e.genome)


    def thread_evaluate(self, e):
        e.score = self.evaluate(e.genome)


    def active_mutate(self,gen):
        if self.mutation_rate<=0:
            return gen
        for i in range(len(gen)):
            if np.random.random()<self.mutation_rate:
                gen = self.mutate(i, gen)

        return gen



    # May be used if necessary
    def custom_crossover(self, genA, genB):
        raise Exception("Should be override to be used")

    # Must be override
    def random_genome(self):
        raise Exception("Should be override")

    # Must be override
    def evaluate(self):
        raise Exception("Should be override")

    # Must be override
    def mutate(self):
        raise Exception("Should be override")