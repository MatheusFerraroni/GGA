import GGA
import numpy as np

def random_genome():
    gen = np.arange(30)
    np.random.shuffle(gen)
    return gen



def custom_mutate(pos, gen):
    aux = gen[pos]
    change_with = np.random.randint(low=0,high=len(gen))
    gen[pos] = gen[change_with]
    gen[change_with] = aux
    return gen


def custom_fitness(genome):
    worst = (len(genome)*(len(genome)+1))/2-len(genome)


    collision = 0
    for i in range(0,len(genome),1):
        for j in range(i,len(genome),1):
            if genome[j]>genome[i]:
                collision += 1

    return 1-collision/worst

def custom_crossover(genA, genB):

    new = np.array([],dtype=int)
    for i in range(len(genA)):

        if np.random.random()<0.5:
            if not genA[i] in new:
                new = np.append(new, genA[i])
                continue
        else:
            if not genB[i] in new:
                new = np.append(new, genB[i])
                continue

        while True:
            pos = np.random.randint(0, high=len(genA))
            if np.random.random()<0.5:
                if not genA[pos] in new:
                    new = np.append(new, genA[pos])
                    break
            else:
                if not genB[pos] in new:
                    new = np.append(new, genB[pos])
                    break
    return new

def main():

    a = [19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]

    g = GGA.GeneticAlgorithm()

    population_size   = 100
    iteration_limit   = 100
    cut_half_pop      = True
    replicate_best    = 0.05

    # override functions
    g.set_evaluate(custom_fitness)
    g.set_random_genome(random_genome)
    g.set_mutate(custom_mutate)
    g.set_custom_crossover(custom_crossover)


    # set parameters
    g.threads(False)
    g.set_mutation_rate(0.01)
    g.set_crossover_type(4) # 4 is for custom crossover
    g.set_population_size(population_size)
    g.set_iteration_limit(iteration_limit)
    g.set_cut_half_population(cut_half_pop)
    g.set_replicate_best(replicate_best)


    solution = g.run()

    print(solution.genome)
    print(sum(solution.genome))


if __name__ == '__main__':
    main()