# from abc import ABC, abstractmethod
from src.ec import *
from src.ec.util import *
from functools import cmp_to_key

class Breeder:
    """
    A Breeder is a singleton object responsible for the breeding process during
    an evolutionary run. Only one Breeder is created and stored in the EvolutionState object.

    Breeders typically operate by applying a Species' breeding pipelines on
    subpopulations to produce new individuals.

    Breeders may be multithreaded, and care must be taken when accessing shared resources.
    """

    P_BREEDER = "breed"
    P_ELITE = "elite"
    P_REEVALUATE_ELITES = "reevaluate-elites"

    max_tries = 5

    def __init__(self):
        self.elitenum = None
        self.toReevaluateElite = None
        self.operatorNum = None
        self.operators = None
        self.operatorRate = None

    def setup(self, state:EvolutionState, base:Parameter):
        p = Parameter(state.P_POP).push(Population.P_SIZE)
        default_p = BreedDefault.base

        size = state.parameters.getInt(p, None)

        self.elitenum = [int]*size
        self.toReevaluateElite = [bool]*size

        for x in range(size): # for each subpopulation
            # load the elite settings

            if state.parameters.exists(base.push(self.P_ELITE).push(str(x)), default_p.push(self.P_ELITE).push(str(x))):
                self.elitenum[x] = state.parameters.getInt(
                    base.push(self.P_ELITE).push(str(x)), 
                    default_p.push(self.P_ELITE).push(str(x)))
            else:
                self.elitenum[x] = 1
            
            if state.parameters.exists(base.push(self.P_REEVALUATE_ELITES).push(""+str(x)), default_p.push(self.P_REEVALUATE_ELITES).push(""+str(x))):
                self.toReevaluateElite[x] = state.parameters.getBoolean(base.push(self.P_REEVALUATE_ELITES).push(""+str(x)),
                                                                        default_p.push(self.P_REEVALUATE_ELITES).push(""+str(x)),
                                                                        True)
            else:
                self.toReevaluateElite[x] = True


    def breedPopulation(self, state:EvolutionState)->Population:
        """
        Breeds state.population, returning a new population. In general,
        state.population should not be modified.
        """
        newpop:Population = state.population.emptyclone()

        self.loadElites(state, newpop)

        for subpop_i in range(len(newpop.subpops)):
            subp:Subpopulation = newpop.subpops[subpop_i]
            old_subp:Subpopulation = state.population.subpops[subpop_i]
            from_i = 0
            numind_i = len(subp.individuals) - self.elitenum[subpop_i]

            bp:BreedingPipeline = subp.species.pipe_prototype

            bp.prepareToProduce(state,subpop_i,0)

            x = from_i

            while x < numind_i:
                for tryi in range(self.max_tries):
                    tmp_x = bp.produce(
                        1,numind_i-x,x,subpop_i,
                        subp.individuals,
                        state,0
                    )

                    # res_inds = subp.individuals[x : x+tmp_x]
                    # str_res_inds = [ind.printTrees() for ind in res_inds]

                    exist = False # no duplication
                    # if subp.numDuplicateRetries >= 1: # use subpopulation.duplicateSet to eliminate duplicate individuals
                    #     for str_ind in str_res_inds:
                    #         exist = exist or (True if subp.duplicateSet is not None and str_ind in subp.duplicateSet 
                    #                           and old_subp.duplicateSet is not None and str_ind in old_subp.duplicateSet
                    #                           else False)                            
                    
                    if not exist or tryi + tmp_x > self.max_tries: # we can move on producing more new individuals now
                        x += tmp_x
                        # if subp.numDuplicateRetries >= 1:
                        #     for str_ind in str_res_inds:
                        #         subp.duplicateSet.add(str_ind) 
                        break

            bp.finishProducing(state,subpop_i,0)

        return newpop

    def loadElites(self, state:EvolutionState, newpop:Population):
        for x in range(len (state.population.subpops) ):
            if self.elitenum[x] >= len( state.population.subpops[x].individuals ):
                state.output.error("The number of elites for subpopulation " + str(x) 
                                   + " equals or exceeds the actual size of the subpopulation")
                
        for x in range(len(state.population.subpops)):
            if self.elitenum[x] == 1: #  if the number of elites is 1, then we handle this by just finding the best one.
                bestInd = None
                for ind in state.population.subpops[x].individuals:
                    if bestInd is None or ind.fitness.betterThan(bestInd.fitness):
                        bestInd = ind
                    
                inds = newpop.subpops[x].individuals
                inds[len(inds) - 1] = bestInd.clone()
            elif self.elitenum[x] > 1: # we will need to sort
                sortedInds = sorted(state.population.subpops[x].individuals, key=cmp_to_key(compare))

                inds = newpop.subpops[x].individuals
                
                for x in range(len(inds) - self.elitenum[x], len(inds)):
                    inds[x] = sortedInds[x].clone()

        # optionally force reevaluation
        for x in range(len(state.population.subpops)):
            if self.toReevaluateElite[x] == True:
                for e in range(self.elitenum[x]):
                    length = len( newpop.subpops[x].individuals )
                    newpop.subpops[x].individuals[length - e - 1].evaluated = False

    # def BreederDefault(self)->Parameter:
    #     return Parameter(self.P_BREEDER)
                 

def compare(a:GPIndividual, b:GPIndividual):
    if a.fitness.betterThan(b.fitness):
        return 1
    elif b.fitness.betterThan(a.fitness):
        return -1
    return 0


from dataclasses import dataclass
@dataclass
class BreedDefault:
    base:Parameter = Parameter("breed")