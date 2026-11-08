from typing import List
from src.lgp.individual import *
from src.ec import *
from src.ec.util import *
from src.lgp.individual.reproduce.tournament_selection import TournamentSelection

class SizeAwareTournamentSelection(TournamentSelection):
    def betterThan(self, first:LGPIndividual, second:LGPIndividual, subpopulation, state, thread):
        if max(first.getMaxNumTrees()*0.4, first.getTreesLength()) < max(second.getMaxNumTrees()*0.4, second.getTreesLength()) \
        and state.random[thread].random() < 0.5:
            return True
        
        return first.fitness.betterThan(second.fitness)