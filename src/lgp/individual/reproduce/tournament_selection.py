
from typing import List
from src.lgp.individual import *
from src.ec import *
from src.ec.util import *

class TournamentSelection(SelectionMethod):
    P_TOURNAMENT = "tournament"
    P_PICKWORST = "pick-worst"
    P_SIZE = "size"

    def __init__(self):
        super().__init__()
        self.size = 1
        self.probabilityOfPickingSizePlusOne = 0.0
        self.pickWorst = False

    def defaultBase(self)->Parameter:
        return SelectDefaults.base.push(self.P_TOURNAMENT)
    
    def setup(self, state, base):
        super().setup(state, base)
        def_ = self.defaultBase()

        val = state.parameters.getDoubleWithDefault(
            base.push(self.P_SIZE),
            def_.push(self.P_SIZE), 1.0)
        
        if val < 1.0:
            state.output.fatal(
                "Tournament size must be >= 1.",
                base.push(self.P_SIZE),
                def_.push(self.P_SIZE))
        elif val == int(val):  # it's just an integer
            self.size = int(val)
            self.probabilityOfPickingSizePlusOne = 0.0
        else:
            self.size = int(val)
            self.probabilityOfPickingSizePlusOne = val - self.size

        self.pickWorst = state.parameters.getBoolean(
            base.push(self.P_PICKWORST),
            def_.push(self.P_PICKWORST), False)

    def getTournamentSizeToUse(self, random):
        if self.probabilityOfPickingSizePlusOne == 0.0:
            return self.size
        return self.size + (1 if random.uniform(0, 1) < self.probabilityOfPickingSizePlusOne else 0)

    def getRandomIndividual(self, number:int, subpopulation, state:EvolutionState, thread):
        oldinds = state.population.subpops[subpopulation].individuals
        return state.random[thread].randint(0, len(oldinds)-1)

    def betterThan(self, first:GPIndividual, second:GPIndividual, subpopulation, state, thread):
        return first.fitness.betterThan(second.fitness)

    def produce_select(self, subpopulation, state, thread):
        oldinds = state.population.subpops[subpopulation].individuals
        best = self.getRandomIndividual(0, subpopulation, state, thread)
        
        s = self.getTournamentSizeToUse(state.random[thread])
        
        if self.pickWorst:
            for x in range(1, s):
                j = self.getRandomIndividual(x, subpopulation, state, thread)
                if not self.betterThan(oldinds[j], oldinds[best], subpopulation, state, thread):
                    best = j
        else:
            for x in range(1, s):
                j = self.getRandomIndividual(x, subpopulation, state, thread)
                if self.betterThan(oldinds[j], oldinds[best], subpopulation, state, thread):
                    best = j
        
        return best

    # SteadyState methods
    def individualReplaced(self, state, subpopulation, thread, individual):
        pass
    
    def sourcesAreProperForm(self, state):
        pass


from dataclasses import dataclass
@dataclass
class SelectDefaults:
    base:Parameter = Parameter("select")