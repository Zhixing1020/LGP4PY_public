import math
from src.ec import *
from src.lgp.individual.reproduce.tournament_selection import TournamentSelection

class AnealingTournamentSelection(TournamentSelection):

    def produce(self, subpopulation:int, state:EvolutionState, thread:int):
        # pick size random individuals, then pick the best.
        oldinds = state.population.subpops[subpopulation].individuals
        
        # Pick the first candidate
        best = self.getRandomIndividual(0, subpopulation, state, thread)
        
        # Determine the maximum tournament size defined in parameters
        max_s = self.getTournamentSizeToUse(state.random[thread])
        
        # Calculate dynamic tournament size based on current generation
        # Formula: min(max_s, ceil(sqrt(current_gen / total_gens) * max_s) + 1)
        current_gen_ratio = float(state.generation) / state.numGenerations
        scaled_s = math.ceil(math.sqrt(current_gen_ratio) * max_s) + 1
        
        s = int(min(max_s, scaled_s))
                
        if self.pickWorst:
            for x in range(1, s):
                j = self.getRandomIndividual(x, subpopulation, state, thread)
                # if j is at least as bad as best (not better)
                if not self.betterThan(oldinds[j], oldinds[best], subpopulation, state, thread):
                    best = j
        else:
            for x in range(1, s):
                j = self.getRandomIndividual(x, subpopulation, state, thread)
                # if j is better than best
                if self.betterThan(oldinds[j], oldinds[best], subpopulation, state, thread):
                    best = j
            
        return best