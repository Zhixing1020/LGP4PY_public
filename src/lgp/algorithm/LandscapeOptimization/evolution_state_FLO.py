from src.ec import *
from src.lgp.algorithm.LandscapeOptimization.subpopulationFLO import SubpopulationFLO

class EvolutionStateFLO (EvolutionState):

    def evolve(self) -> int:
        if self.generation > 0:
            self.output.message("Generation " + str(self.generation))

        # EVALUATION
        self.statistics.preEvaluationStatistics(self)
        self.evaluator.evaluatePopulation(self)
        self.statistics.postEvaluationStatistics(self)

        # SHOULD WE QUIT?
        if self.evaluator.runComplete(self) and self.quitOnRunComplete:
            self.output.message("Found Ideal Individual")
            return self.R_SUCCESS

        # SHOULD WE QUIT?
        if self.generation == self.numGenerations - 1:
            return self.R_FAILURE

        # PRE-BREEDING EXCHANGING
        # self.statistics.prePreBreedingExchangeStatistics(self)
        # self.population = self.exchanger.preBreedingExchangePopulation(self)
        # self.statistics.postPreBreedingExchangeStatistics(self)
        
        # update the leading board and indexes for sub-populations
        for s in range(len(self.population.subpops)):
            subpop = self.population.subpops[s]
            
            # Check if this subpopulation is of type SubpopulationFLO
            if isinstance(subpop, SubpopulationFLO):
                
                if self.generation % subpop.updateInterval == 0:
                    spop = subpop # Casting is implicit in Python
                    
                    thread = 0 # numThreads==1
                    
                    spop.updateBoard(self, thread)
                    spop.optimizeIndexes(self, thread)


        # BREEDING
        self.statistics.preBreedingStatistics(self)

        self.population = self.breeder.breedPopulation(self)

        # POST-BREEDING EXCHANGING
        self.statistics.postBreedingStatistics(self)

        # POST-BREEDING EXCHANGING
        # self.statistics.prePostBreedingExchangeStatistics(self)
        # self.population = self.exchanger.postBreedingExchangePopulation(self)
        # self.statistics.postPostBreedingExchangeStatistics(self)

        # INCREMENT GENERATION AND CHECKPOINT
        self.generation += 1
        # if self.checkpoint and (self.generation % self.checkpointModulo == 0):
        #     self.output.message("Checkpointing")
        #     self.statistics.preCheckpointStatistics(self)
        #     Checkpoint.setCheckpoint(self)
        #     self.statistics.postCheckpointStatistics(self)

        return self.R_NOTDONE