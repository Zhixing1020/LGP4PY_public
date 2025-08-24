import time
from typing import List, Optional
from pathlib import Path
from .statistics import Statistics
from src.ec.util import *
from ..gp_species import GPSpecies
from ..evolution_state import EvolutionState
from ..gp_node import GPNode
from ..gp_individual import GPIndividual

class SimpleShortStatistics(Statistics):
    P_STATISTICS_MODULUS = "modulus"
    P_COMPRESS = "gzip"
    P_FULL = "gather-full"
    P_DO_SIZE = "do-size"
    P_DO_TIME = "do-time"
    P_DO_SUBPOPS = "do-subpops"
    P_STATISTICS_FILE = "file"
    P_DO_DEPTH = "do-depth"

    def __init__(self):
        super().__init__()
        self.statisticslog = 0  # stdout by default
        self.modulus = 1
        self.doSize = False
        self.doTime = False
        self.doSubpops = False
        self.doDepth = False
        
        # Data structures
        self.bestSoFar:list[GPIndividual] = []
        self.totalSizeSoFar = []
        self.totalIndsSoFar = []
        self.totalIndsThisGen = []
        self.totalSizeThisGen = []
        self.totalFitnessThisGen = []
        self.bestOfGeneration:list[GPIndividual] = []
        
        # Tree statistics
        self.totalDepthSoFarTree = []
        self.totalSizeSoFarTree = []
        self.totalSizeThisGenTree = []
        self.totalDepthThisGenTree = []
        
        # Timing
        self.lastTime = 0

    def setup(self, state, base):
        super().setup(state, base)
        statisticsFile = state.parameters.getFile(base.push(self.P_STATISTICS_FILE), None)

        self.modulus = state.parameters.getIntWithDefault(
            base.push(self.P_STATISTICS_MODULUS), None, 1)

        if self.silentFile:
            self.statisticslog = Output.NO_LOGS
        elif statisticsFile is not None:
            try:
                compress = state.parameters.getBoolean(base.push(self.P_COMPRESS), None, False)
                self.statisticslog = state.output.addLog(
                    statisticsFile, not compress, compress)
            except IOError as i:
                state.output.fatal("An IOException occurred while trying to create the log " + 
                                 str(statisticsFile) + ":\n" + str(i))
        else:
            state.output.warning("No statistics file specified, printing to stdout at end.", 
                              base.push(self.P_STATISTICS_FILE))

        self.doSize = state.parameters.getBoolean(base.push(self.P_DO_SIZE), None, False)
        self.doTime = state.parameters.getBoolean(base.push(self.P_DO_TIME), None, False)
        
        if state.parameters.exists(base.push(self.P_FULL), None):
            state.output.warning(
                self.P_FULL + " is deprecated. Use " + self.P_DO_SIZE + " and " + 
                self.P_DO_TIME + " instead. Also be warned that the table columns have been reorganized.", 
                base.push(self.P_FULL), None)
            gather = state.parameters.getBoolean(base.push(self.P_FULL), None, False)
            self.doSize = self.doSize or gather
            self.doTime = self.doTime or gather
        
        self.doSubpops = state.parameters.getBoolean(base.push(self.P_DO_SUBPOPS), None, False)
        self.doDepth = state.parameters.getBoolean(base.push(self.P_DO_DEPTH), None, False)

    def getBestSoFar(self):
        return self.bestSoFar
    
    def preInitializationStatistics(self, state:EvolutionState):
        super().preInitializationStatistics(state)
        output = (state.generation % self.modulus == 0)
        
        if output and self.doTime:
            self.lastTime = time.time() * 1000  # milliseconds
    
    def postInitializationStatistics(self, state:EvolutionState):
        super().postInitializationStatistics(state)
        output = (state.generation % self.modulus == 0)
        
        # set up our bestSoFar array
        self.bestSoFar = [None] * len(state.population.subpops)
        
        # print out our generation number
        if output:
            state.output.print("0 ", self.statisticslog)
        
        # gather timings
        self.totalSizeSoFar = [0] * len(state.population.subpops)
        self.totalIndsSoFar = [0] * len(state.population.subpops)
        
        if output and self.doTime:
            state.output.print(f"{int(time.time() * 1000 - self.lastTime)} ", self.statisticslog)
        
        self.totalDepthSoFarTree = [None] * len(state.population.subpops)
        self.totalSizeSoFarTree = [None] * len(state.population.subpops)
        
        for x in range(len(state.population.subpops)):
            # check to make sure they're the right class
            if not isinstance(state.population.subpops[x].species, GPSpecies):
                state.output.fatal(f"Subpopulation {x} is not of the species form GPSpecies. Cannot do timing statistics with KozaShortStatistics.")
            
            i = state.population.subpops[x].individuals[0]
            self.totalDepthSoFarTree[x] = [0] * i.getTreesLength()
            self.totalSizeSoFarTree[x] = [0] * i.getTreesLength()
    
    # def postInitializationStatistics_extensive(self, state:EvolutionState):
    #     super().postInitializationStatistics(state)
    
    def preBreedingStatistics(self, state:EvolutionState):
        super().preBreedingStatistics(state)
        output = (state.generation % self.modulus == self.modulus - 1)
        
        if output and self.doTime:
            self.lastTime = time.time() * 1000  # milliseconds
        
        self.totalDepthThisGenTree = [None] * len(state.population.subpops)
        self.totalSizeThisGenTree = [None] * len(state.population.subpops)
        
        for x in range(len(state.population.subpops)):
            i = state.population.subpops[x].individuals[0]
            self.totalDepthThisGenTree[x] = [0] * i.getTreesLength()
            self.totalSizeThisGenTree[x] = [0] * i.getTreesLength()
    
    def postBreedingStatistics(self, state:EvolutionState):
        super().postBreedingStatistics(state)
        output = (state.generation % self.modulus == self.modulus - 1) or \
                    (state.generation == state.numGenerations - 2)
        
        if output:
            state.output.print(f"{state.generation + 1} ", self.statisticslog)
        
        if output and self.doTime:
            state.output.print(f"{int(time.time() * 1000 - self.lastTime)} ", self.statisticslog)
    
    def preEvaluationStatistics(self, state:EvolutionState):
        super().preEvaluationStatistics(state)
        output = (state.generation % self.modulus == 0)
        
        if output and self.doTime:
            self.lastTime = time.time() * 1000  # milliseconds
    
    def prepareStatistics(self, state:EvolutionState):
        pass
    
    def printExtraSubpopStatisticsAfter(self, state:EvolutionState, subpop):
        pass
    
    def gatherExtraPopStatistics(self, state:EvolutionState, subpop):
        pass
    
    def printExtraPopStatisticsAfter(self, state:EvolutionState):
        pass
    
    def gatherExtraSubpopStatistics(self, state:EvolutionState, subpop:int, individual:int):
        i:GPIndividual = state.population.subpops[subpop].individuals[individual]
        for z in range(i.getTreesLength()):
            self.totalDepthThisGenTree[subpop][z] += i.getTree(z).child.depth()
            self.totalDepthSoFarTree[subpop][z] += self.totalDepthThisGenTree[subpop][z]
            self.totalSizeThisGenTree[subpop][z] += i.getTree(z).child.numNodes(GPNode.NODESEARCH_ALL)
            self.totalSizeSoFarTree[subpop][z] += self.totalSizeThisGenTree[subpop][z]
    
    def printExtraSubpopStatisticsBefore(self, state:EvolutionState, subpop):
        if self.doDepth:
            state.output.print("[ ", self.statisticslog)
            for z in range(len(self.totalDepthThisGenTree[subpop])):
                val = (self.totalDepthThisGenTree[subpop][z] / self.totalIndsThisGen[subpop]) if self.totalIndsThisGen[subpop] > 0 else 0
                state.output.print(f"{val} ", self.statisticslog)
            state.output.print("] ", self.statisticslog)
        
        if self.doSize:
            state.output.print("[ ", self.statisticslog)
            for z in range(len(self.totalSizeThisGenTree[subpop])):
                val = (self.totalSizeThisGenTree[subpop][z] / self.totalIndsThisGen[subpop]) if self.totalIndsThisGen[subpop] > 0 else 0
                state.output.print(f"{val} ", self.statisticslog)
            state.output.print("] ", self.statisticslog)
    
    def printExtraPopStatisticsBefore(self, state:EvolutionState):
        totalDepthThisGenTreePop = [0] * len(self.totalDepthSoFarTree[0])
        totalSizeThisGenTreePop = [0] * len(self.totalSizeSoFarTree[0])
        totalIndsThisGenPop = 0
        
        subpops = len(state.population.subpops)
        
        for y in range(subpops):
            totalIndsThisGenPop += self.totalIndsThisGen[y]
            for z in range(len(totalSizeThisGenTreePop)):
                totalSizeThisGenTreePop[z] += self.totalSizeThisGenTree[y][z]
            for z in range(len(totalDepthThisGenTreePop)):
                totalDepthThisGenTreePop[z] += self.totalDepthThisGenTree[y][z]
        
        if self.doDepth:
            state.output.print("[ ", self.statisticslog)
            for z in range(len(totalDepthThisGenTreePop)):
                val = (totalDepthThisGenTreePop[z] / totalIndsThisGenPop) if totalIndsThisGenPop > 0 else 0
                state.output.print(f"{val} ", self.statisticslog)
            state.output.print("] ", self.statisticslog)
        
        if self.doSize:
            state.output.print("[ ", self.statisticslog)
            for z in range(len(totalSizeThisGenTreePop)):
                val = (totalSizeThisGenTreePop[z] / totalIndsThisGenPop) if totalIndsThisGenPop > 0 else 0
                state.output.print(f"{val} ", self.statisticslog)
            state.output.print("] ", self.statisticslog)
    
    def postEvaluationStatistics(self, state:EvolutionState):
        super().postEvaluationStatistics(state)
        
        output = (state.generation % self.modulus == 0)
        
        if output and self.doTime:
            # import resource
            # curU = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            state.output.print(f"{int(time.time() * 1000 - self.lastTime)} ", self.statisticslog)
        
        subpops = len(state.population.subpops)
        self.totalIndsThisGen = [0] * subpops
        self.bestOfGeneration = [None] * subpops
        self.totalSizeThisGen = [0] * subpops
        self.totalFitnessThisGen = [0.0] * subpops
        meanFitnessThisGen = [0.0] * subpops
        
        self.prepareStatistics(state)
        
        for x in range(subpops):
            for y in range(len(state.population.subpops[x].individuals)):
                if state.population.subpops[x].individuals[y].evaluated:
                    size = state.population.subpops[x].individuals[y].size()
                    self.totalSizeThisGen[x] += size
                    self.totalSizeSoFar[x] += size
                    self.totalIndsThisGen[x] += 1
                    self.totalIndsSoFar[x] += 1
                    
                    if (self.bestOfGeneration[x] is None or
                        state.population.subpops[x].individuals[y].fitness.betterThan(self.bestOfGeneration[x].fitness)):
                        self.bestOfGeneration[x] = state.population.subpops[x].individuals[y]
                        if (self.bestSoFar[x] is None or 
                            self.bestOfGeneration[x].fitness.betterThan(self.bestSoFar[x].fitness)):
                            self.bestSoFar[x] = self.bestOfGeneration[x].clone()
                    
                    self.totalFitnessThisGen[x] += state.population.subpops[x].individuals[y].fitness.fitness()
                    
                    self.gatherExtraSubpopStatistics(state, x, y)
            
            meanFitnessThisGen[x] = (self.totalFitnessThisGen[x] / self.totalIndsThisGen[x]) if self.totalIndsThisGen[x] > 0 else 0
            
            if output and self.doSubpops:
                self.printExtraSubpopStatisticsBefore(state, x)
            
            if output and self.doSize and self.doSubpops:
                val1 = (self.totalSizeThisGen[x] / self.totalIndsThisGen[x]) if self.totalIndsThisGen[x] > 0 else 0
                val2 = (self.totalSizeSoFar[x] / self.totalIndsSoFar[x]) if self.totalIndsSoFar[x] > 0 else 0
                state.output.print(f"{val1} {val2} ", self.statisticslog)
                state.output.print(f"{float(self.bestOfGeneration[x].size())} {float(self.bestSoFar[x].size())} ", self.statisticslog)
            
            if output and self.doSubpops:
                state.output.print(f"{meanFitnessThisGen[x]} ", self.statisticslog)
                state.output.print(f"{self.bestOfGeneration[x].fitness.fitness()} ", self.statisticslog)
                state.output.print(f"{self.bestSoFar[x].fitness.fitness()} ", self.statisticslog)
            
            if output and self.doSubpops:
                self.printExtraSubpopStatisticsAfter(state, x)
        
        popTotalInds = sum(self.totalIndsThisGen)
        popTotalIndsSoFar = sum(self.totalIndsSoFar)
        popTotalSize = sum(self.totalSizeThisGen)
        popTotalSizeSoFar = sum(self.totalSizeSoFar)
        popTotalFitness = sum(self.totalFitnessThisGen)
        popBestOfGeneration = None
        popBestSoFar = None
        
        for x in range(subpops):
            if (self.bestOfGeneration[x] is not None and 
                (popBestOfGeneration is None or self.bestOfGeneration[x].fitness.betterThan(popBestOfGeneration.fitness))):
                popBestOfGeneration = self.bestOfGeneration[x]
            if (self.bestSoFar[x] is not None and 
                (popBestSoFar is None or self.bestSoFar[x].fitness.betterThan(popBestSoFar.fitness))):
                popBestSoFar = self.bestSoFar[x]
            
            self.gatherExtraPopStatistics(state, x)
        
        popMeanFitness = (popTotalFitness / popTotalInds) if popTotalInds > 0 else 0
        
        if output:
            self.printExtraPopStatisticsBefore(state)
        
        if output and self.doSize:
            val1 = (popTotalSize / popTotalInds) if popTotalInds > 0 else 0
            val2 = (popTotalSizeSoFar / popTotalIndsSoFar) if popTotalIndsSoFar > 0 else 0
            state.output.print(f"{val1} {val2} ", self.statisticslog)
            state.output.print(f"{float(popBestOfGeneration.size())} {float(popBestSoFar.size())} ", self.statisticslog)
        
        if output:
            state.output.print(f"{popMeanFitness} ", self.statisticslog)
            state.output.print(f"{float(popBestOfGeneration.fitness.fitness())} ", self.statisticslog)
            state.output.print(f"{float(popBestSoFar.fitness.fitness())} ", self.statisticslog)
        
        if output:
            self.printExtraPopStatisticsAfter(state)
        
        if output:
            state.output.println("", self.statisticslog)