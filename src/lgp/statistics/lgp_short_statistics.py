


import time
from src.ec import *
from src.lgp.individual import LGPIndividual

class LGPStatistics(SimpleShortStatistics):
    def __init__(self):
        super().__init__()
        self.start = 0
    
    def setup(self, state, base):
        super().setup(state, base)
    
    def preInitializationStatistics(self, state:EvolutionState):
        totalTime = 0
        # self.start = yimei.util.Timer.getCpuTime()
        self.start = time.time()  # Using Python's time as approximation for yimei.util.Timer.getUserTime()
    
    def postInitializationStatistics(self, state:EvolutionState):
        super().postInitializationStatistics(state)
        
        self.totalDepthSoFarTree = [None] * len(state.population.subpops)
        self.totalSizeSoFarTree = [None] * len(state.population.subpops)
        
        for x in range(len(state.population.subpops)):
            if not isinstance(state.population.subpops[x].species, GPSpecies):
                state.output.fatal(f"Subpopulation {x} is not of the species form GPSpecies. Cannot do timing statistics with KozaShortStatistics.")
            
            i:LGPIndividual = state.population.subpops[x].individuals[0]
            self.totalDepthSoFarTree[x] = [0] * i.getMaxNumTrees()
            self.totalSizeSoFarTree[x] = [0] * i.getMaxNumTrees()
    
    def postEvaluationStatistics(self, state:EvolutionState):
        output = (state.generation % self.modulus == 0) or (state.generation == state.numGenerations-1)
        
        if not output:
            return

        
        if output and self.doTime:
            state.output.print(f"{int(time.time() * 1000 - self.lastTime)} ", self.statisticslog)
        
        subpops = len(state.population.subpops)
        self.totalIndsThisGen = [0] * subpops
        self.bestOfGeneration = [None] * subpops
        self.totalSizeThisGen = [0] * subpops
        self.totalFitnessThisGen = [0.0] * subpops
        
        totalAbsProgLength = [0] * subpops
        totalEffProgLength = [0] * subpops
        totalEffRate = [0.0] * subpops
        absProgLength_bestind = [0] * subpops
        effProgLength_bestind = [0] * subpops
        
        meanFitnessThisGen = [0.0] * subpops
        
        # piplines:list[map] = []

        self.prepareStatistics(state)
        
        for x in range(subpops):
            # breedpipeline_map = {source : 0 for source in state.population.subpops[x].species.pipe_prototype.sources}
            for y, ind in enumerate(state.population.subpops[x].individuals):
                if ind.evaluated:
                    size = ind.size()
                    proglength = ind.getTreesLength()
                    effproglength = ind.countStatus()
                    
                    self.totalSizeThisGen[x] += size
                    self.totalSizeSoFar[x] += size
                    self.totalIndsThisGen[x] += 1
                    self.totalIndsSoFar[x] += 1

                    # if ind.breedingPipe:
                    #     breedpipeline_map[ind.breedingPipe] += 1

                    if (self.bestOfGeneration[x] is None or
                        ind.fitness.betterThan(self.bestOfGeneration[x].fitness)):
                        self.bestOfGeneration[x] = ind
                        absProgLength_bestind[x] = proglength
                        effProgLength_bestind[x] = effproglength
                        
                        if (self.bestSoFar[x] is None or 
                            self.bestOfGeneration[x].fitness.betterThan(self.bestSoFar[x].fitness)):
                            self.bestSoFar[x] = self.bestOfGeneration[x].clone()
                    
                    self.totalFitnessThisGen[x] += ind.fitness.fitness()
                    
                    totalAbsProgLength[x] += proglength
                    totalEffProgLength[x] += effproglength
                    totalEffRate[x] += effproglength / proglength if proglength > 0 else 0
                    
                    self.gatherExtraSubpopStatistics(state, x, y)
                
            # piplines.append(breedpipeline_map)
            
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
        
        popAbsProgLength = sum(totalAbsProgLength)
        popEffProgLength = sum(totalEffProgLength)
        popEffRate = sum(totalEffRate)
        popBestAbsProgLength = 0
        popBestEffProgLength = 0
        
        popBestOfGeneration = None
        popBestSoFar = None
        
        for x in range(subpops):
            if (self.bestOfGeneration[x] is not None and 
                (popBestOfGeneration is None or self.bestOfGeneration[x].fitness.betterThan(popBestOfGeneration.fitness))):
                popBestOfGeneration = self.bestOfGeneration[x]
                popBestAbsProgLength = absProgLength_bestind[x]
                popBestEffProgLength = effProgLength_bestind[x]
            
            if (self.bestSoFar[x] is not None and 
                (popBestSoFar is None or self.bestSoFar[x].fitness.betterThan(popBestSoFar.fitness))):
                popBestSoFar = self.bestSoFar[x]
            
            self.gatherExtraPopStatistics(state, x)
        
        state.nodeEvaluation += popTotalSize
        
        popMeanFitness = (popTotalFitness / popTotalInds) if popTotalInds > 0 else 0
        
        if popTotalInds > 0:
            popAbsProgLength /= popTotalInds
            popEffProgLength /= popTotalInds
            popEffRate /= popTotalInds
        else:
            popAbsProgLength = 0
            popEffProgLength = 0
            popEffRate = 0
        
        finish = time.time()  # Using Python's time as approximation for yimei.util.Timer.getUserTime()
        duration = (finish - self.start)
        
        if output:
            self.printExtraPopStatisticsBefore(state)
        
        if output and self.doSize:
            val1 = (popTotalSize / popTotalInds) if popTotalInds > 0 else 0
            val2 = (popTotalSizeSoFar / popTotalIndsSoFar) if popTotalIndsSoFar > 0 else 0
            state.output.print(f"{val1} {val2} ", self.statisticslog)
            state.output.print(f"{float(popBestOfGeneration.size())} {float(popBestSoFar.size())} ", self.statisticslog)
        
        if output:
            state.output.print(f"{popMeanFitness}\t", self.statisticslog)
            state.output.print(f"{popBestOfGeneration.fitness.fitness()}\t", self.statisticslog)
            state.output.print(f"{float(popBestSoFar.fitness.fitness())}\t", self.statisticslog)
            state.output.print(f"{popAbsProgLength}\t", self.statisticslog)
            state.output.print(f"{popEffProgLength}\t", self.statisticslog)
            state.output.print(f"{popEffRate}\t", self.statisticslog)
            state.output.print(f"{popBestAbsProgLength}\t", self.statisticslog)
            state.output.print(f"{popBestEffProgLength}\t", self.statisticslog)
            state.output.print(f"{popBestEffProgLength / popBestAbsProgLength if popBestAbsProgLength > 0 else 0}\t", self.statisticslog)
            state.output.print(f"{duration}", self.statisticslog)

            # for subp_map in piplines:
            #     for _, value in subp_map.items():
            #         state.output.print(f"{value}\t", self.statisticslog)
            #     state.output.print(f"\t", self.statisticslog)
        
        if output:
            self.printExtraPopStatisticsAfter(state)
        
        if output:
            state.output.println("", self.statisticslog)
    
    def prepareStatistics(self, state:EvolutionState):
        self.totalDepthThisGenTree = [None] * len(state.population.subpops)
        self.totalSizeThisGenTree = [None] * len(state.population.subpops)
        
        for x in range(len(state.population.subpops)):
            i:LGPIndividual = state.population.subpops[x].individuals[0]
            self.totalDepthThisGenTree[x] = [0] * i.getMaxNumTrees()
            self.totalSizeThisGenTree[x] = [0] * i.getMaxNumTrees()
    
    def gatherExtraSubpopStatistics(self, state:EvolutionState, subpop:int, individual:int):
        i = state.population.subpops[subpop].individuals[individual]
        for z in range(i.getTreesLength()):
            self.totalDepthThisGenTree[subpop][z] += i.getTree(z).child.depth()
            self.totalDepthSoFarTree[subpop][z] += self.totalDepthThisGenTree[subpop][z]
            self.totalSizeThisGenTree[subpop][z] += i.getTree(z).child.numNodes(GPNode.NODESEARCH_ALL)
            self.totalSizeSoFarTree[subpop][z] += self.totalSizeThisGenTree[subpop][z]
    
    def finalStatistics(self, state:EvolutionState, result):
        if self.bestOfGeneration is None:
            print("The best individuals should be null?")
            return
        
        outIndividual = self.bestOfGeneration[0]

        for child in self.children:
            child.finalStatistics(state, result)