from src.ec.util import *
from src.ec import *

class SimpleStatistics(Statistics):
    """Python implementation of SimpleStatistics with all original functionality"""
    
    P_STATISTICS_FILE = "file"
    P_COMPRESS = "gzip"
    P_DO_FINAL = "do-final"
    P_DO_GENERATION = "do-generation"
    P_DO_MESSAGE = "do-message"
    P_DO_DESCRIPTION = "do-description"
    P_DO_PER_GENERATION_DESCRIPTION = "do-per-generation-description"

    def __init__(self):
        super().__init__()
        self.statisticslog = 0  # stdout
        self.best_of_run:list[GPIndividual] = None
        self.compress = False
        self.doFinal = True
        self.doGeneration = True
        self.doMessage = True
        self.doDescription = True
        self.doPerGenerationDescription = False
        self.warned = False

    def getBestSoFar(self):
        return self.best_of_run

    def setup(self, state:EvolutionState, base:Parameter):
        super().setup(state, base)
        
        self.compress = state.parameters.getBoolean(base.push(self.P_COMPRESS), None, False)
        
        statisticsFile = state.parameters.getFile(base.push(self.P_STATISTICS_FILE), None)

        self.doFinal = state.parameters.getBoolean(base.push(self.P_DO_FINAL), None, True)
        self.doGeneration = state.parameters.getBoolean(base.push(self.P_DO_GENERATION), None, True)
        self.doMessage = state.parameters.getBoolean(base.push(self.P_DO_MESSAGE), None, True)
        self.doDescription = state.parameters.getBoolean(base.push(self.P_DO_DESCRIPTION), None, True)
        self.doPerGenerationDescription = state.parameters.getBoolean(
            base.push(self.P_DO_PER_GENERATION_DESCRIPTION), None, False)

        if self.silentFile:
            self.statisticslog = Output.NO_LOGS
        elif statisticsFile is not None:
            try:
                self.statisticslog = state.output.addLog(statisticsFile, not self.compress, self.compress)
            except IOError as i:
                state.output.fatal(
                    f"An IOException occurred while trying to create the log {statisticsFile}:\n{i}")
        else:
            state.output.warning(
                "No statistics file specified, printing to stdout at end.", 
                f"{base}.{self.P_STATISTICS_FILE}")

    def postInitializationStatistics(self, state:EvolutionState):
        super().postInitializationStatistics(state)
        # Initialize best_of_run array after subpopulations are known
        self.best_of_run = [None] * len(state.population.subpops)

    def postEvaluationStatistics(self, state:EvolutionState):
        super().postEvaluationStatistics(state)
        
        best_i:list[GPIndividual] = [None] * len(state.population.subpops)
        for x in range(len(state.population.subpops)):
            best_i[x] = state.population.subpops[x].individuals[0]
            for y in range(1, len(state.population.subpops[x].individuals)):
                individual = state.population.subpops[x].individuals[y]
                
                if individual is None:
                    if not self.warned:
                        state.output.warnOnce("Null individuals found in subpopulation")
                        self.warned = True
                elif (best_i[x] is None or 
                      individual.fitness.betterThan(best_i[x].fitness)):
                    best_i[x] = individual
                    
                if best_i[x] is None and not self.warned:
                    state.output.warnOnce("Null individuals found in subpopulation")
                    self.warned = True
            
            # Update best_of_run if better individual found
            if (self.best_of_run[x] is None or 
                best_i[x].fitness.betterThan(self.best_of_run[x].fitness)):
                self.best_of_run[x] = best_i[x].clone()
        
        # Print generation statistics
        if self.doGeneration:
            state.output.println(f"\nGeneration: {state.generation}", self.statisticslog)
            state.output.println("Best Individual:", self.statisticslog)
            
        for x in range(len(state.population.subpops)):
            if self.doGeneration:
                state.output.println(f"Subpopulation {x}:", self.statisticslog)
                state.output.println(best_i[x].printIndividualForHuman(state), self.statisticslog)
                # best_i[x].printIndividualForHumans(state, self.statisticslog)
            
            if self.doMessage and not self.silentPrint:
                eval_status = " " if best_i[x].evaluated else " (evaluated flag not set): "
                state.output.message(
                    f"Subpop {x} best fitness of generation{eval_status}"
                    f"{best_i[x].fitness.value}")
            
            # if (self.doGeneration and self.doPerGenerationDescription and 
            #     isinstance(state.evaluator.p_problem, SimpleProblemForm)):
            #     state.evaluator.p_problem.clone().describe(
            #         state, best_i[x], x, 0, self.statisticslog)

    def finalStatistics(self, state:EvolutionState, result):
        super().finalStatistics(state, result)
        
        if self.doFinal:
            state.output.println("\nBest Individual of Run:", self.statisticslog)
            
        for x in range(len(state.population.subpops)):
            if self.doFinal:
                state.output.println(f"Subpopulation {x}:", self.statisticslog)
                state.output.println(self.best_of_run[x].printIndividualForHuman(state), self.statisticslog)
                # self.best_of_run[x].printIndividualForHumans(state, self.statisticslog)
            
            if self.doMessage and not self.silentPrint:
                state.output.message(
                    f"Subpop {x} best fitness of run: "
                    f"{self.best_of_run[x].fitness.value}")
            
            # if (self.doFinal and self.doDescription and 
            #     isinstance(state.evaluator.p_problem, SimpleProblemForm)):
            #     state.evaluator.p_problem.clone().describe(
            #         state, self.best_of_run[x], x, 0, self.statisticslog)