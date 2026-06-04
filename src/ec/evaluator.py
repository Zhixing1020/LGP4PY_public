from tasks.problem import Problem
from src.ec import EvolutionState
from src.ec.util import Parameter, ParameterDatabase
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import copy

class Evaluator:
    """defining how to evaluate the individuals in the population"""

    P_EVALUATOR = "evaluator"
    P_PROBLEM = "problem"
    P_CLONE_PROBLEM = "clone-problem"

    def __init__(self, p_problem=None, numTests=1, cloneProblem=False):
        self.p_problem:Problem = p_problem
        self.numTests = numTests
        self.cloneProblem = cloneProblem

    def setup(self, state:EvolutionState, base:Parameter):
        
        def_base = Parameter(self.P_EVALUATOR)

        self.p_problem = state.parameters.getInstanceForParameter(base.push(self.P_PROBLEM), def_base.push(self.P_PROBLEM), Problem)
        if self.p_problem is None:
            state.output.fatal(f"Problem instance not found in parameters: {base.push(self.P_PROBLEM)} or {def_base.push(self.P_PROBLEM)}")
        self.p_problem.setup(state, base.push(self.P_PROBLEM))

        self.cloneProblem = state.parameters.getBoolean(base.push(self.P_CLONE_PROBLEM), def_base.push(self.P_CLONE_PROBLEM), False)
        if not self.cloneProblem and state.breedthreads > 1:
            state.output.fatal(f"The Evaluator is not cloning its Problem, but you have more than one thread: {base.push(self.P_CLONE_PROBLEM)} or {def_base.push(self.P_CLONE_PROBLEM)}.")

        # self.numTests = state.parameters.get_int(base.push("num-tests"), None, 1)
        # if self.numTests < 1:
        #     self.numTests = 1
        # elif self.numTests > 1:
        #     m = state.parameters.get_string(base.push("merge"), None)
        #     if m is None:
        #         state.output.warning("Merge method not provided to SimpleEvaluator. Assuming 'mean'")
        #     elif m.lower() == self.MERGE_MEAN:
        #         self.mergeForm = self.MERGE_MEAN
        #     elif m.lower() == self.MERGE_MEDIAN:
        #         self.mergeForm = self.MERGE_MEDIAN
        #     elif m.lower() == self.MERGE_BEST:
        #         self.mergeForm = self.MERGE_BEST
        #     else:
        #         state.output.fatal(f"Bad merge method: {m}", base.push("num-tests"), None)

        # if not state.parameters.exists(base.push("chunk-size"), None):
        #     self.chunkSize = self.C_AUTO
        # else:
        #     chunk_string = state.parameters.get_string(base.push("chunk-size"), None)
        #     if chunk_string.lower() == self.C_AUTO:
        #         self.chunkSize = self.C_AUTO
        #     else:
        #         self.chunkSize = state.parameters.get_int(base.push("chunk-size"), None, 1)
        #         if self.chunkSize == 0:
        #             state.output.fatal("Chunk Size must be either an integer >= 1 or 'auto'", base.push("chunk-size"), None)


    def evaluatePopulation(self, state:EvolutionState):
        # if self.numTests > 1:
        #     self.expand(state)

        # individualCounter = 0
        # subPopCounter = 0

        subpops = state.population.subpops
        num_subpops = len(subpops)
        numinds = [len(subpop.individuals) for subpop in subpops]
        from_index = [0] * num_subpops

        if state.evalthreads == 1:
            prob = copy.deepcopy(self.p_problem) if self.cloneProblem else self.p_problem
            self.evalPopChunk(state, numinds, from_index, 0, prob)
        else:
            with ProcessPoolExecutor(max_workers=state.evalthreads) as executor:
            # with ThreadPoolExecutor(max_workers=state.evalthreads) as executor:
                futures = []
                for i in range(state.evalthreads):
                    new_state = state.lightClone_w_pickable()
                    prob = copy.deepcopy(self.p_problem) if self.cloneProblem else self.p_problem
                    futures.append(
                        executor.submit(self.evalPopChunk, new_state, numinds, from_index, i, prob)
                    )
                for future in futures:
                    results = future.result()  # Wait for completion
                    for pop_idx, ind_idx, ind in results:
                        state.population.subpops[pop_idx].individuals[ind_idx] = ind

        # if self.numTests > 1:
        #     self.contract(state)

    def evalPopChunk(self, state:EvolutionState, numinds, from_index, threadnum, problem:Problem):
        results = []
        for pop_index, subpop in enumerate(state.population.subpops):
            n = len(subpop.individuals)
            chunk_size = (n + state.evalthreads - 1) // state.evalthreads
            start = threadnum * chunk_size
            end = min(start + chunk_size, n)
            for idx, ind in enumerate(subpop.individuals[start:end]):
                problem.evaluate(state, ind, pop_index, threadnum)
                results.append((pop_index, start + idx, ind))

        return results

    # def expand(self, state):
    #     pass  # stub for numTests > 1 case

    # def contract(self, state):
    #     pass  # stub for numTests > 1 case

    def runComplete(self, state:EvolutionState)->bool:
        for sp in state.population.subpops:
            for ind in sp.individuals:
                if(ind.fitness.isIdealFitness()):
                    return True
        return False