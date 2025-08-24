from .mutation import MutationPipeline
from .micro_mutation import LGPMicroMutationPipeline
from src.lgp.individual import *
from src.ec import *
from src.ec.util import *
from src.lgp.individual.primitive import *

class LGPSwapPipeline(LGPMicroMutationPipeline):
    SWAP = "swap"
    P_EFFFLAG = "effective"  # swap effective instructions or not
    P_STEP = "step"  # the number of swapping instructions on one side
    P_MICROMUTBASE = "micro_base"
    
    def __init__(self):
        super().__init__()
        self.stepSize = 0
        self.effflag = False
        self.microMutation:LGPMicroMutationPipeline = None
    
    def setup(self, state, base):
        super().setup(state, base)
        
        def_ = LGPDefaults.base.push(self.SWAP)
        
        self.stepSize = state.parameters.getInt(
            base.push(self.P_STEP), 
            def_.push(self.P_STEP))
        if self.stepSize < 1:
            state.output.fatal(
                "LGPFreeMutation Pipeline has an invalid number of step size (it must be >= 1).",
                base.push(self.P_STEP), def_.push(self.P_STEP))
        
        self.effflag = state.parameters.getBoolean(
            base.push(self.P_EFFFLAG),
            def_.push(self.P_EFFFLAG), True)
        
        microbase = Parameter(state.parameters.getString(
            base.push(self.P_MICROMUTBASE),
            def_.push(self.P_MICROMUTBASE)))
        self.microMutation = None
        if str(microbase) != "null":
            self.microMutation = state.parameters.getInstanceForParameter(
                microbase, def_.push(self.P_MICROMUTBASE), MutationPipeline)
            self.microMutation.setup(state, microbase)
    
    def produce(self, min, max, start, subpopulation, inds:list[LGPIndividual], state:EvolutionState, thread)->int:
        # grab individuals from our source
        n = self.sources[0].produce(min, max, start, subpopulation, inds, state, thread)

        # should we bother?
        # if not state.random[thread].nextBoolean(self.likelihood):
        #     return self.reproduce(n, start, subpopulation, inds, state, thread, False)

        # initializer = state.initializer
        
        # now let's mutate them
        for q in range(start, n + start):
            i = inds[q]
            inds[q] = self.produce_individual(subpopulation, i, state, thread)
        
        return n
    
    def produce_individual(self, subpopulation, ind, state:EvolutionState, thread)->LGPIndividual:
        i:LGPIndividual = ind
        j:LGPIndividual = None

        if isinstance(self.sources[0], BreedingPipeline):
            # it's already a copy
            j = i
        else:
            # need to clone the individual
            j = i.lightClone()
        
        if j.getTreesLength() == 1:
            # swapping only works with individuals with more than one instruction
            if self.microMutation is not None:
                j = self.microMutation.produce_individual(subpopulation, j, state, thread)
            return j
        
        if isinstance(self.sources[0], BreedingPipeline):
                j = i
        else:
            j = i.lightClone()

        # get the swap step size
        step = state.random[thread].randint(0, self.stepSize-1) + 1
        
        # perform the swaps
        for s in range(step):
            t = self.getLegalMutateIndex(j, state, thread)
            des = min(t + 1, j.getTreesLength() - 1)
            self.swap_instructions(j, t, des)
        
        j.evaluated = False
        
        # old_effrate = j.getEffTreesLength(update_status=False) / j.getTreesLength()

        # for x in range(self.numTries):
        #     if isinstance(self.sources[0], BreedingPipeline):
        #         j = i
        #     else:
        #         j = i.lightClone()

        #     # get the swap step size
        #     step = state.random[thread].randint(0, self.stepSize-1) + 1
            
        #     # perform the swaps
        #     for s in range(step):
        #         t = self.getLegalMutateIndex(j, state, thread)
        #         des = min(t + 1, j.getTreesLength() - 1)
        #         self.swap_instructions(j, t, des)
            
        #     j.evaluated = False
            
        #     new_effrate = j.getEffTreesLength(update_status=False) / j.getTreesLength()
            
        #     # break if effectiveness rate hasn't changed too much
        #     if abs(new_effrate - old_effrate) <= 0.3:
        #         break
        
        if self.microMutation is not None:
            j = self.microMutation.produce_individual(subpopulation, j, state, thread)
        
        j.breedingPipe = self
        return j
    
    def getLegalMutateIndex(self, ind, state, thread):
        res = state.random[thread].randint(0, ind.getTreesLength() - 1)
        
        if self.effflag:
            res = ind.getConditionIndex(state, thread,
                                         lambda x: x.status)
            # guarantee the effectiveness of the selected instruction
            # for x in range(self.numTries):
            #     if ind.getTree(res).status:
            #         break
            #     res = state.random[thread].randint(0, ind.getTreesLength() - 1)
        
        return res
    
    def move_instruction(self, ind:LGPIndividual, src, des):
        """Move instruction from src index to des index"""
        instr = ind.getTree(src)
        ind.removeTree(src, update_status=False)
        ind.addTree(des, instr)
    
    def swap_instructions(self, ind:LGPIndividual, p1, p2):
        """Swap instructions at positions p1 and p2"""
        instr1 = ind.getTree(p1)
        instr2 = ind.getTree(p2)
        
        ind.setTree(p1, instr2, update_status=False)
        ind.setTree(p2, instr1)