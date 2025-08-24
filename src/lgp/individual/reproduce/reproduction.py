from src.lgp.individual import *
from src.ec import *
from src.ec.util import *


class ReproductionPipeline(BreedingPipeline):
    P_REPRODUCE = "reproduce"
    P_MUSTCLONE = "must-clone"
    NUM_SOURCES = 1
    
    def __init__(self):
        super().__init__()
        self.mustClone = False
    
    def defaultBase(self)->Parameter:
        return BreedDefaults.base.push(self.P_REPRODUCE)
    
    def numSources(self):
        return self.NUM_SOURCES
    
    def setup(self, state, base):
        super().setup(state, base)
        def_ = self.defaultBase()
        self.mustClone = state.parameters.getBoolean(
            base.push(self.P_MUSTCLONE),
            def_.push(self.P_MUSTCLONE), False)
        
        # if self.likelihood != 1.0:
        #     state.output.warning(
        #         "ReproductionPipeline given a likelihood other than 1.0. "
        #         "This is nonsensical and will be ignored.",
        #         base.push(P_LIKELIHOOD),
        #         def_.push(P_LIKELIHOOD))
    
    def produce(self, min, max, start, subpopulation, inds, state, thread)->int:
        # grab individuals from our source and stick them right into inds
        n = self.sources[0].produce(min, max, start, subpopulation, inds, state, thread)
        
        if self.mustClone or isinstance(self.sources[0], SelectionMethod):
            for q in range(start, n + start):
                inds[q] = inds[q].clone()
        return n
    
from dataclasses import dataclass
@dataclass
class BreedDefaults:
    base:Parameter = Parameter("breed")