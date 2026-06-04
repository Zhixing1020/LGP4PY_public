from src.lgp.algorithm.typed_lgp.individual.primitives.typed_input import TypedInput
import tasks.symbreg.DSL.basic_dsl.sr_types as srtype
from src.lgp.individual.primitive import InputFeatureGPNode
from src.ec import *
from src.lgp.algorithm.typed_lgp.ec import TypedData

class SRInputFeature(TypedInput, InputFeatureGPNode):
    def __init__(self):
        InputFeatureGPNode.__init__(self)
        TypedInput.__init__(self)
        self.output_type = srtype.Float

    def __str__(self):
        return InputFeatureGPNode.__str__(self)
    
    def eval(self, state:EvolutionState, thread:int, input:TypedData, individual, problem, argval: list = None):
        InputFeatureGPNode.eval(self, state, thread, input, individual, problem, argval)

    def __eq__(self, other):
        return isinstance(other, SRInputFeature) and InputFeatureGPNode.__eq__(self, other) and TypedInput.__eq__(self, other)
    
    def resetNode(self, state, thread):
        TypedInput.resetNode(self, state, thread)
        InputFeatureGPNode.resetNode(self, state, thread)

    def lightClone(self) -> 'SRInputFeature':
        obj:SRInputFeature = TypedInput.lightClone(self)
        obj.setIndex(self.index)
        obj.setRange(self.range)
        return obj

    def clone(self) -> 'TypedInput':
        obj:SRInputFeature = TypedInput.clone(self)
        obj.setIndex(self.index)
        obj.setRange(self.range)
        return obj    