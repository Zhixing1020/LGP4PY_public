from src.lgp.algorithm.typed_lgp.individual.primitives.typed_input import TypedInput
import tasks.symbreg.DSL.numpy_dsl.sr_np_types as sr_np_type
from src.lgp.individual.primitive import InputFeatureGPNode
from src.ec import *
from src.lgp.algorithm.typed_lgp.ec import TypedData
from tasks.symbreg.individual.primitives.sr_typed_inputfeature import SRInputFeature

class SRNumpyInputFeature(SRInputFeature):
    def __init__(self):
        super().__init__()
        self.output_type = sr_np_type.NpFloat

    def __eq__(self, other):
        return isinstance(other, SRNumpyInputFeature) and InputFeatureGPNode.__eq__(self, other) and TypedInput.__eq__(self, other)

    # def lightClone(self) -> 'SRNumpyInputFeature':
    #     obj:SRNumpyInputFeature = TypedInput.lightClone(self)
    #     obj.setIndex(self.index)
    #     obj.setRange(self.range)
    #     return obj

    # def clone(self) -> 'TypedInput':
    #     obj:SRInputFeature = TypedInput.clone(self)
    #     obj.setIndex(self.index)
    #     obj.setRange(self.range)
    #     return obj    