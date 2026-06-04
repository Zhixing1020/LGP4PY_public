from src.lgp.algorithm.typed_lgp.individual.primitives.typed_input import TypedInput
import tasks.symbreg.DSL.numpy_dsl.sr_np_types as sr_np_type
from src.ec import *
from src.lgp.algorithm.typed_lgp.ec import TypedData
from tasks.symbreg.individual.primitives.sr_typed_input import SRInput

class SRNumpyInput(SRInput):
    def __init__(self):
        super().__init__()
        self.output_type = sr_np_type.NpFloat
