from src.lgp.algorithm.typed_lgp.individual.primitives.typed_input import TypedInput
import tasks.symbreg.DSL.basic_dsl.sr_types as srtype
from src.ec import *
from src.lgp.algorithm.typed_lgp.ec import TypedData

class SRInput(TypedInput):
    def __init__(self):
        super().__init__()
        self.output_type = srtype.Floats
