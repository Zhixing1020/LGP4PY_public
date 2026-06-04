from src.ec import *
from src.ec.util.parameter import Parameter
from src.lgp.algorithm.typed_lgp.ec.typed_data import TypedData
from tasks.problem import Problem
from typing import override
import numpy as np
from src.lgp.algorithm.typed_lgp.ec import *
# from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
# from src.lgp.individual.primitive.add import Add
from tasks.symbreg.DSL.numpy_dsl.sr_np_types import NpFloat
from tasks.symbreg.individual.primitives.typed_add import TypedAdd

class TypedNumpyAdd(TypedAdd):
    def __init__(self):
        super().__init__()
        self.output_type = NpFloat
        self.input_types = (NpFloat, NpFloat)
        
