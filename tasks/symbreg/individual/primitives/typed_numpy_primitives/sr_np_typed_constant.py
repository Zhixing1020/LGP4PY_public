
from src.ec import *
from tasks.problem import Problem
from src.lgp.algorithm.typed_lgp.ec.typed_data import TypedData
from src.lgp.algorithm.typed_lgp.individual.primitives import TypedConstant
import numpy as np
# import tasks.ARC.DSL.constants as constants
import tasks.symbreg.DSL.numpy_dsl.sr_np_types as sr_np_type
# import src.lgp.algorithm.typed_lgp.dsl.type_checker as tc
# from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
# from typing import Union

from tasks.symbreg.individual.primitives import SRConstant

class SRNumpyConstant(SRConstant):

    def __init__(self, val=0.0, name:str=None):
        super().__init__(val=val, name=name)

    def setValue(self, val=0, name:str=None):
        self.value = np.float64(val)
        
        if isinstance(self.value, np.floating):
            self.output_type = sr_np_type.NpFloat
        else:
            raise Exception(f"unknown output type of {self.value}")    
        self.name = name

