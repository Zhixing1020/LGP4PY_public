
from src.ec import *
from tasks.problem import Problem
from src.lgp.algorithm.typed_lgp.ec.typed_data import TypedData
from src.lgp.algorithm.typed_lgp.individual.primitives import TypedConstant
import numpy as np
# import tasks.ARC.DSL.constants as constants
import tasks.symbreg.DSL.basic_dsl.sr_types as srtype
# import src.lgp.algorithm.typed_lgp.dsl.type_checker as tc
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
from typing import Union

class SRConstant(TypedConstant):

    def __init__(self, val=0.0, name:str=None):
        super().__init__(val=val, name=name)

    def setValue(self, val=0, name:str=None):
        self.value = val
        
        if self.agent.tc.check_type(self.value, srtype.Float):
            self.output_type = srtype.Float
        elif self.agent.tc.check_type(self.value, srtype.Floats):
            self.output_type = srtype.Floats
        elif self.agent.tc.check_type(self.value, srtype.Boolean):
            self.output_type = srtype.Boolean
        # elif self.agent.tc.check_type(self.value, arctype.Integer):
        #     self.output_type = arctype.Integer
        # elif self.agent.tc.check_type(self.value, arctype.IntegerTuple):
        #     self.output_type = arctype.IntegerTuple    
        # elif self.agent.tc.check_type(self.value, arctype.Integers):
        #     self.output_type = arctype.Integers
        else:
            raise Exception(f"unknown output type of {self.value}")    
        self.name = name

