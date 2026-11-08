from src.lgp.algorithm.typed_lgp.dsl.agent import Agent
from src.ec import *
import copy
from typing import Any, Callable

# Global cache for type metadata to avoid repeated get_origin/get_args calls
_type_index_cache = {}

class TypedRegister:
    __slots__ = ("agent", "type", "value")
    def __init__(self, agent:Agent=None):
        # super().__init__()
        self.type = None  # will be a dict of {type: value}
        self.agent:Agent = agent
        # self._tmp_index = 0 # recently used index for speeding up
        if self.agent is not None:
            self.value = dict(self.agent.getTypeSet())  # can be the intermediate result of an instance, or an np.array of instances
        else:
            self.value = dict()

    def __eq__(self, other):
        if not isinstance(other, TypedRegister):
            return False
        res = self.type == other.getType()
        res = res and len(self.value) == len(other.value) and all(self.value[i] == other.value[i] for i, _ in enumerate(self.value.items()))
        return res
    
    def _get_type_index(self, typ):
        """Helper to get cached origin and args for a type."""
        if typ not in _type_index_cache:
            for (ty, _) in self.value.items():
                if self.agent.tc.type_equal(typ, ty):
                    _type_index_cache[typ] = ty
                    return ty
            else:
                raise Exception(f"Type {typ} not in register.")

        return _type_index_cache[typ]
    
    def setAgent(self, agent):
        self.agent = agent
        if self.agent is not None:
            self.value = dict(self.agent.getTypeSet())  # can be the intermediate result of an instance, or an np.array of instances
    
    def getType(self):
        return self.type
    
    def setType(self, type):
        self.type = type

    def getValue(self, type):
        # if type == Any:
        #     type = self.type
        # if (type is None and self.type is None) \
        #     or (type !=None and not any( self.agent.tc.type_equal(type, ty) for ty in self.value.keys())):
        #     raise Exception(f"Type {type} not in register.")
        
        # return self.value[type if type is not None else self.type]
        # type = type if type is not None and type!=Any else self.type

        # check the recently used index of type and value
        # value_list = list(self.value.items())
        # ty, val = value_list[self._tmp_index]
        # if self.agent.tc.type_equal(type, ty):
        #     res = val
        # else: # the recently used type and vlaue do not match, we need to enumerate all the types to find the matched one.
        #     for i,  (ty, val) in enumerate(value_list):
        #         if self.agent.tc.type_equal(type, ty):
        #             res = val
        #             self._tmp_index = i
        #             break
        #         # elif type == None and self.agent.tc.type_equal(self.type, ty):
        #         #     res = val
        #         #     break
        #     else:
        #         raise Exception(f"Type {type} not in register.")
        
        if self.agent.tc.is_parameterized_callable(type):
            # first, we check whether the register has the exact parameterized callable type, if not, we will check whether it has the parameterized callable type
            # if type not in _type_index_cache:
            #     for (ty, _) in self.value.items():
            #         if self.agent.tc.type_equal(type, ty):
            #             _type_index_cache[type] = ty
            #             return self.value[ty]
            #     # second, if yes, we return the value directly, if not, we will return a default callable function, returning the value with the output type in register regardless the input
            #     else:
            #         input_types, return_type = self.agent.tc.get_callable_types_param(type) 
            #         default_callable = self.agent.callable_default_types.get(Callable[..., return_type])
            #         return default_callable
            
            if type in self.value:
                return self.value[type]
            else:
                input_types, return_type = self.agent.tc.get_callable_types_param(type) 
                default_callable = self.agent.callable_default_types.get(Callable[..., return_type])
                return default_callable


        target_type = self._get_type_index(type)
        res = self.value[target_type]

        '''we do not check the type of output values from registers here, because we have checked the value when it is put into the register'''
        # if not self.agent.tc.check_type(res, type):
        #     raise Exception("getValue from registers return inconsistent results and types")  

        return res

        # raise Exception(f"Type: {type}, or self.type: {self.type}, are not in register.")
    
    def setValue(self, value, type=None, to_check_type=False):

        # if isinstance(value, TypedRegister):
        #     # self.value = copy.deepcopy(value.value)
        #     self.value = value.value.copy()
        #     return

        """Here, we assume the given type must be consistent with the value. There will be other errors in executing typed functions"""
        if to_check_type and type != None:
            if not self.agent.tc.check_type(value, type):
                # self.agent.tc.check_type(value, type)
                raise Exception(f"Value {value} is not of type {type}.")
            
        # # we make sure the value belongs to the type, but the type might be generic
        # possible_types = self.agent.getPossibleTypes(value)
        # if type != None and not any( self.agent.tc.type_equal(type, ty) for ty in possible_types)\
        #     and len(possible_types) == 0:
        # # if type != None and type not in possible_types:
        #     raise Exception(f"type {type} is not in the possible types: {possible_types}")
        
        # # we parameterize the type if it is generic
        # if not self.agent.tc.is_parameterized(type):
        #     for param_type in possible_types:
        #         if self.agent.tc.type_equal(param_type, type):
        #             type = param_type
        #             break
        
        self.setType(type)
        # for k in possible_types:
        self.value[type] = value

    def lightClone_wo_val(self):
        clone:TypedRegister = TypedRegister(self.agent)
        clone.setType(self.type)
        return clone

    def clone(self):
        clone:TypedRegister = TypedRegister(self.agent)
        clone.setType(self.type)
        clone.value = copy.deepcopy(self.value)
        return clone
    
    def __deepcopy__(self):
        return self.clone()
    