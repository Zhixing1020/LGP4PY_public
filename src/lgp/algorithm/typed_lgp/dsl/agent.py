# function name, argument types, return types

import os
import json
import inspect
import importlib
import copy

from src.ec import *
from src.ec.util import *

from typing import Any, Union, Callable, Optional, Tuple, List, FrozenSet, Container, TypeAlias, get_type_hints, get_origin, get_args
from collections.abc import Container as runtimeContainer
import collections.abc
from functools import wraps
from src.lgp.algorithm.typed_lgp.dsl.type_checker import TypeChecker

class Agent:
    P_AGENT = "typeagent"
    P_DSL = "dsl_module"
    P_TYPE = "type_module"
    P_DEFAULT_VAL = "default_val_module"
    P_CONSTANT = "constant_module"
    P_INPUT_TYPE = "input_type"
    P_TARGET_TYPE = "target_type"

    def __init__(self):
        #scan the dsl.py file to get the definitions of all functions (including input and output types)
        # the dictionary is like {"func_name": (arg_types, return_type)}

        # self._dsl_module = None
        # self._type_module = None
        # self._default_val_module = None
        # self._constant_module = None

        self.dsl_name = None
        self.type_name = None
        self.defval_name = None
        self.constant_name = None

        # need run_dsl_tests and test_solvers_formatting first to ensure the dsl.py and solvers.py are correctly implemented
        self.initial_funs = None 
        # self.all_exec_funs = None # all functions except those whose output types are callable
        self.basic_funs = set() # all input / output types are non-callable types

        self.all_funs:dict = None  # all functions by parameterizing the child function and input output types
        self.funs_output_lookup:dict = None # a dictionary for checking functions with a certain output type
        self.all_maps:list = None  # a list of all possible map entity

        self.funs_w_comp = set() # some of the input types are callable functions (2 args) returning boolean type
        self.comp_funs = set() # accepting two input non-callable arguments, returning a boolean (denoted by argument name: compfunc)

        self.funs_w_cond = set() # some of the input types are callable functions (1 args) returning boolean type
        self.condition_funs = set() # accepting one input non-callable argument, returning a boolean (denoted by arg name: condition)

        self.funs_w_exec_container = set() # functions with 1 callable basic functions (basic_funs) and apply to all the elements of a container, 
        self.container_funs = set()

        self.basic_types = None  # defined manually
        self.basic_type_names = None # access the basic types by their names
        self.all_types = None   # include basic types and the automatically instantiated types

        self.callable_default_types = None # the default types for callable functions, which can be used to instantiate the parameterized callable functions. The default types are defined by the output types of the basic functions that can be used as child functions.

        self.all_constants = None
        self.cons_output_lookup:dict = None # the key is the "output type", to speed up constants look up with output type

        self.basic_feature:dict = {}  # all typed features, {name, obj}
        self.feature_output_lookup:dict = {} # a dictionary for looking up features with a certain output type

        self.input_type = None # the raw input type of the problem
        self.target_type = None # the target output type of the problem
        self.effSourceTypes:set = None # the effective input types based on constants and features

        self.tc = TypeChecker() # type checker

    @classmethod
    def defaultBase(cls):
        return Parameter(cls.P_AGENT)

    @classmethod
    def load_module(cls, module_name: str):
        return importlib.import_module(module_name)
    
    # @property
    # def dsl_module(self):
    #     if self._dsl_module is None:
    #         self._dsl_module = Agent.load_module(self.dsl_name)
    #     return self._dsl_module
    
    # @property
    # def default_val_module(self):
    #     if self._default_val_module is None:
    #         self._default_val_module = Agent.load_module(self.defval_name)
    #     return self._default_val_module
    
    # @property
    # def constant_module(self):
    #     if self._constant_module is None:
    #         self._constant_module = Agent.load_module(self.constant_name)
    #     return self._constant_module
    
    # @property
    # def tc(self):
    #     if self._type_checker_module is None:
    #         self._type_checker_module = tc
    #     return self._type_checker_module
        
    def setup(self, state: EvolutionState, base: Parameter):
        # load the modules
        def_base = self.defaultBase()

        self.dsl_name = state.parameters.getString(base.push(self.P_DSL), def_base.push(self.P_DSL))
        # self._dsl_module = Agent.load_module(self.dsl_name)

        self.type_name = state.parameters.getString(base.push(self.P_TYPE), def_base.push(self.P_TYPE))
        # self._type_module = Agent.load_module(self.type_name)

        self.defval_name = state.parameters.getString(base.push(self.P_DEFAULT_VAL), def_base.push(self.P_DEFAULT_VAL))
        # self._default_val_module = Agent.load_module(self.defval_name)

        self.constant_name = state.parameters.getString(base.push(self.P_CONSTANT), def_base.push(self.P_CONSTANT))
        # self._constant_module = Agent.load_module(self.constant_name)

        # TODO: apply "run_dsl_tests" and "test_solvers_formatting" to check the dsl definition

        self.getTypeSet()
        input_type_name = state.parameters.getString(base.push(self.P_INPUT_TYPE),
                                               def_base.push(self.P_INPUT_TYPE))
        output_type_name = state.parameters.getString(base.push(self.P_TARGET_TYPE),
                                               def_base.push(self.P_TARGET_TYPE))
        self.input_type = self.getTypeByName(input_type_name)
        self.target_type = self.getTypeByName(output_type_name)

        
        self._getInitialFunctionSet()

        self._check_DSL_formatting()

        self.getConstantSet()
        
        self.getFunctionSet()

        self.getFeatureSet()

        self.getSourceTypes()

    def _getInitialFunctionSet(self)->dict:
        # return the dictionary of all possible functions
        if self.initial_funs is not None:
            return self.initial_funs
        
        dsl_module = Agent.load_module(self.dsl_name)

        self.initial_funs = {
            name: obj
            for name, obj in dsl_module.__dict__.items() if not name.startswith('_') 
            and callable(getattr(dsl_module, name)) 
            and getattr(dsl_module, name).__module__ == dsl_module.__name__
        }

        # update all_exec_funs, basic_funs, funs_w_bool, funs_w_exec
        self._categorize_functions_features(self.initial_funs)
   
        return self.initial_funs
    
    def _categorize_functions_features(self, function_set:dict):
        """update all_exec_funs, basic_funs, funs_w_bool, funs_w_exec, and features"""
        self.basic_funs |= set([f for f in function_set.values() 
                                if self.getOutputType(f) != Callable 
                                and Callable not in self.getInputTypes(f)])
        self.funs_w_comp |= set([f for f in function_set.values() 
                                if self.getInputTypes(f).count(Callable) == 1 
                                and 'compkey' in get_type_hints(f).keys()
                                and 'container' in get_type_hints(f).keys()])
        self.comp_funs |= set([f for f in self.basic_funs 
                               if "Integer" in self.basic_type_names.keys() and self.getOutputType(f) == self.basic_type_names["Integer"]
                                and len(self.getInputTypes(f)) == 1])
        
        self.funs_w_cond |= set([f for f in function_set.values() 
                                if self.getInputTypes(f).count(Callable) == 1 
                                and 'condition' in get_type_hints(f).keys()
                                and 'container' in get_type_hints(f).keys()])
        self.condition_funs |= set([f for f in self.basic_funs 
                               if  "Boolean" in self.basic_type_names.keys() and  self.getOutputType(f) == self.basic_type_names["Boolean"]
                                and len(self.getInputTypes(f)) == 1])
        
        self.funs_w_exec_container |= set([f for f in function_set.values() 
                                if self.getInputTypes(f).count(Callable) == 1
                                and 'function' in get_type_hints(f).keys()
                                and 'container' in get_type_hints(f).keys()
                                ])

        self.container_funs |= set([f for f in self.basic_funs 
                               if len(self.getInputTypes(f)) == 1])
        
        
        # self.basic_feature = {}
        # self.feature_output_lookup = {}

        tmp = {name: f for name, f in function_set.items() 
                                  if len(self.getInputTypes(f)) == 1
                                   and self.input_type in self.getInputTypes(f) }
        
        for name, fun in tmp.items():

            map_list = self._get_map_list(name, fun)

            self.basic_feature.update({name: map_list})

            for map in map_list:
                if map['output_type'] not in self.feature_output_lookup:
                    self.feature_output_lookup[ map['output_type'] ] = [map]
                else:
                    self.feature_output_lookup[ map['output_type'] ].append(map)


    def _get_map_list(self, name:str, fun:Callable, arg=None, arg_out=None, level_down_nest:bool=False)->list:
        # return a list of IO mapping given arg, arg_out, and level_down_nest. Each mapping is a dictionry (refer to map entity)
        # arg: the primary hint type
        # arg_out: the output hint type. for some functions, the output hint type is different from the primary hint type due to the child function
        # level_down_nest: if true, the IO mapping can only use the type T, where container[T] is in the type set as well.
        
        # 1. a function that enumerate all possible seeding type (the first sampling type)
        # 2. a function that enumerate all possible type lists (as IO types are correlated) given a function and a seed type
        # 3. construct a parameterized function

        map_list = []

        # list all possible seed types
        def get_seed_type_list(fun, arg=None, arg_out=None, level_down_nest:bool=False)->list:
            """return a list of possible seed types. A seed type carries the information to construct a possible type list
            arg and arg_out are for child function. The seed type is also determined by the IO types of the child function"""
            to_do_type_list = Agent.getInputTypes(fun) + [Agent.getOutputType(fun)]
            
            #here, because get_seed_type_list() only accepts basic functions as "fun" whose input and output types are specified, 
            #self.tc.is_parameterized() returns true for Callable function is acceptable. 
            unparam_type_list = [tar for tar in to_do_type_list if not self.tc.is_parameterized(tar)]

            if len(unparam_type_list) == 0:
                return []
                
            if len([ty for ty in unparam_type_list if get_origin(ty) != None and issubclass(get_origin(ty), runtimeContainer)]) > 0:
                # e.g., container, frozenset, tuple
                # hinttype can be any container, value must be args(hinttype) 
                if level_down_nest: # it is parameterizing a child function
                    type_list = list(set(get_args(typ)[0] for typ in self.getTypeSet().keys()
                            if len(get_args(typ)) == 1
                            and self.tc.type_equal(get_args(typ)[0], Container)))
                elif arg != None: # the hint type is given, we must find container types whose agrs is the same as arg
                    type_list = [typ for typ in self.getTypeSet().keys()
                            if self.tc.type_equal(typ, Container)
                            and self.tc.type_equal( get_args(typ)[0], arg)]
                else: 
                    type_list = [typ for typ in self.getTypeSet().keys()
                            if self.tc.type_equal(typ, Container)] # if the generic type (unparameterized type) is frozen or tuple, 
                    # we only use the args of the type. Thus, the "Container" here does not matter.
                    
            elif len([ty for ty in unparam_type_list if ty is Any]) > 0:
                # e.g., Any
                # hinttype can be anything
                if level_down_nest:
                    type_list = list(set(get_args(typ)[0] for typ in self.getTypeSet().keys()
                            if len(get_args(typ))==1 ))
                elif arg == None:
                    type_list = list(self.getTypeSet().keys())
            else:
                # unknown generic
                raise Exception("unknown situation when parameterizing hint types")

            if len(type_list) == 0:
                raise Exception(f"there is no parameterized type consistent with the hint type {Container}")
            
            return type_list

        def get_IOtyple_tuple(fun, seed_type, arg=None, arg_out=None)->tuple:
            # return a IO type tuple. Each tuple is used to produce one parameterized function
            to_do_type_list = Agent.getInputTypes(fun) + [Agent.getOutputType(fun)]
            
            unparam_type_list = [tar for tar in to_do_type_list if not self.tc.is_parameterized(tar)]

            IO_types = list(to_do_type_list)

            if len(unparam_type_list) == 0:
                return ()

            if len([ty for ty in unparam_type_list if get_origin(ty) != None and issubclass(get_origin(ty), runtimeContainer)]) > 0:
                # e.g., container
                origin = get_origin(seed_type)

                if arg == None:
                    # get the (nested) hint type from the args
                    arg = get_args(seed_type)[0]

                for i, typ in enumerate(IO_types):
                    if self.tc.is_parameterized(typ):
                        continue
                    if i == len(IO_types) - 1 and arg_out != None:
                        arg = arg_out
                    # parameterized based on the seed type
                    if get_origin(typ) != None and issubclass(get_origin(typ), runtimeContainer) and get_origin(typ) != runtimeContainer: # e.g., frozenset
                        # in this case, the (nested) type is type_equal to the arg
                        IO_types[i] = get_origin(typ)[arg]
                    elif get_origin(typ) == runtimeContainer: # container
                        # find a type which is a container and its arg the same as arg
                        IO_types[i] = origin[arg]
                    elif typ is Any:
                        IO_types[i] = arg
            
            elif len([ty for ty in unparam_type_list if ty is Any]) > 0:
                # e.g., Any
                arg = seed_type if arg == None else arg

                for i, typ in enumerate(IO_types):
                    if self.tc.is_parameterized(typ):
                        continue
                    if i == len(IO_types) - 1 and arg_out != None:
                        arg = arg_out
                    if typ is Any:
                        IO_types[i] = arg
            else:
                # unknown generic
                raise Exception("unknown situation when parameterizing hint types")
            
            # check the resulting type consistent with the current generic input output types
            # if all good, update input output types
            for i, t in enumerate(to_do_type_list):
                if not self.tc.type_equal(to_do_type_list[i], IO_types[i]):
                    raise Exception("inconsistent types after parameterization")
                if t!=Callable and not any( self.tc.type_equal(t, ty) for ty in self.getTypeSet().keys()):
                    raise Exception(f"Type {t} not in register.")
            
            return tuple(IO_types)
            
        def construct_map_entity(name, obj, child_fun, input:tuple, output)->dict:
            return {
                'name':name,
                'obj':obj,
                'child_fun':child_fun,
                'input_type':input,
                'output_type':output
            }

        IO_types = Agent.getInputTypes(fun) + [Agent.getOutputType(fun)]
            
        if len( [tar for tar in IO_types if not self.tc.is_parameterized(tar)] ) == 0:
            # this function is parameterized, we add it to all_funs
            map = construct_map_entity(name, fun, None, tuple(Agent.getInputTypes(fun)), Agent.getOutputType(fun))
            map_list.append(map)
            
        elif Callable not in self.getInputTypes(fun): # this function has no child function
            seed_type_list = get_seed_type_list(fun, arg, arg_out, level_down_nest)
            for seed_type in seed_type_list:
                IOtype = get_IOtyple_tuple(fun, seed_type, arg, arg_out)
                map = construct_map_entity(name, fun, None, tuple(IOtype[:-1]), IOtype[-1] )
                map_list.append(map)

        else:
            # then we have to handle child function
            # 1. get all possible child functions
            if fun in self.funs_w_comp:
                child_fun_set = self.comp_funs
            elif fun in self.funs_w_cond:
                child_fun_set = self.condition_funs
            elif fun in self.funs_w_exec_container:
                child_fun_set = self.container_funs
            else:
                raise Exception(f"there is no matched child function for {name}")

            # for each child function, get all possible IO hint_types
            for cf in child_fun_set:
                child_map_list = self._get_map_list(cf.__name__, cf, 
                                        level_down_nest='container' in get_type_hints(fun).keys())

                # after we have all the IO hint types of child function, we can have the type list of a certain function with every IO hint types
                for c_map in child_map_list:
                    hint_type_in = next(t for t in c_map['input_type'] if t != Callable) 
                    hint_type_out = hint_type_in if ('compkey' in get_type_hints(fun).keys() \
                    or 'condition' in get_type_hints(fun).keys()) \
                    and 'container' in get_type_hints(fun).keys() else c_map['output_type']

                    seed_type_list = get_seed_type_list(fun, arg=hint_type_in, arg_out=hint_type_out)
                    for seed_type in seed_type_list:
                        IOtype = get_IOtyple_tuple(fun, seed_type, arg=hint_type_in, arg_out=hint_type_out)
                        map = construct_map_entity(name, fun, c_map['obj'], tuple(IOtype[:-1]), IOtype[-1] )
                        map_list.append(map)

        return map_list

    @staticmethod
    def getInputTypes(fun: Callable)->list:
        # return the list of input types
        hints = get_type_hints(fun)

        res = []
        for k, v in hints.items():
            if k != 'return':
                res.append(v)
        
        return res

    @staticmethod
    def getOutputType(fun:Callable)->Any:
        # return the output type
        hints = get_type_hints(fun)

        res = hints.get('return', None)
        
        return res
    
    def getFunctionSet(self)->dict:
        # for all initial functions, we parameterize it by all possible child functions and parameterized types
        if self.all_funs is not None:
            return self.all_funs
        
        self.all_funs = {} # all functions, indicated by "name, possible mappings"
        self.funs_output_lookup = {} # the key is the "output type", to speed up functions look up with output type
        self.all_maps = [] # all possible mappings

        for name, fun in self.initial_funs.items():

            map_list = self._get_map_list(name, fun)

            self.all_funs.update({name: map_list})
            self.all_maps += map_list

            for map in map_list:
                if map['output_type'] not in self.funs_output_lookup:
                    self.funs_output_lookup[ map['output_type'] ] = [map]
                else:
                    self.funs_output_lookup[ map['output_type'] ].append(map)
            
        if len(self.all_funs) == 0:
            raise Exception("We have no functions. Please check the primitive set")

        return self.all_funs

    
    def getFunctionListWithOutput(self, output_type=None, input_type=None, 
                                  state:EvolutionState=None, thread:int=None)->list:
        # return a list of map tuple
        res = []
        # for map_list in self.all_funs.values():
        #     for map in map_list:
        #         if output_type == None or tc.type_equal(map['output_type'], output_type):
        #             res.append(map)
        if output_type == None: # return all possible map tuples
            res = self.all_maps
        elif any(self.tc.type_equal(output_type, t) for t in self.funs_output_lookup.keys()):
            ty = next((t for t in self.funs_output_lookup.keys() if self.tc.type_equal(output_type, t)), None)
            tmp = self.funs_output_lookup[ty]
            if input_type == None or len(input_type) == 0:
                res = tmp
            else:
                res = [f for f in tmp if all( t in input_type for t in f['input_type'])]
                if len(res) == 0: # make sure the output_type is satisfied, then the input type
                    res = tmp
        # else:
        #     raise Exception(f"unknown expected output type {output_type}")

        return res
    
    def getFunctionListWithOutputInput(self, output_type=None, input_type=None, 
                                       state:EvolutionState=None, thread:int=None)->list:
        # return a list of map tuple with exact matching output type and input type
        res = []
        # for map_list in self.all_funs.values():
        #     for map in map_list:
        #         if output_type == None or tc.type_equal(map['output_type'], output_type):
        #             res.append(map)
        if output_type == None: # return all possible map tuples
            res = self.all_maps
        elif any(self.tc.type_equal(output_type, t) for t in self.funs_output_lookup.keys()):
            ty = next((t for t in self.funs_output_lookup.keys() if self.tc.type_equal(output_type, t)), None)
            tmp = self.funs_output_lookup[ty]
            if input_type == None or len(input_type) == 0:
                res = tmp
            else:
                res = [f for f in tmp if len(f['input_type']) == len(input_type) and all( self.tc.type_equal(t1, t2) for t1, t2 in zip(f['input_type'], input_type) )]
                # if len(res) == 0: # make sure the output_type is satisfied, then the input type
                #     res = tmp
        # else:
        #     raise Exception(f"unknown expected output type {output_type}")

        return res

    def getFunction(self, func_name:str)->Callable:
        # return the function object given its name
        if func_name not in self.initial_funs:
            raise Exception(f"Function {func_name} not found in DSL.")
        return self.initial_funs[func_name]

    def getTypeByName(self, name):
        # return the type object based on the type name
        res = self.basic_type_names.get(name, None)
        # res = getattr(self.type_module, name, None)
        if res == None:
            raise ValueError(f"the module {self.type_name} has no attribute {name}")
        return res
    
    def getTypeSet(self)->dict:
        # return the dictionary of all possible types
        if self.all_types is not None:
            return self.all_types
        
        type_module = Agent.load_module(self.type_name)

        self.basic_type_names = {
            name : getattr(type_module, name)
            for name in dir(type_module) if name[0].isupper() 
            and self.tc.is_parameterized(getattr(type_module, name))
            and getattr(type_module, name) is not Callable
            and getattr(type_module, name) is not TypeAlias
            or getattr(getattr(type_module, name), "__module__", None) == "builtins" # or is a built-in type like int, bool
        }

        extend_basic_types = [
            get_args(typ)[0]
            for typ in self.basic_type_names.values()
            if len(get_args(typ)) == 1
        ]

        _default_val_module = Agent.load_module(self.defval_name)

        self.basic_types = {
            ty : _default_val_module.values[ty] for ty in list(self.basic_type_names.values()) + extend_basic_types
        }

        self.callable_default_types = {
            ty : _default_val_module.values[ty] for ty in _default_val_module.values.keys() if get_origin(ty) is collections.abc.Callable
        }

        if len(self.basic_types) == 0:
            raise Exception("We have no basic types. Please check the DSL.")

        # tmp_generic_list = [ty for ty in tmp_all_types if not self.is_parameterized(ty)]
        tmp_generic_list = [
            getattr(type_module, name)
            for name in dir(type_module) if name[0].isupper() # only pick those starting with uppercase
            and get_origin(getattr(type_module, name)) is not None   # exclude typing.Any, Union, etc.
            and not self.tc.is_parameterized(getattr(type_module, name))
            and get_args(getattr(type_module, name)) is not Callable
            and getattr(type_module, name) not in (Union, Optional, runtimeContainer, Container)
        ]

        self.all_types = self._parameterize_all_generic(tmp_generic_list, self.basic_types)

        return self.all_types   
 

    def get_all_basic_types(self):
        return self.basic_types
    
    def _parameterize_all_generic(self, generic_types:list, parameterized_types_vals:dict)->dict:
        # parameterize all the generic types by all the parameterized types
        # parameterized_types_vals: a dictionary recording all the parameterized types and their default values
        # return a dict, keys: types, values: default values
        
        res = dict(parameterized_types_vals)

        # replace wrapper
        for pt, value in parameterized_types_vals.items():
            gt = get_origin(pt)
            if gt != None and len(get_args(pt)) == 1:

                # re-wrap by other generic types
                arg = get_args(pt)[0]
                val = list(value)[0]
                for rt in generic_types:
                    if gt == rt:
                        continue
                    
                    new_type = rt[arg]
                    if rt == Container:
                        new_def_val = tuple([val])
                    elif rt == FrozenSet:
                        new_def_val = frozenset([val])
                    elif rt == Tuple:
                        new_def_val = tuple([val])
                    elif rt == List:
                        new_def_val = list([val])
                    else:
                        new_def_val = rt(val)               
            
                    res[new_type] = new_def_val

        # add one wrapper
        tmp_res = dict(res)
        for gt in generic_types:
            for pt, value in tmp_res.items():
                if gt == Container:
                    new_type = runtimeContainer[pt]
                else:
                    new_type = gt[pt]

                if gt == Container:
                    new_def_val = tuple([value])
                elif gt == FrozenSet:
                    new_def_val = frozenset([value])
                elif gt == Tuple:
                    new_def_val = tuple([value])
                elif gt == List:
                    new_def_val = list([value])
                else:
                    new_def_val = gt(value)
                
                res[new_type] = new_def_val
        
        

        return res

    # def getPossibleTypes(self, var:Any)->list:
    #     # return a list of possible types that var can be
    #     possible_types = []
    #     all_types = self.getTypeSet()
    #     for t in all_types.keys():
    #         if tc.check_type(var, t):
    #             possible_types.append(t)
    #     return possible_types
    
    def getConstantSet(self)->dict:
        if self.all_constants is not None:
            return self.all_constants
        
        _constant_module = Agent.load_module(self.constant_name)

        self.all_constants = {
            name: getattr(_constant_module, name) 
            for name in dir(_constant_module) if name[0].isupper()
        }

        self.cons_output_lookup = {}

        if len(self.all_constants) == 0:
            raise Exception("We have no constants. Please check the primitive set")

        return self.all_constants
    
    def getConstantSetWithOutput(self, output_type=None,
                                 state:EvolutionState=None, thread:int=None)->dict:
        # output_type=None indicates Any output type
        if output_type == None:
            return self.all_constants
        
        output_type_key = str(output_type)

        if output_type_key in self.cons_output_lookup:
            res_constants = self.cons_output_lookup[output_type_key]
        else:
            res_constants = {
                name: val
                for name, val in self.all_constants.items() if self.tc.check_type(val, output_type)
            }
            if len(res_constants) > 0: # there exist some constants with the output type, we cache the result for future look up
                self.cons_output_lookup[output_type_key] = res_constants

        return res_constants
    
    def getFeatureSet(self)->dict:
        
        if self.basic_feature is not None:
            return self.basic_feature
        
        self.basic_feature = {}

        tmp = {name: f for name, f in self.initial_funs.items() 
                                  if len(self.getInputTypes(f)) == 1
                                   and self.input_type in self.getInputTypes(f) }
        
        for name, fun in tmp.items():

            map_list = self._get_map_list(name, fun)

            self.basic_feature.update({name: map_list})

            for map in map_list:
                if map['output_type'] not in self.feature_output_lookup:
                    self.feature_output_lookup[ map['output_type'] ] = [map]
                else:
                    self.feature_output_lookup[ map['output_type'] ].append(map)

        if len(self.basic_feature) == 0:
            raise Exception("We have no typed features. Please check DSL and the primitive set")

        return self.basic_feature
    
    def getFeaturesWithOutput(self, output_type=None)->list:
        # return a list of map tuple

        res = []

        if output_type == None: 
            for lt in self.basic_feature.values():
                res += lt
        # elif output_type in self.feature_output_lookup:
        #     res = self.feature_output_lookup[output_type]
        elif any(self.tc.type_equal(output_type, t) for t in self.feature_output_lookup.keys()):
            ty = next((t for t in self.feature_output_lookup.keys() if self.tc.type_equal(output_type, t)), None)
            res = self.feature_output_lookup[ty]
        # res_features = [
        #     f for f in self.basic_feature if output_type == None or self.tc.check_type(f, output_type)
        # ]
        return res
    
    def getCallableTerminalListWithOutputInput(self, output_type=None, input_type=None, 
                                       state:EvolutionState=None, thread:int=None)->list:
        # return a list of map tuple with exact matching output type and input type
        res = []
        # for map_list in self.all_funs.values():
        #     for map in map_list:
        #         if output_type == None or tc.type_equal(map['output_type'], output_type):
        #             res.append(map)
        if output_type == None: # return all possible map tuples
            res = self.all_maps
        elif any(self.tc.type_equal(output_type, t) for t in self.funs_output_lookup.keys()):
            ty = next((t for t in self.funs_output_lookup.keys() if self.tc.type_equal(output_type, t)), None)
            tmp = self.funs_output_lookup[ty]
            if input_type == None or len(input_type) == 0:
                res = tmp
            else:
                res = [f for f in tmp if len(f['input_type']) == len(input_type) and all( self.tc.type_equal(t1, t2) for t1, t2 in zip(f['input_type'], input_type) )]
                # if len(res) == 0: # make sure the output_type is satisfied, then the input type
                #     res = tmp
        # else:
        #     raise Exception(f"unknown expected output type {output_type}")

        #filter out the functions without callable intput types
        res = [f for f in res if all(not self.tc.is_callable(t) for t in f['input_type'])]

        return res
    
    def getSourceTypes(self):
        """return a set of types that are effective as input types"""
        if self.effSourceTypes != None:
            return set(self.effSourceTypes)

        self.effSourceTypes = set([self.input_type])

        # get all possible types
        types = self.getTypeSet().keys()

        # for each type, get the constants and features. If there are some constants and features, add the type into effSourceTypes
        for ty in types:
            if len(self.getConstantSetWithOutput(ty)) > 0 or len(self.getFeaturesWithOutput(ty)) > 0:
                self.effSourceTypes.add(ty)

        return self.effSourceTypes

    def lightClone_w_pickable(self):
        """Create a picklable light clone of this Agent.
        
        Preserves all computation results (function sets, type info, features, constants)
        but replaces non-picklable module objects with None. Module names are preserved
        as strings to allow module reloading if needed.
        """
        
        # Shallow copy to preserve all pickable attributes
        new_agent = copy.copy(self)
        
        return new_agent
    
    def _check_DSL_formatting(self):
        checklist = [(name, obj) for name, obj in self.initial_funs.items() if not name.startswith('_') 
            and callable(obj) 
            # and obj.__module__ == self.dsl_module.__name__
            ]
        
        for (name, obj) in checklist:
            # get the output type
            outtype = self.getOutputType(obj)

            # get the input arguments
            inputtypelist = self.getInputTypes(obj)

            # 1. all output types must not be Callable. 
            if self.tc.type_equal(outtype, Callable) and not self.tc.type_equal(outtype, Any):
                raise Exception(f"All output types must not be Callable. Check {name}")
            
            # 2. have up to 1 callable function, as the first argument
            if inputtypelist.count(Callable) > 1 \
            or (Callable in inputtypelist and inputtypelist.index(Callable) > 0):
                raise Exception(f"DSL functions have up to 1 callable function argument, and it must be the first argument. Check {name}")
            
            # # 3. For functions with callable argument, the rest of its arguments must be a Container argument
            # if inputtypelist.count(Callable) == 1:
            #     if not (len(inputtypelist) == 2 and Container in inputtypelist):
            #         raise Exception(f"For functions with callable argument, the rest of its arguments must be a Container argument. Check {name}")
                
            # 6. all functions should not accept parameterized types, while output generic types.
            if (Any is outtype or Container is outtype) and (Any not in inputtypelist and Container not in inputtypelist):
                raise Exception(f"all functions should not accept parameterized types, while output generic types. Check {name}")
            
