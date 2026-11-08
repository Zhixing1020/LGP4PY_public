from functools import wraps
from collections.abc import Container as runtimeContainer
import collections.abc
from typing import get_origin, get_args, get_type_hints
from typing import (
    List,
    Union,
    Tuple,
    Any,
    Container,
    Callable,
    FrozenSet,
    Iterable,
    Callable,
    Optional
)
import inspect

# Global cache for type metadata to avoid repeated get_origin/get_args calls
_type_metadata_cache = {}

class TypeChecker:

    @staticmethod
    def _get_type_metadata(typ):
        """Helper to get cached origin and args for a type."""
        typ_key = str(typ)
        if typ_key not in _type_metadata_cache:
            _type_metadata_cache[typ_key] = (get_origin(typ), get_args(typ))
        return _type_metadata_cache[typ_key]

    @staticmethod
    def check_type(value, typ) -> bool:
        """Recursively check if value (a variable) conforms to type hint typ."""
        # value: the variable (an instance) to check
        # typ: the type hint to check against

        if typ is Any:
            return True
        
        if typ == Callable:
            return callable(value)
        
        origin, args = TypeChecker._get_type_metadata(typ)  # Use cached metadata

        # Case: Plain types or aliases (like int, bool, your Integer alias)
        if origin is None:
            return isinstance(value, typ)

        if origin is collections.abc.Callable:
            if not callable(value):
                return False
            if TypeChecker.is_lambda(value):
                return True  # we cannot check the input output types of lambda functions, so we simply allow all lambda functions to match any Callable type
            input_types = args[0]   # list of input types
            return_type = args[1]   # return type
            fun_input_types, fun_output_type = TypeChecker.get_func_types(value)

            input_match = len(fun_input_types) == 0 or \
                len(fun_input_types) == len(input_types) \
                and all(TypeChecker.type_equal(fit, it) for fit, it in zip(fun_input_types, input_types))

            output_match = return_type is None or TypeChecker.type_equal(fun_output_type, return_type)

            return input_match and output_match

        # Case: Tuple[...] (fixed-length or variable-length)
        if origin is tuple:
            if not isinstance(value, tuple):
                return False
            elif len(args) == 0: # typ does not specify a sub type argument, so can be any.
                return True
            if len(value) == len(args):
                return all(TypeChecker.check_type(v, t) for v, t in zip(value, args))
            if len(args) == 2 and args[1] is Ellipsis or len(args) == 1:  # Tuple[T, ...]
                return all(TypeChecker.check_type(v, args[0]) for v in value)
            

        # Case: FrozenSet
        if origin is frozenset:
            if len(args) == 0: # typ does not specify a sub type argument, so can be any.
                return isinstance(value, frozenset)
            else:
                return isinstance(value, frozenset) and all(TypeChecker.check_type(v, args[0]) for v in value)

        # Case: Container (generic catch-all)
        if origin in (list, set, dict):
            if len(args) == 0: # typ does not specify a sub type argument, so can be any.
                return isinstance(value, origin) 
            else:
                return isinstance(value, origin) \
            and all(TypeChecker.check_type(v, args[0]) for v in (value if origin != dict else value.values()))

        # Case: Union
        if origin is Union:
            return any(TypeChecker.check_type(value, arg) for arg in args)
        
        # Case: Container of Container
        # here is tricky, in runtime, we need to detect "collections.abc.Container", rather than the type hint, typing.Container
        if origin is runtimeContainer:
            if len(args) == 0: # typ does not specify a sub type argument, so can be any.
                return isinstance(value, runtimeContainer)
            else:
                return isinstance(value, runtimeContainer) and all(TypeChecker.check_type(v, args[0]) for v in value)

        return False

    @staticmethod
    def type_equal(typ1, typ2) -> bool:
        """Recursively check the equivalence of two types."""
    
        if typ1 is Any or typ2 is Any:
            return True
        
        if typ1 == typ2:
            return True

        origin1, args1 = TypeChecker._get_type_metadata(typ1)  # Use cached metadata
        origin2, args2 = TypeChecker._get_type_metadata(typ2)  # Use cached metadata

        if origin1 == None and origin2 == None: # both typ1 and typ2 are basic types
            return typ1 == typ2

        if origin1 == origin2 \
            or (origin2 != None and origin1 == runtimeContainer and issubclass(origin2, runtimeContainer)) \
            or (origin1 != None and origin2 == runtimeContainer and issubclass(origin1, runtimeContainer)) \
            or (origin1 == runtimeContainer and origin2 == runtimeContainer):
            if args1 == () or args2 == (): # one of the two types is generic type
                return True
            else:
                return len(args1) == len(args2) \
                        and all(TypeChecker.type_equal(t1, t2) for t1, t2 in zip(args1, args2))
            
        if origin1 in (Union, Optional):
            return any(TypeChecker.type_equal(t1, typ2) for t1 in args1)
        
        if origin2 in (Union, Optional):
            return any(TypeChecker.type_equal(typ1, t2) for t2 in args2)

        # type_equal cannot accept union, which might lead to ambiguous  
        # if origin1 is Union:
        #     return any(check_type(arg, typ2) for arg in args1)
        # if origin2 is Union:
        #     return any(check_type(arg, typ1) for arg in args2)
        
        return False

    @staticmethod
    def is_parameterized(type)->bool:
        """detect if the type is parameterized"""
        if type in (Any, Union, Optional): # exclude the special form objects, simply enumerating all possible special forms
            return False
        
        if type is Callable: # we do not consider parameterizing Callable type
            return True

        origin = get_origin(type)
        args = get_args(type)

        if origin == runtimeContainer:
            return False

        if origin is None and args == ():  # not a generic
            return True
        elif origin is None and args != ():  # weird corner case
            raise ValueError(f"unknown type: {type}")
        elif args != (): # with origin and args, need further check
            return all(TypeChecker.is_parameterized(arg) for arg in args)
        else:
            return False
        
    @staticmethod
    def is_parameterized_callable(typ)->bool:
        """detect if the type is a parameterized Callable"""
        origin, args = TypeChecker._get_type_metadata(typ)
        return origin is collections.abc.Callable and args != ()
    
    @staticmethod
    def is_callable(typ) -> bool:
        """detect if the type is a Callable (parameterized or not)"""
        origin, _ = TypeChecker._get_type_metadata(typ)
        return typ == Callable or origin is collections.abc.Callable
        
    # def get_nested_origin_args(paramtype, temptype):
    #     # get the nested origin type and args type of paramtype based on the given temptype

    #     if not is_parameterized(paramtype) or is_parameterized(temptype):
    #         raise Exception(f"{paramtype} must be parameterized or {temptype} must be generic")
        
    #     if temptype is Any:
    #         return None, paramtype
        
    #     if not type_equal(paramtype, temptype):
    #         raise Exception(f"{paramtype} and {temptype} are not the same type")
        
    #     origin_p = get_origin(paramtype)
    #     args_p = get_args(paramtype)
    #     origin_t = get_origin(temptype)
    #     args_t = get_args(temptype)

    #     if len(args_t) == 0 and len(args_p) == 1:
    #         return origin_p, args_p[0]

    #     if origin_t is not None and len(args_t) == 1 and len(args_p) == 1:
    #         nest_origin, nest_arg = get_nested_origin_args(args_p[0], args_t[0])
    #         return origin_p[nest_origin], nest_arg
        
    #     raise Exception(f"unknown situations in get_nested_origin_args with {paramtype} and {temptype}")
        
    @staticmethod
    def get_func_types(func):
        sig = inspect.signature(func)
        
        input_types = [
            param.annotation
            for param in sig.parameters.values()
            if param.annotation is not inspect.Parameter.empty  # skip unannotated
        ]
        
        return_type = sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else None
        
        
        return input_types, return_type
    
    @staticmethod
    def get_callable_types_param(typ):
        args = TypeChecker._get_type_metadata(typ)[1]
        input_types = args[0]   # list of input types
        return_type = args[1]   # return type
        return input_types, return_type
    
    @staticmethod
    def type_in_list(type:Any, typelist:list) -> bool:
        return any( TypeChecker.type_equal(type, ty) for ty in typelist )

    @staticmethod
    def is_lambda(func) -> bool:
        return (
            callable(func) and
            inspect.isfunction(func) and      # rules out builtins, classes
            func.__name__ == "<lambda>"
        )

    @staticmethod
    def enforce_types(func):
        # Decorator to enforce type hints at runtime
        hints = get_type_hints(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = list(args)
            kw_items = kwargs.items()

            # Check positional + keyword args
            for (name, typ), arg in zip(hints.items(), bound_args):
                if name == "return":
                    continue
                if not TypeChecker.check_type(arg, typ):
                    raise TypeError(f"{func.__name__}: Argument '{name}' must be {typ}, got {arg}")

            for name, arg in kw_items:
                if name in hints and not TypeChecker.check_type(arg, hints[name]):
                    raise TypeError(f"{func.__name__}: Argument '{name}' must be {hints[name]}, got {arg}")

            # Call original
            result = func(*args, **kwargs)

            # Check return type
            if "return" in hints and not TypeChecker.check_type(result, hints["return"]):
                raise TypeError(f"{func.__name__}: Return must be {hints['return']}, got {result}")

            return result

        return wrapper