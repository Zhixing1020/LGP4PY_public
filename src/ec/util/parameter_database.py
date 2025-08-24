
from pathlib import Path
from src.ec.util.parameter import Parameter

class ParameterDatabase:
    def __init__(self, filename):
        self.params = {}
        self._load(filename)

    def _load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip comments and whitespace
                line = line.split("#")[0].strip()
                if not line:
                    continue
                if "=" not in line:
                    raise ValueError(f"Invalid line in parameter file: {line}")
                key, value = map(str.strip, line.split("=", 1))
                self.params[key] = value

    def getParamValue(self, param:Parameter, default:Parameter=None):
        if self.exists(str(param)):
            return self.params.get(str(param))
        else:
            if self.exists(str(default)):
                return self.params.get(str(default))
            else:
                return None

    def getString(self, param:Parameter, default:Parameter=None)->str:
        res = self.getParamValue(param, default)
        return str(res) if res is not None else None
    
    def getBoolean(self, param:Parameter, default_param:Parameter=None, default_val:bool=False)->bool:
        val = self.getParamValue(param, default_param)
        return val.lower() == 'true' if val is not None else default_val

    def getInt(self, param:Parameter, default_param:Parameter=None)->int:
        val = self.getParamValue(param, default_param)

        if val is not None:
            return int(val) 
        else:
            raise SystemExit(f"Fatal error: cannot find the parameter either {param} or {default_param}")

    def getIntWithDefault(self, param:Parameter, default_param:Parameter, default_val)->int:

        val = self.getParamValue(param, default_param)
        return int(val) if val is not None else default_val
    
    def getDoubleWithDefault(self, param:Parameter, default_param:Parameter, default_val)->float:

        val = self.getParamValue(param, default_param)
        return float(val) if val is not None else default_val

    def getInstanceForParameter(self, param:Parameter, default_param:Parameter, cls_type):
        
        class_name = self.getParamValue(param, default_param)
        if class_name is None:
            raise ValueError(f"Parameter '{param}' is not defined.")
        
        components = class_name.split('.')
        
        module = __import__(".".join(components[:-1]), fromlist=[components[-1]])
        return getattr(module, components[-1])()

    def exists(self, param:str, default:str=None)->bool:
        if isinstance(param, Parameter):
            param = str(param)
        elif not isinstance(param, str):
            raise ValueError(f"Parameter '{param}' is an unknow datatype {type(param)}")
        if default is not None and isinstance(default, Parameter):
            default = str(default)

        return param in self.params or default in self.params
    
    def getFile(self, parameter:Parameter, defaultParameter:Parameter=None)->Path:
        """
        Gets a file from either the specified parameter or a default parameter.
        Maintains the same behavior as the Java version.
        """
        # self.printGotten(parameter, defaultParameter, False)
        if self.exists(parameter):
            return self._getFile(parameter)
        elif defaultParameter is not None:
            return self._getFile(defaultParameter)
        return None

    def _getFile(self, parameter:Parameter)->Path:
        """
        Internal method to get a file from a single parameter.
        Implements the same logic as the Java version:
        - Handles paths starting with "$"
        - Resolves absolute/relative paths
        - Marks parameter as used
        """
        if not self.exists(parameter):
            return None
        
        p = self.getString(parameter)
        if p is None:
            return None
        
        # Constants that would be defined at class level
        C_HERE = "$"
        C_CLASS = "@"
        
        if p.startswith(C_HERE):
            return Path(p[len(C_HERE):])
        elif p.startswith(C_CLASS):
            return None  # Can't start with @
        else:
            path = Path(p)
            if path.is_absolute():
                return path
            else:
                # directoryFor would be a method that returns the base directory
                base_dir = self.find_base_dir(path.parent, p) 
                return (Path(base_dir) / p) if base_dir else None

    def find_base_dir(start_path: Path, file_name: str) -> Path:
        """Search upward for a directory containing the marker file"""
        for parent in start_path.parents:
            if (parent / file_name).exists():
                return parent
        return start_path  # Fallback to original directory

if __name__ == "__main__":
    db = ParameterDatabase('D:\\zhixing\\科研\\LGP4PY\\LGP4PY\\tasks\\Symbreg\\parameters\\simpleLGP_SRMT.params')
    param = Parameter.Parameter("stat").push("child").push("0")
    test = db.getParamValue(str(param))  # returns value
    print(test)
    print(db)
