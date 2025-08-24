# Translation of GPSymbolicRegressionMultiTarget from Java to Python
# Note: This translation keeps method and variable names identical to the Java version
# Supporting classes like EvolutionState, Individual, GPIndividual, DoubleData, etc., should be defined elsewhere.

import os
import math
import numpy as np
from typing import List

from src.ec import *
from src.ec.util import Parameter,ParameterDatabase
from tasks.problem import Problem
from tasks.supervisedproblem import SupervisedProblem
from tasks.symbreg.individual.lgpindividual4SR import LGPIndividual4SR

from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

class GPSymbolicRegression(Problem, SupervisedProblem):

    PROBLEM_P = "SymbolicRegression"
    LOCATION_P = "location"
    DATA_NAME_P = "dataname"
    FITNESS_P = "fitness"
    NORMALIZE_P = "normalize"
    KFOLDINDEX_P = "Kfold_index"
    KFOLDNUM_P = "Kfold_num"
    TARGETNUM_P = "target_num"
    TARGETS_P = "targets"
    VALIDATION_P = "do-validation"

    def __init__(self, loca:str=None, datan:str=None, fitn:str=None, istraining:bool=None, parameters:ParameterDatabase=None):

        if parameters is None:
            return
        
        self.location = ""
        self.dataname = ""
        self.fitness = ""
        self.istraining = False
        self.doValidation = False
        self.normalized = False

        self.foldnum = 0
        self.foldindex = 0

        self.datadim = 0
        self.outputnum = 0
        self.outputdim = 0

        self.target_num = 0
        self.targets = []

        self.datanum = 0
        self.validatenum = 0

        self.data = []
        self.data_output = []
        self.normdata = []

        self.norm_mean = []
        self.norm_std = []
        self.out_mean = []
        self.out_std = []
        self.data_max = []
        self.data_min = []

        self.validate_data = []
        self.validate_data_output = []

        self.X = []
        self.X_index = 0

        base = Parameter('eval.problem')
        default = Parameter(self.PROBLEM_P)

        self.foldindex = parameters.getIntWithDefault(base.push(self.KFOLDINDEX_P), default.push(self.KFOLDINDEX_P), 0)
        self.foldnum = parameters.getIntWithDefault(base.push(self.KFOLDNUM_P), default.push(self.KFOLDNUM_P), 1)
        if self.foldnum <= 0:
            raise ValueError("A symbolic regression problem needs a positive K-fold number.")

        self.target_num = parameters.getIntWithDefault(base.push(self.TARGETNUM_P), default.push(self.TARGETNUM_P), 1)
        if self.target_num <= 0:
            raise ValueError("A symbolic regression problem must have at least one target.")

        self.targets = []
        for t in range(self.target_num):
            b = base.push(self.TARGETS_P).push(str(t))
            tar = parameters.getIntWithDefault(b, None, 0)
            if tar < 0:
                raise ValueError("Target index must be >= 0.")
            self.targets.append(tar)

        self.normalized = parameters.getBoolean(base.push(self.NORMALIZE_P), default.push(self.NORMALIZE_P))
        self.doValidation = parameters.getBoolean(self.VALIDATION_P, False)

        self.setProblem(None, loca, datan, fitn, istraining)

    def setup(self, state, base):
        super().setup(state, base)

        def_param = Parameter(self.PROBLEM_P)

        if not isinstance(self.input, GPData):
            state.output.fatal(f"data class must subclass from GPData: {base.push(self.P_DATA)} or {def_param.push(self.P_DATA)}")

        self.location = state.parameters.getString(base.push(self.LOCATION_P), def_param.push(self.LOCATION_P))
        if self.location is None or self.location == "":
            state.output.fatal(f"Empty location for the data: {base.push(self.LOCATION_P)} or {def_param.push(self.LOCATION_P)}")

        self.dataname = state.parameters.getString(base.push(self.DATA_NAME_P), def_param.push(self.DATA_NAME_P))
        if self.dataname is None or self.dataname == "":
            state.output.fatal(f"Empty name for the data: {base.push(self.DATA_NAME_P)} or {def_param.push(self.DATA_NAME_P)}")

        self.fitness = state.parameters.getString(base.push(self.FITNESS_P), def_param.push(self.FITNESS_P))
        self.normalized = state.parameters.getBoolean(base.push(self.NORMALIZE_P), def_param.push(self.NORMALIZE_P))
        self.doValidation = state.parameters.getBoolean(base.push(self.VALIDATION_P), def_param.push(self.NORMALIZE_P))

        self.foldindex = state.parameters.getIntWithDefault(base.push(self.KFOLDINDEX_P), def_param.push(self.KFOLDINDEX_P), 0)
        self.foldnum = state.parameters.getIntWithDefault(base.push(self.KFOLDNUM_P), def_param.push(self.KFOLDNUM_P), 1)
        if self.foldnum <= 0:
            raise ValueError("A symbolic regression problem needs a positive K-fold number.")

        self.target_num = state.parameters.getIntWithDefault(base.push(self.TARGETNUM_P), def_param.push(self.TARGETNUM_P), 1)
        if self.target_num <= 0:
            state.output.fatal(f"A symbolic regression problem must have at least one target: {base.push(self.TARGETNUM_P)} or {def_param.push(self.TARGETNUM_P)}")

        self.targets = []
        for t in range(self.target_num):
            b = base.push(self.TARGETS_P).push(str(t))
            tar = state.parameters.getIntWithDefault(b, None, 0)
            if tar < 0:
                raise ValueError("Target index must be >= 0.")
            self.targets.append(tar)

        self.setProblem(state, self.location, self.dataname, self.fitness, True)

    def setProblem(self, state:EvolutionState, loca:str, datan:str, fitn:str, istraining:bool):
        self.location = loca
        self.dataname = datan
        self.istraining = istraining

        sep = os.sep  # Use the OS-specific path separator, like '\\' for Windows or '/' for Unix-like systems
        if not self.location.endswith(sep):
            self.location += sep
        dataname_address = f"{self.dataname}{sep}" if not self.dataname.endswith(sep) else ""

        suffix = "train" if self.istraining else "test"
        filename_X = f"{self.location}{dataname_address}{self.dataname}_X_{suffix}_F{self.foldindex}.txt"
        filename_y = f"{self.location}{dataname_address}{self.dataname}_y_{suffix}_F{self.foldindex}.txt"

        print(f"evaluating on X: {filename_X}, Y: {filename_y}")

        if not os.path.exists(filename_X):
            raise FileNotFoundError(f"The dataset {filename_X} does not exist")

        if self.istraining and not os.path.exists(filename_y):
            raise FileNotFoundError(f"The dataset {filename_y} does not exist")

        self.read_X_file(filename_X)
        self.read_y_file(filename_y)

        self.fitness = fitn
        if self.fitness not in ["RMSE", "MSE", "R2", "RSE", "WRSE", "ERR"]:
            raise ValueError(f"{self.fitness} must be one of: RMSE, MSE, R2, RSE, WRSE, ERR")

        if self.normalized:
            self.normalizedataBasedTraining()

        if self.istraining and state is not None and self.doValidation:
            self.split_validation(state)

    
    def read_X_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header = lines[0].strip().split()
        self.datanum, self.datadim = int(header[0]), int(header[1])
        self.data = []
        self.data_max = [-1e7] * self.datadim
        self.data_min = [1e7] * self.datadim

        for line in lines[1:self.datanum+1]:
            instance = list(map(float, line.strip().split()))
            for i in range(self.datadim):
                self.data_max[i] = max(self.data_max[i], instance[i])
                self.data_min[i] = min(self.data_min[i], instance[i])
            self.data.append(instance)

        self.data = np.array(self.data, dtype=float)

    def read_y_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header = lines[0].strip().split()
        self.outputnum, self.outputdim = int(header[0]), int(header[1])
        self.data_output = [list(map(float, line.strip().split())) for line in lines[1:self.outputnum+1]]

        self.data_output = np.array(self.data_output, dtype=float)

    def split_validation(self, state: EvolutionState):
        self.validate_data = []
        self.validate_data_output = []
        self.validatenum = math.ceil(0.1 * self.datanum)

        usingdata = self.normdata if self.normalized else self.data
        usingoutput = self.data_output

        # Randomly sample validation indices (without replacement to avoid duplicates)
        indices = [state.random[0].randint(0, len(usingdata) - 1) for _ in range(self.validatenum)]
        
        # Convert to numpy array for fancy indexing
        indices = np.array(indices)

        # Extract validation set
        self.validate_data = usingdata[indices]
        self.validate_data_output = usingoutput[indices]

        # Remove selected rows from training set
        usingdata = np.delete(usingdata, indices, axis=0)
        usingoutput = np.delete(usingoutput, indices, axis=0)

        # Update attributes
        self.data = usingdata
        self.data_output = usingoutput
        self.datanum = usingdata.shape[0]
        self.outputnum = usingoutput.shape[0]

    def normalizedataBasedTraining(self):
        if self.data is None or len(self.data) == 0:
            return

        # num = len(self.data)
        # dim = len(self.data[0])
        # self.norm_mean = [sum(col) / num for col in zip(*self.data)]
        # self.norm_std = [math.sqrt(sum((x - mean) ** 2 for x in col) / num)
        #                  for col, mean in zip(zip(*self.data), self.norm_mean)]

        # self.normdata = [
        #     [(x - mean) / std if std > 0 else 0. for x, mean, std in zip(row, self.norm_mean, self.norm_std)]
        #     for row in self.data
        # ]

        # num_out = len(self.data_output)
        # dim_out = len(self.data_output[0])
        # self.out_mean = [sum(col) / num_out for col in zip(*self.data_output)]
        # self.out_std = [math.sqrt(sum((x - mean) ** 2 for x in col) / num_out)
        #                 for col, mean in zip(zip(*self.data_output), self.out_mean)]
        

        self.norm_mean = self.data.mean(axis=0)             
        self.norm_std = self.data.std(axis=0, ddof=0)      

        safe_std = np.where(self.norm_std > 0, self.norm_std, 1.0)
        self.normdata = ((self.data - self.norm_mean) / safe_std)

        self.out_mean = self.data_output.mean(axis=0)
        self.out_std = self.data_output.std(axis=0, ddof=0)

    def getRMSE(self, real, predict):
        # res = math.sqrt(sum((r - p) ** 2 for r, p in zip(real, predict)) / len(real))
        res = root_mean_squared_error(real, predict)
        return 1e6 if math.isinf(res) or math.isnan(res) else res

    def getMSE(self, real, predict):
        # res = sum((r - p) ** 2 for r, p in zip(real, predict)) / len(real)
        res = mean_squared_error(real, predict)
        return 1e6 if math.isinf(res) or math.isnan(res) else res

    def getR2(self, real, predict):
        # avg = sum(real) / len(real)
        # var = sum((r - avg) ** 2 for r in real) / len(real)
        # mse = self.getMSE(real, predict)
        res =  r2_score(real, predict)
        return 1e6 if math.isinf(res) or math.isnan(res) else res
        # return -1 * (1 - mse / var) if not math.isinf(mse) and not math.isnan(mse) else 1e6

    def getRSE(self, real, predict):
        # avg = sum(real) / len(real)
        # var = sum((r - avg) ** 2 for r in real) / len(real)
        # mse = self.getMSE(real, predict)
        # return mse / var if not math.isinf(mse) and not math.isnan(mse) else 1e6
        res = 1. - self.getR2(real, predict)
        return 1e6 if math.isinf(res) or math.isnan(res) else res

    def getWRSE(self, real, predict, k):
        avg = sum(real) / len(real)
        var = sum((r - avg) ** 2 for r in real)
        normvar = sum((r - avg) ** 2 ** k for r in real)
        wmse = sum((r - p) ** 2 * ((r - avg) ** 2) ** k for r, p in zip(real, predict))
        return math.sqrt(wmse / (var * normvar))

    def getVar(self, data):
        avg = sum(data) / len(data)
        return sum((x - avg) ** 2 for x in data) / len(data)

    def getError(self, real, predict):
        res = sum(1 for r, p in zip(real, predict) if round(p) != r)
        return res / len(real)

    def getDatanum(self): return self.datanum
    def getDatadim(self): return self.datadim
    def getOutputnum(self): return self.outputnum
    def getOutputdim(self): return self.outputdim
    def getTargets(self): return self.targets
    def getTargetNum(self): return self.target_num
    def getDataMax(self): return self.data_max
    def getDataMin(self): return self.data_min
    def getData(self): return self.normdata if self.normalized else self.data
    def getDataOutput(self): return self.data_output
    def getX(self): return self.X
    def getX_index(self): return self.X_index
    def setX_index(self, ind): self.X_index = ind
    def isnormalized(self): return self.normalized
    def istraining(self): return self.istraining
    def setFoldIndex(self, ind, istraining):
        self.foldindex = ind
        self.setProblem(None, self.location, self.dataname, self.fitness, istraining)
    def getFoldNum(self): return self.foldnum

    def evaluate(self, state:EvolutionState, ind:LGPIndividual4SR, subpopulation:int, threadnum:int):
        if not ind.evaluated:
            if self.data is None or self.data_output is None:
                raise RuntimeError("we have an empty data source")

            # hits = 0
            result = 0
            normwrap = 0
            real = self.data_output

            # predict = []
            # for y in range(self.datanum):
            #     tmp = GPData()
            #     self.X = [self.normdata[y][d] if self.normalized else self.data[y][d] for d in range(self.datadim)]
            #     self.X_index = y
                
            #     ind.setDataIndex(y)
            #     pred = ind.execute(state, threadnum, tmp, ind, self, False)
            #     predict.append(pred)

            tmp = GPData()
            tmp.to_vectorize = True
            tmp.values = np.zeros((self.datanum, 1))
            self.X = self.normdata if self.normalized else self.data
            ind.preExecution(state, threadnum)
            predict = ind.execute(state, threadnum, tmp, ind, self, False)
            
            predict = np.concatenate(predict, axis=1)

            #convert "predict" as a list of 2d ndarray
            if isinstance(predict, list) and isinstance(predict[0], list):
                predict = np.array(predict)

            mask = np.isnan(predict) | np.isinf(predict)
            predict[mask] = 1e6

            if self.normalized:
                indices = np.array([self.targets[od] for od in range(self.target_num)])
                predict = predict * self.out_std[indices] + self.out_mean[indices]

            if ind.IsWrap():
                indices = np.array([self.targets[od] for od in range(self.target_num)])
                real_care = real[:, indices]

                predict = ind.wrapper(predict, real_care, state, threadnum, self)
                # normwrap = ind.getWeightNorm()

            for od in range(self.target_num):
                real_d = real[:, self.targets[od]] # convert the 2D array into 1D
                predict_d = predict[:, od]

                if self.fitness == "RMSE":
                    result += self.getRMSE(real_d, predict_d) / self.target_num
                elif self.fitness == "MSE":
                    result += self.getMSE(real_d, predict_d) / self.target_num
                elif self.fitness == "R2":
                    result += self.getR2(real_d, predict_d) / self.target_num
                elif self.fitness == "RSE":
                    result += self.getRSE(real_d, predict_d) / self.target_num
                elif self.fitness == "WRSE":
                    result += self.getWRSE(real_d, predict_d, 1) / self.target_num
                elif self.fitness == "ERR":
                    result += self.getError(real_d, predict_d) / self.target_num
                else:
                    raise ValueError("unknown fitness objective " + self.fitness)

            validate_res = self.validationevaluation(state, ind, subpopulation, threadnum)
            fitness_val = result + normwrap + 0.1 * validate_res
            # f = ind.fitness
            ind.fitness.setFitness(state, fitness_val)
            ind.evaluated = True

    def validationevaluation(self, state:EvolutionState, ind:LGPIndividual4SR, subpopulation:int, threadnum:int):
        if not self.doValidation:
            return 0.
        real = self.validate_data_output
        # predict = []
        # for y in range(self.validatenum):
        #     tmp = GPData()
        #     self.X = [self.validate_data[y][d] for d in range(self.datadim)]
        #     self.X_index = y
        #     pred = ind.execute(state, threadnum, tmp, ind, self, True)
        #     predict.append(pred)

        tmp = GPData()
        tmp.to_vectorize = True
        tmp.values = np.zeros((self.datanum, 1))
        self.X = self.validate_data
        # ind.preExecution(state, threadnum)
        predict = ind.execute(state, threadnum, tmp, ind, self, False)
        
        predict = np.concatenate(predict, axis=1)

        #convert "predict" as a list of 2d ndarray
        if isinstance(predict, list) and isinstance(predict[0], list):
            predict = np.array(predict)

        mask = np.isnan(predict) | np.isinf(predict)
        predict[mask] = 1e6

        result = 0
        for od in range(self.target_num):
            # real_d = [real[y][self.targets[od]] for y in range(self.validatenum)]
            # predict_d = [predict[y][od] for y in range(self.validatenum)]
            real_d = real[:, self.targets[od]] # convert the 2D array into 1D
            predict_d = predict[:, od]
            
            if self.fitness == "RMSE":
                result += self.getRMSE(real_d, predict_d) / self.target_num
            elif self.fitness == "MSE":
                result += self.getMSE(real_d, predict_d) / self.target_num
            elif self.fitness == "R2":
                result += self.getR2(real_d, predict_d) / self.target_num
            elif self.fitness == "RSE":
                result += self.getRSE(real_d, predict_d) / self.target_num
            else:
                raise ValueError("unknown fitness objective " + self.fitness)

        return result

    def simpleevaluate(self, ind:LGPIndividual4SR):
        if not ind.evaluated:
            if self.data is None or self.data_output is None:
                raise RuntimeError("we have an empty data source")

            real = self.data_output
            # predict = []

            # for y in range(self.datanum):
            #     tmp = GPData()
            #     self.X = [self.normdata[y][d] if self.normalized else self.data[y][d] for d in range(self.datadim)]
            #     self.X_index = y
            #     ind.setDataIndex(y)                    
            #     pred = ind.execute(None, 0, tmp, ind, self, True)
            #     predict.append(pred)

            tmp = GPData()
            tmp.to_vectorize = True
            tmp.values = np.zeros((self.datanum, 1))
            self.X = self.normdata if self.normalized else self.data
            # ind.preExecution(None, 0)
            predict = ind.execute(None, 0, tmp, ind, self, False)
            
            predict = np.concatenate(predict, axis=1)

            #convert "predict" as a list of 2d ndarray
            if isinstance(predict, list) and isinstance(predict[0], list):
                predict = np.array(predict)

            mask = np.isnan(predict) | np.isinf(predict)
            predict[mask] = 1e6

            result = 0
            for od in range(self.target_num):
                real_d = real[:, self.targets[od]] # convert the 2D array into 1D
                predict_d = predict[:, od]
                if self.fitness == "RMSE":
                    result += self.getRMSE(real_d, predict_d) / self.target_num
                elif self.fitness == "MSE":
                    result += self.getMSE(real_d, predict_d) / self.target_num
                elif self.fitness == "R2":
                    result += self.getR2(real_d, predict_d) / self.target_num
                elif self.fitness == "RSE":
                    result += self.getRSE(real_d, predict_d) / self.target_num
                elif self.fitness == "WRSE":
                    result += self.getWRSE(real_d, predict_d, 1) / self.target_num
                elif self.fitness == "ERR":
                    result += self.getError(real_d, predict_d) / self.target_num
                else:
                    raise ValueError("unknown fitness objective " + self.fitness)

            if ind.fitness is None:
                ind.fitness = Fitness()
            f = ind.fitness
            ind.fitness.setFitness(None, result)
            ind.evaluated = True

    def quickevaluate(self, ind:LGPIndividual4SR, X:np.ndarray=None):

        if X is not None:
            self.setX(X)

        if self.data is None:
            raise RuntimeError("we have an empty data source")

        # predict = []
        # for y in range(self.datanum):
        #     tmp = GPData()
        #     self.X = [self.normdata[y][d] if self.normalized else self.data[y][d] for d in range(self.datadim)]
        #     self.X_index = y
        #     ind.setDataIndex(y)                
        #     pred = ind.execute(None, 0, tmp, ind, self, True)
        #     predict.append(pred)

        tmp = GPData()
        tmp.to_vectorize = True
        tmp.values = np.zeros((self.datanum, 1))
        self.X = self.normdata if self.normalized else self.data
        # ind.preExecution(None, 0)
        predict = ind.execute(None, 0, tmp, ind, self, True)
        
        predict = np.concatenate(predict, axis=1)

        #convert "predict" as a list of 2d ndarray
        if isinstance(predict, list) and isinstance(predict[0], list):
            predict = np.array(predict)

        mask = np.isnan(predict) | np.isinf(predict)
        predict[mask] = 1e6

        # res = []
        # for y in range(self.datanum):
        #     tmp = []
        #     for od in range(self.target_num):
        #         di = self.targets[od]
        #         if self.normalized:
        #             predict[y][od] = predict[y][od] * self.out_std[di] + self.out_mean[di]
        #         if math.isinf(predict[y][od]) or math.isnan(predict[y][od]):
        #             predict[y][od] = 1e6
        #         tmp.append(predict[y][od])
        #     res.append(tmp)
        return predict