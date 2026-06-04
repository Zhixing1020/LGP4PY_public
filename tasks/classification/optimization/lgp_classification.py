import os
import math
from typing import override
import numpy as np

from src.ec import *
from src.ec.util import Parameter, ParameterDatabase
from tasks.problem import Problem
from tasks.supervisedproblem import SupervisedProblem
from tasks.classification.individual.lgpindividual4Class import LGPIndividual4Class


class LGPClassificationProblem(Problem, SupervisedProblem):
    PROBLEM_P = "ClassificationProblem"
    LOCATION_P = "location"
    DATA_NAME_P = "dataname"
    FITNESS_P = "fitness"
    NORMALIZE_P = "normalize"
    KFOLDINDEX_P = "Kfold_index"
    KFOLDNUM_P = "Kfold_num"
    TARGETNUM_P = "target_num"
    TARGETS_P = "targets"
    VALIDATION_P = "do-validation"

    CLASSNUM_P = "class_num"
    CLASS_P = "class"

    def __init__(self, loca: str = None, datan: str = None, fitn: str = None, istraining: bool = None, parameters: ParameterDatabase = None):
        Problem.__init__(self)

        self.location = ""
        self.dataname = ""
        self.fitness = ""
        self._istraining = False
        self.doValidation = False
        self.normalized = False

        self.foldnum = 0
        self.foldindex = 0

        self.datadim = 0
        self.outputnum = 0
        self.outputdim = 0

        self.target_num = 0
        self.targets: list[int] = []

        self.datanum = 0
        self.validatenum = 0

        self.data: list[list[float]] = []
        self.data_output: list[list[float]] = []
        self.normdata: list[list[float]] = []
        self.norm_mean: list[float] = []
        self.norm_std: list[float] = []
        self.out_mean: list[float] = []
        self.out_std: list[float] = []
        self.data_max: list[float] = []
        self.data_min: list[float] = []

        self.validate_data: list[list[float]] = []
        self.validate_data_output: list[list[float]] = []

        self.X: list[float] = []
        self.X_index = 0

        self.class_num = 2
        self.class_labels: list[float] = []

        if parameters is None:
            return

        base = Parameter("eval.problem")
        default = Parameter(self.PROBLEM_P)

        self.foldindex = parameters.getIntWithDefault(base.push(self.KFOLDINDEX_P), default.push(self.KFOLDINDEX_P), 0)
        self.foldnum = parameters.getInt(base.push(self.KFOLDNUM_P), default.push(self.KFOLDNUM_P))
        if self.foldnum < 0:
            raise ValueError("A multi-target symbolic regression problem need to set a K-fold number.")

        self.target_num = parameters.getIntWithDefault(base.push(self.TARGETNUM_P), default.push(self.TARGETNUM_P), 1)
        if self.target_num <= 0:
            raise ValueError("A multi-target symbolic regression problem at least has one target.")

        self.targets = []
        for t in range(self.target_num):
            b = base.push(self.TARGETS_P).push(str(t))
            tar = parameters.getIntWithDefault(b, None, 0)
            if tar < 0:
                raise ValueError("target index must be >= 0.")
            self.targets.append(tar)

        self.normalized = parameters.getBoolean(base.push(self.NORMALIZE_P), default.push(self.NORMALIZE_P), False)
        self.doValidation = parameters.getBoolean(base.push(self.VALIDATION_P), default.push(self.VALIDATION_P), False)

        self.class_num = parameters.getInt(base.push(self.CLASSNUM_P), default.push(self.CLASSNUM_P))
        if self.class_num < 2:
            raise ValueError("the number of classes must be >= 2")

        self.class_labels = []
        for c in range(self.class_num):
            cl = parameters.getDoubleWithDefault(
                base.push(self.CLASS_P).push(str(c)),
                default.push(self.CLASS_P).push(str(c)),
                0.0,
            )
            self.class_labels.append(cl)

        self.setProblem(None, loca, datan, fitn, istraining)

    def _build_data_paths(self, loca: str, datan: str, istraining: bool):
        location = loca if loca is not None else ""
        dataname = datan if datan is not None else ""

        sep = os.sep
        if location and not location.endswith(sep):
            location += sep

        dataname_address = ""
        if dataname:
            if location and dataname in location:
                dataname_address = ""
            else:
                dataname_address = dataname + sep

        suffix = "train" if istraining else "test"
        filename_X = f"{location}{dataname_address}{dataname}_X_{suffix}_F{self.foldindex}.txt"
        filename_y = f"{location}{dataname_address}{dataname}_y_{suffix}_F{self.foldindex}.txt"
        return filename_X, filename_y

    def _read_matrix_file(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"empty dataset file: {filename}")

        header = lines[0].replace(",", "\t").split("\t")
        if len(header) < 2:
            raise ValueError(f"invalid header in dataset file: {filename}")

        num = int(float(header[0]))
        dim = int(float(header[1]))

        data = []
        for line in lines[1:1 + num]:
            cols = line.replace(",", "\t").split("\t")
            if len(cols) < dim:
                raise ValueError(f"invalid row in dataset file: {filename}")
            data.append([float(cols[i]) for i in range(dim)])

        return num, dim, data

    def setProblem(self, state: EvolutionState, loca: str, datan: str, fitn: str, istraining: bool):
        self.location = loca
        self.dataname = datan
        self._istraining = istraining

        filename_X, filename_y = self._build_data_paths(self.location, self.dataname, self._istraining)
        print(f"evaluating on X: {filename_X}, Y: {filename_y}")

        if not os.path.exists(filename_X):
            raise FileNotFoundError(f"the dataset {filename_X} does not exist")
        if self._istraining and not os.path.exists(filename_y):
            raise FileNotFoundError(f"the dataset {filename_y} does not exist")

        self.datanum, self.datadim, self.data = self._read_matrix_file(filename_X)
        self.data_max = [-1e7] * self.datadim
        self.data_min = [1e7] * self.datadim
        for row in self.data:
            for i in range(self.datadim):
                if row[i] > self.data_max[i]:
                    self.data_max[i] = row[i]
                if row[i] < self.data_min[i]:
                    self.data_min[i] = row[i]

        self.outputnum, self.outputdim, self.data_output = self._read_matrix_file(filename_y)

        self.class_num = len(set([row[0] for row in self.data_output]))
        if self.class_num < 2:
            raise ValueError("the number of classes must be >= 2")
        self.class_labels = sorted(list(set([row[0] for row in self.data_output])))

        self.fitness = fitn
        if self.fitness not in ["ACC", "ERR", "RSE", "Fisher", "CONF"]:
            raise ValueError(f"{self.fitness} must be one of the following objectives: ACC, ERR, RSE, Fisher, CONF")

        if self.normalized:
            self.normalizedataBasedTraining()

        if self._istraining and state is not None and self.doValidation:
            self.validate_data = []
            self.validate_data_output = []

            self.validatenum = int(math.ceil(0.1 * self.datanum))

            usingdata = self.normdata if self.normalized else self.data

            for _ in range(self.validatenum):
                i = state.random[0].randint(0, len(usingdata) - 1)

                self.validate_data.append(usingdata[i])
                self.validate_data_output.append(self.data_output[i])

                usingdata.pop(i)
                self.data_output.pop(i)

            self.datanum = len(usingdata)
            self.outputnum = len(self.data_output)

    def load_data(self, state:EvolutionState, loca:str, datan:str, istraining:bool=True):
        """mostly the same as setProblem(...) but without the setting of fitness function"""
        self.location = loca
        self.dataname = datan
        self._istraining = istraining

        filename_X, filename_y = self._build_data_paths(self.location, self.dataname, self._istraining)
        print(f"evaluating on X: {filename_X}, Y: {filename_y}")

        if not os.path.exists(filename_X):
            raise FileNotFoundError(f"the dataset {filename_X} does not exist")
        if self._istraining and not os.path.exists(filename_y):
            raise FileNotFoundError(f"the dataset {filename_y} does not exist")

        self.datanum, self.datadim, self.data = self._read_matrix_file(filename_X)
        self.data_max = [-1e7] * self.datadim
        self.data_min = [1e7] * self.datadim
        for row in self.data:
            for i in range(self.datadim):
                if row[i] > self.data_max[i]:
                    self.data_max[i] = row[i]
                if row[i] < self.data_min[i]:
                    self.data_min[i] = row[i]

        self.outputnum, self.outputdim, self.data_output = self._read_matrix_file(filename_y)

        self.class_num = len(set([row[0] for row in self.data_output]))
        if self.class_num < 2:
            raise ValueError("the number of classes must be >= 2")
        self.class_labels = sorted(list(set([row[0] for row in self.data_output])))
        
        if self.normalized:
            self.normalizedataBasedTraining()

        if self._istraining and state is not None and self.doValidation:
            self.validate_data = []
            self.validate_data_output = []

            self.validatenum = int(math.ceil(0.1 * self.datanum))

            usingdata = self.normdata if self.normalized else self.data

            for _ in range(self.validatenum):
                i = state.random[0].randint(0, len(usingdata) - 1)

                self.validate_data.append(usingdata[i])
                self.validate_data_output.append(self.data_output[i])

                usingdata.pop(i)
                self.data_output.pop(i)

            self.datanum = len(usingdata)
            self.outputnum = len(self.data_output)

    @override
    def setX(self, X):
        super().setX(X)

    @override
    def setY(self, y):
        super().setY(y)

        self.class_num = len(set([row[0] for row in self.data_output]))
        if self.class_num < 2:
            raise ValueError("the number of classes must be >= 2")
        
        self.class_labels = sorted(list(set([row[0] for row in self.data_output])))


    @override
    def setup(self, state: EvolutionState, base: Parameter):
        super().setup(state, base)

        default = Parameter(self.PROBLEM_P)

        if not isinstance(self.input, GPData):
            state.output.fatal(f"GPData class must subclass from {GPData}", base.push(self.P_DATA), None)

        self.location = state.parameters.getString(base.push(self.LOCATION_P), default.push(self.LOCATION_P))
        if self.location == "":
            state.output.fatal("we got empty location for the data", base.push(self.LOCATION_P), default.push(self.LOCATION_P))

        self.dataname = state.parameters.getString(base.push(self.DATA_NAME_P), default.push(self.DATA_NAME_P))
        if self.dataname == "":
            state.output.fatal("we got empty name for the data", base.push(self.DATA_NAME_P), default.push(self.DATA_NAME_P))

        self.fitness = state.parameters.getString(base.push(self.FITNESS_P), default.push(self.FITNESS_P))
        self.normalized = state.parameters.getBoolean(base.push(self.NORMALIZE_P), default.push(self.NORMALIZE_P), False)
        self.doValidation = state.parameters.getBoolean(base.push(self.VALIDATION_P), default.push(self.VALIDATION_P), False)

        self.foldindex = state.parameters.getIntWithDefault(base.push(self.KFOLDINDEX_P), default.push(self.KFOLDINDEX_P), 0)
        self.foldnum = state.parameters.getInt(base.push(self.KFOLDNUM_P), default.push(self.KFOLDNUM_P))
        if self.foldnum < 0:
            raise ValueError("A multi-target symbolic regression problem need to set a K-fold number.")

        self.target_num = state.parameters.getIntWithDefault(base.push(self.TARGETNUM_P), default.push(self.TARGETNUM_P), 1)
        if self.target_num <= 0:
            state.output.fatal(
                "A multi-target symbolic regression problem at least has one target.",
                base.push(self.TARGETNUM_P),
                default.push(self.TARGETNUM_P),
            )

        self.targets = []
        for t in range(self.target_num):
            b = base.push(self.TARGETS_P).push(str(t))
            tar = state.parameters.getIntWithDefault(b, None, 0)
            if tar < 0:
                raise ValueError("target index must be >= 0.")
            self.targets.append(tar)

        self.class_num = state.parameters.getInt(base.push(self.CLASSNUM_P), default.push(self.CLASSNUM_P))
        if self.class_num < 2:
            raise ValueError("the number of classes must be >= 2")

        self.class_labels = []
        for c in range(self.class_num):
            cl = state.parameters.getDoubleWithDefault(
                base.push(self.CLASS_P).push(str(c)),
                default.push(self.CLASS_P).push(str(c)),
                0.0,
            )
            self.class_labels.append(cl)

        self.setProblem(state, self.location, self.dataname, self.fitness, True)

    def getDatanum(self):
        return self.datanum

    def getDatadim(self):
        return self.datadim

    def getOutputnum(self):
        return self.outputnum

    def getOutputdim(self):
        return self.outputdim

    def getTargets(self):
        return self.targets

    def getTargetNum(self):
        return self.target_num

    def getDataMax(self):
        return self.data_max

    def getDataMin(self):
        return self.data_min

    def getData(self):
        return self.normdata if self.normalized else self.data

    def getDataOutput(self):
        return self.data_output

    def getX(self):
        return self.X

    def getX_index(self):
        return self.X_index

    def setX_index(self, ind: int):
        self.X_index = ind

    def isnormalized(self):
        return self.normalized

    def istraining(self):
        return self._istraining

    def setFoldIndex(self, ind: int, istraining: bool):
        self.foldindex = ind
        self.setProblem(None, self.location, self.dataname, self.fitness, istraining)

    def getFoldNum(self):
        return self.foldnum

    def getClassNum(self):
        return self.class_num

    def getClassLabels(self):
        return self.class_labels

    def getClassLabelFromOutput(self, res: list[float]):
        if len(res) < self.class_num:
            raise ValueError("the number of outputs is smaller the number of classes")

        logits = np.asarray(res[:self.class_num], dtype=float)
        # Stable softmax: subtract max logit to avoid overflow.
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= (np.sum(probs) + 1e-7)

        index = int(np.argmax(probs))
        return self.class_labels[index]

    @override
    def evaluate(self, state: EvolutionState, ind: GPIndividual, subpopulation: int, threadnum: int):
        if ind.evaluated:
            return

        if len(self.data) == 0 or len(self.data_output) == 0:
            raise RuntimeError("we have an empty data source")

        real = self.data_output
        predict = []

        for y in range(self.datanum):
            tmp = GPData()

            self.X = [0.0] * self.datadim
            self.X_index = y
            for d in range(self.datadim):
                if self.normalized:
                    self.X[d] = self.normdata[y][d]
                else:
                    self.X[d] = self.data[y][d]

            if isinstance(ind, LGPIndividual4Class):
                ind.setDataIndex(y)

            predict.append(ind.execute_outs(state, threadnum, tmp, ind, self))

        if ind.IsWrap():
            real_care = []
            for i in range(len(real)):
                tmp = [0.0] * self.target_num
                for od in range(self.target_num):
                    tmp[od] = real[i][self.targets[od]]
                real_care.append(tmp)
            predict = ind.wrapper(predict, real_care, state, threadnum, self)

        result = 0.0
        for od in range(self.target_num):
            result_tmp = 0.0
            real_d = [0.0] * self.datanum
            predict_d = [0.0] * self.datanum
            predict_confidence = [[0.0] * len(predict[0]) for _ in range(self.datanum)]

            for y in range(self.datanum):
                di = self.targets[od]
                real_d[y] = real[y][di]
                predict_d[y] = self.getClassLabelFromOutput(predict[y])
                for i in range(len(predict[y])):
                    predict_confidence[y][i] = predict[y][i]

            if self.fitness == "ACC":
                result_tmp = self.getAccuracy(real_d, predict_d)
            elif self.fitness == "ERR":
                result_tmp = self.getError(real_d, predict_d)
            elif self.fitness == "RSE":
                result_tmp = self.getRSE(real_d, predict_d)
            elif self.fitness == "CONF":
                result_tmp = self.getConfidence(real_d, predict_confidence)
            else:
                raise ValueError("unknown fitness objective " + self.fitness)

            result += result_tmp / self.target_num

        validate_res = self.validationevaluation(state, ind, subpopulation, threadnum)
        fitness_value = result + 0.1 * validate_res

        if ind.fitness is None:
            ind.fitness = Fitness()
        ind.fitness.setFitness(state, fitness_value)
        ind.evaluated = True

    def validationevaluation(self, state: EvolutionState, ind: GPIndividual, subpopulation: int, threadnum: int):
        if not self.doValidation:
            return 0.0

        real = self.validate_data_output
        predict = []

        for y in range(self.validatenum):
            tmp = GPData()

            self.X = [0.0] * self.datadim
            self.X_index = y
            for d in range(self.datadim):
                self.X[d] = self.validate_data[y][d]

            predict.append(ind.execute_outs_wrap(state, threadnum, tmp, ind, self))

        result = 0.0
        for od in range(self.target_num):
            result_tmp = 0.0
            real_d = [0.0] * self.validatenum
            predict_d = [0.0] * self.validatenum
            predict_confidence = [[0.0] * len(predict[0]) for _ in range(self.validatenum)]

            for y in range(self.validatenum):
                di = self.targets[od]
                real_d[y] = real[y][di]
                predict_d[y] = self.getClassLabelFromOutput(predict[y])
                for i in range(len(predict[y])):
                    predict_confidence[y][i] = predict[y][i]

            if self.fitness == "ACC":
                result_tmp = self.getAccuracy(real_d, predict_d)
            elif self.fitness == "ERR":
                result_tmp = self.getError(real_d, predict_d)
            elif self.fitness == "RSE":
                result_tmp = self.getRSE(real_d, predict_d)
            elif self.fitness == "CONF":
                result_tmp = self.getConfidence(real_d, predict_confidence)
            else:
                raise ValueError("unknown fitness objective " + self.fitness)

            result += result_tmp / self.target_num

        return result

    def simpleevaluate(self, ind: GPIndividual):
        if ind.evaluated:
            return

        if len(self.data) == 0 or len(self.data_output) == 0:
            raise RuntimeError("we have an empty data source")

        real = self.data_output
        predict = []

        for y in range(self.datanum):
            tmp = GPData()

            self.X = [0.0] * self.datadim
            self.X_index = y
            for d in range(self.datadim):
                if self.normalized:
                    self.X[d] = self.normdata[y][d]
                else:
                    self.X[d] = self.data[y][d]

            if isinstance(ind, LGPIndividual4Class):
                ind.setDataIndex(y)

            predict.append(ind.execute_outs_wrap(None, 0, tmp, ind, self))

        result = 0.0
        for od in range(self.target_num):
            result_tmp = 0.0
            real_d = [0.0] * self.datanum
            predict_d = [0.0] * self.datanum
            predict_confidence = [[0.0] * len(predict[0]) for _ in range(self.datanum)]

            for y in range(self.datanum):
                di = self.targets[od]
                real_d[y] = real[y][di]
                predict_d[y] = self.getClassLabelFromOutput(predict[y])
                for i in range(len(predict[y])):
                    predict_confidence[y][i] = predict[y][i]

            if self.fitness == "ACC":
                result_tmp = self.getAccuracy(real_d, predict_d)
            elif self.fitness == "ERR":
                result_tmp = self.getError(real_d, predict_d)
            elif self.fitness == "RSE":
                result_tmp = self.getRSE(real_d, predict_d)
            elif self.fitness == "CONF":
                result_tmp = self.getConfidence(real_d, predict_confidence)
            else:
                raise ValueError("unknown fitness objective " + self.fitness)

            result += result_tmp / self.target_num

        if ind.fitness is None:
            ind.fitness = Fitness()
        ind.fitness.setFitness(None, result)
        ind.evaluated = True

    def quickevaluate(self, ind: GPIndividual):
        if len(self.data) == 0:
            raise RuntimeError("we have an empty data source")

        predict = []
        for y in range(self.datanum):
            tmp = GPData()

            self.X = [0.0] * self.datadim
            self.X_index = y
            for d in range(self.datadim):
                if self.normalized:
                    self.X[d] = self.normdata[y][d]
                else:
                    self.X[d] = self.data[y][d]

            if isinstance(ind, LGPIndividual4Class):
                ind.setDataIndex(y)

            predict.append(ind.execute_outs_wrap(None, 0, tmp, ind, self))

        res = []
        for y in range(self.datanum):
            label = self.getClassLabelFromOutput(predict[y])
            res.append([label])

        return res

    def getRSE(self, real: list[float], predict: list[float]):
        avg = sum(real) / len(real)
        var = 0.0
        for y in range(len(real)):
            var += (real[y] - avg) * (real[y] - avg)
        var /= len(real)

        mse = self.getMSE(real, predict)
        if math.isinf(mse) or math.isnan(mse):
            mse = 1e6

        if var == 0:
            return 1e6
        return mse / var

    def getMSE(self, real: list[float], predict: list[float]):
        res = 0.0
        for y in range(len(real)):
            tmp = abs(real[y] - predict[y])
            tmp = tmp * tmp
            res += tmp
        res = res / len(real)

        if math.isinf(res) or math.isnan(res):
            res = 1e6
        return res

    def getAccuracy(self, real: list[float], predict: list[float]):
        hit = 0.0
        for i in range(len(real)):
            if real[i] == predict[i]:
                hit += 1.0
        return hit / len(real)

    def getError(self, real: list[float], predict: list[float]):
        return 1.0 - self.getAccuracy(real, predict)

    def getConfidence(self, real: list[float], predict_conf: list[list[float]]):
        res = 0.0

        class_cnt = [0.0] * self.class_num
        class_index = [0] * len(real)
        sums = [0.0] * len(real)

        for i in range(len(real)):
            classi = 0
            while classi < self.class_num:
                if self.class_labels[classi] == real[i]:
                    break
                classi += 1

            if classi >= self.class_num:
                classi = 0

            class_cnt[classi] += 1.0
            class_index[i] = classi

            for c in range(self.class_num):
                sums[i] += math.exp(min(predict_conf[i][c], 10))
            sums[i] += 1e-7

        for i in range(len(real)):
            classi = class_index[i]
            denom = class_cnt[classi] * sums[i]
            if denom == 0:
                continue

            for c in range(self.class_num):
                if c == classi:
                    continue
                res += (
                    math.exp(min(predict_conf[i][c], 10))
                    - math.exp(min(predict_conf[i][classi], 10))
                ) / denom

        res /= ((self.class_num - 1) * (self.class_num - 1))
        return res

    def normalizedataBasedTraining(self):
        if self.data is None:
            return

        filename_X, filename_y = self._build_data_paths(self.location, self.dataname, True)

        _, _, traindata = self._read_matrix_file(filename_X)
        _, _, traindata_output = self._read_matrix_file(filename_y)

        num = len(traindata)
        dim = len(traindata[0])

        self.norm_mean = [0.0] * dim
        self.norm_std = [0.0] * dim
        self.normdata = []

        for d in range(dim):
            mean = 0.0
            for j in range(num):
                mean += traindata[j][d] / num
            self.norm_mean[d] = mean

        for d in range(dim):
            std = 0.0
            for j in range(num):
                std += ((traindata[j][d] - self.norm_mean[d]) ** 2) / num
            self.norm_std[d] = math.sqrt(std)

        for j in range(len(self.data)):
            tmp = [0.0] * dim
            for d in range(dim):
                if self.norm_std[d] > 0:
                    tmp[d] = (self.data[j][d] - self.norm_mean[d]) / self.norm_std[d]
                else:
                    tmp[d] = 0.0
            self.normdata.append(tmp)

        num = len(traindata_output)
        dim = len(traindata_output[0])

        self.out_mean = [0.0] * dim
        self.out_std = [0.0] * dim

        for d in range(dim):
            mean = 0.0
            for j in range(num):
                mean += traindata_output[j][d] / num
            self.out_mean[d] = mean

        for d in range(dim):
            std = 0.0
            for j in range(num):
                std += ((traindata_output[j][d] - self.out_mean[d]) ** 2) / num
            self.out_std[d] = math.sqrt(std)

    def getEuclidean(self, a: list[float], b: list[float]):
        if len(a) != len(b):
            raise ValueError("the getEuclidean function in CPClassification got two inconsistent arraies")
        res = 0.0
        for i in range(len(a)):
            res += (a[i] - b[i]) * (a[i] - b[i])
        res = math.sqrt(res / len(a))
        return res
