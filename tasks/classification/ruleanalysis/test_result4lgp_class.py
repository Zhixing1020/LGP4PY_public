from typing import List, Optional
from src.ec import *
from src.ec.util import *
from tasks.classification.individual.lgpindividual4Class import LGPIndividual4Class



class TestResult4LGPClass:

    validationSimSeed = 483561

    def __init__(self):
        self.generationalRules: List[LGPIndividual4Class] = []
        self.generationalTrainFitnesses: List[Fitness] = []
        self.generationalValidationFitnesses: List[Fitness] = []
        self.generationalTestFitnesses: List[Fitness] = []
        self.bestInd: Optional[LGPIndividual4Class] = None
        self.bestTrainingFitness: Optional[Fitness] = None
        self.bestValidationFitness: Optional[Fitness] = None
        self.bestTestFitness: Optional[Fitness] = None

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------

    def getGenerationalRules(self) -> List:
        return self.generationalRules

    def getGenerationalRule(self, idx: int):
        return self.generationalRules[idx]

    def getGenerationalTrainFitnesses(self) -> List:
        return self.generationalTrainFitnesses

    def getGenerationalTrainFitness(self, idx: int):
        return self.generationalTrainFitnesses[idx]

    def getGenerationalValidationFitnesses(self) -> List:
        return self.generationalValidationFitnesses

    def getGenerationalValidationFitness(self, idx: int):
        return self.generationalValidationFitnesses[idx]

    def getGenerationalTestFitnesses(self) -> List:
        return self.generationalTestFitnesses

    def getGenerationalTestFitness(self, idx: int):
        return self.generationalTestFitnesses[idx]

    def getBestRule(self):
        return self.bestInd

    def getBestTrainingFitness(self):
        return self.bestTrainingFitness

    def getBestValidationFitness(self):
        return self.bestValidationFitness

    def getBestTestFitness(self):
        return self.bestTestFitness

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------

    def setGenerationalRules(self, generationalRules: list):
        self.generationalRules = generationalRules

    def addGenerationalRule(self, rule):
        self.generationalRules.append(rule)

    def setGenerationalTrainFitnesses(self, generationalTrainFitnesses: list):
        self.generationalTrainFitnesses = generationalTrainFitnesses

    def addGenerationalTrainFitness(self, f):
        self.generationalTrainFitnesses.append(f)

    def setGenerationalValidationFitnesses(self, generationalValidationFitnesses: list):
        self.generationalValidationFitnesses = generationalValidationFitnesses

    def addGenerationalValidationFitnesses(self, f):
        self.generationalValidationFitnesses.append(f)

    def setGenerationalTestFitnesses(self, generationalTestFitnesses: list):
        self.generationalTestFitnesses = generationalTestFitnesses

    def addGenerationalTestFitnesses(self, f):
        self.generationalTestFitnesses.append(f)

    def setBestRule(self, bestRule):
        self.bestInd = bestRule

    def setBestTrainingFitness(self, bestTrainingFitness):
        self.bestTrainingFitness = bestTrainingFitness

    def setBestValidationFitness(self, bestValidationFitness):
        self.bestValidationFitness = bestValidationFitness

    def setBestTestFitness(self, bestTestFitness):
        self.bestTestFitness = bestTestFitness

    def setGenerationalTimeStat(self, generationalTimeStat):
        self.generationalTimeStat = generationalTimeStat

    # -------------------------------------------------------------------------
    # Static factory methods
    # -------------------------------------------------------------------------

    @staticmethod
    def readFromFile4LGP(file, numRegs: int, maxIterations: int, isMultiObj: bool, outputRegs: List[int]) -> "TestResult4LGPClass":
        from tasks.classification.ruleanalysis.result_file_reader4lgp_class import ResultFileReader4LGPClass
        return ResultFileReader4LGPClass.readTestResultFromFile(file, numRegs, maxIterations, isMultiObj, outputRegs)

    # @staticmethod
    # def readFromFile4TGP(file, isMultiObj: bool) -> "TestResult4LGPClass":
    #     return ResultFileReader4TGPClass.readTestResultFromFile(file, isMultiObj)