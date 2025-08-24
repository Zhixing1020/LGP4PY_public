import pandas as pd
from typing import List, Optional
from src.ec.util import *
from src.ec import Fitness
from tasks.symbreg.individual.lgpindividual4SR import LGPIndividual4SR

class TestResult4LGPSR:
    validationSimSeed = 483561

    def __init__(self):
        self.generationalRules: List['LGPIndividual4SR'] = []
        self.generationalTrainFitnesses: List['Fitness'] = []
        self.generationalValidationFitnesses: List['Fitness'] = []
        self.generationalTestFitnesses: List['Fitness'] = []
        self.bestInd: Optional['LGPIndividual4SR'] = None
        self.bestTrainingFitness: Optional['Fitness'] = None
        self.bestValidationFitness: Optional['Fitness'] = None
        self.bestTestFitness: Optional['Fitness'] = None
        self.generationalTimeStat: Optional[pd.Series] = None  # Pandas can store times

    # ----- Getters -----
    def getGenerationalRules(self) -> List['LGPIndividual4SR']:
        return self.generationalRules

    def getGenerationalRule(self, idx: int) -> 'LGPIndividual4SR':
        return self.generationalRules[idx]

    def getGenerationalTrainFitnesses(self) -> List['Fitness']:
        return self.generationalTrainFitnesses

    def getGenerationalTrainFitness(self, idx: int) -> 'Fitness':
        return self.generationalTrainFitnesses[idx]

    def getGenerationalValidationFitnesses(self) -> List['Fitness']:
        return self.generationalValidationFitnesses

    def getGenerationalValidationFitness(self, idx: int) -> 'Fitness':
        return self.generationalValidationFitnesses[idx]

    def getGenerationalTestFitnesses(self) -> List['Fitness']:
        return self.generationalTestFitnesses

    def getGenerationalTestFitness(self, idx: int) -> 'Fitness':
        return self.generationalTestFitnesses[idx]

    def getBestRule(self) -> 'LGPIndividual4SR':
        return self.bestInd

    def getBestTrainingFitness(self) -> 'Fitness':
        return self.bestTrainingFitness

    def getBestValidationFitness(self) -> 'Fitness':
        return self.bestValidationFitness

    def getBestTestFitness(self) -> 'Fitness':
        return self.bestTestFitness

    def getGenerationalTimeStat(self) -> pd.Series:
        return self.generationalTimeStat

    def getGenerationalTime(self, gen: int) -> float:
        if self.generationalTimeStat is None:
            raise ValueError("Generational time statistics not set.")
        return self.generationalTimeStat.iloc[gen]

    # ----- Setters -----
    def setGenerationalRules(self, generationalRules: List['LGPIndividual4SR']):
        self.generationalRules = generationalRules

    def addGenerationalRule(self, rule: 'LGPIndividual4SR'):
        self.generationalRules.append(rule)

    def setGenerationalTrainFitnesses(self, generationalTrainFitnesses: List['Fitness']):
        self.generationalTrainFitnesses = generationalTrainFitnesses

    def addGenerationalTrainFitness(self, f: 'Fitness'):
        self.generationalTrainFitnesses.append(f)

    def setGenerationalValidationFitnesses(self, generationalValidationFitnesses: List['Fitness']):
        self.generationalValidationFitnesses = generationalValidationFitnesses

    def addGenerationalValidationFitnesses(self, f: 'Fitness'):
        self.generationalValidationFitnesses.append(f)

    def setGenerationalTestFitnesses(self, generationalTestFitnesses: List['Fitness']):
        self.generationalTestFitnesses = generationalTestFitnesses

    def addGenerationalTestFitnesses(self, f: 'Fitness'):
        self.generationalTestFitnesses.append(f)

    def setBestRule(self, bestRule: 'LGPIndividual4SR'):
        self.bestInd = bestRule

    def setBestTrainingFitness(self, bestTrainingFitness: 'Fitness'):
        self.bestTrainingFitness = bestTrainingFitness

    def setBestValidationFitness(self, bestValidationFitness: 'Fitness'):
        self.bestValidationFitness = bestValidationFitness

    def setBestTestFitness(self, bestTestFitness: 'Fitness'):
        self.bestTestFitness = bestTestFitness

    def setGenerationalTimeStat(self, generationalTimeStat: List[float]):
        # Convert to pandas Series for convenience
        self.generationalTimeStat = pd.Series(generationalTimeStat)

    # ----- File Readers -----
    @staticmethod
    def readFromFile4LGP(file_path: str, numRegs: int, maxIterations: int, isMultiObj: bool, outputRegs: List[int]) -> 'TestResult4LGPSR':
        from .result_file_reader4lgp_sr import ResultFileReader4LGPSR
        return ResultFileReader4LGPSR.readTestResultFromFile(file_path, numRegs, maxIterations, isMultiObj, outputRegs)

    # @staticmethod
    # def readFromFile4TGP(file_path: str, isMultiObj: bool) -> 'TestResult4CpxGPSRMT':
    #     from ResultFileReader4TGPSRMT import readTestResultFromFile
    #     return readTestResultFromFile(file_path, isMultiObj)
