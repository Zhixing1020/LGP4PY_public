import sys
import os
import math
import csv
import time
from typing import List, Optional

sys.path.append('D:/data/study/LGP4PY/LGP4PY')
from src.ec import *
from src.ec.util import *
from tasks.classification.individual.lgpindividual4Class import LGPIndividual4Class
from tasks.classification.optimization.lgp_classification import LGPClassificationProblem
from tasks.classification.ruleanalysis.test_result4lgp_class import TestResult4LGPClass

class RuleTest4LGP_Classification:

    maxgenerations = 5500

    def __init__(self, trainPath: str, dataPath: str, dataName: str, numRuns: int,
                 numReg: int, maxIter: int, isMO: bool):
        self.trainPath = trainPath
        self.dataPath = dataPath
        self.numRuns = numRuns
        self.dataName = dataName
        self.objectives: List[str] = []
        self.numRegs = numReg
        self.maxIterations = maxIter
        self.isMultiObj = isMO
        self.parameters = None

    def addParamsfile(self, parameters):
        self.parameters = parameters

    def getDataPath(self) -> str:
        return self.dataPath

    def getNumRuns(self) -> int:
        return self.numRuns

    def getObjectives(self) -> List[str]:
        return self.objectives

    def setObjectives(self, objectives: List[str]):
        self.objectives = objectives

    def addObjective(self, objective: str):
        self.objectives.append(objective)

    def writeToCSV(self):
        problem = LGPClassificationProblem(self.dataPath, self.dataName, self.objectives[0], False, self.parameters)

        targetPath = os.path.join(self.trainPath, "test")
        if not os.path.exists(targetPath):
            os.makedirs(targetPath)

        csvFile = os.path.join(targetPath, self.dataName + ".csv")

        testResults: List[TestResult4LGPClass] = []

        allTestFitness = [[0.0] * self.numRuns for _ in range(self.maxgenerations)]

        numOutRegs = self.parameters.getInt("pop.subpop.0.species.ind.num-output-register")
        if numOutRegs <= 0:
            sys.stderr.write("the number of output registers is illegal in RuleTest")
            sys.exit(1)

        outputRegs: List[int] = []
        for r in range(numOutRegs):
            outputRegs.append(r)

        for i in range(self.numRuns):
            sourceFile = os.path.join(self.trainPath, f"job.{i}.out.stat")
            if self.numRuns > 1:
                problem.setFoldIndex(i % problem.getFoldNum(), False)
            result = TestResult4LGPClass.readFromFile4LGP(sourceFile, self.numRegs, self.maxIterations, self.isMultiObj, outputRegs)

            start = time.time()

            for j in range(len(result.getGenerationalRules())):

                if (j % math.ceil(len(result.getGenerationalRules()) / 50.0) == 0
                        or j == len(result.getGenerationalRules()) - 1):

                    problem.simpleevaluate(result.getGenerationalRule(j))

                    # fitnesses = [0.0] * len(self.objectives)
                    # for f in range(len(self.objectives)):
                    #     fitnesses[f] = result.getGenerationalRule(j).fitness.fitness()

                    fitnesses = result.getGenerationalRule(j).fitness.fitness()

                    result.getGenerationalTestFitness(j).setFitness(None, fitnesses)

                else:
                    # fitnesses = [0.0] * len(self.objectives)
                    # for f in range(len(self.objectives)):
                    #     fitnesses[f] = result.getGenerationalTestFitness(j - 1).fitness()

                    fitnesses = result.getGenerationalTestFitness(j - 1).fitness()
                    result.getGenerationalTestFitness(j).setFitness(None, fitnesses)

                print(f"Generation {j}: test fitness = {result.getGenerationalTestFitness(j).fitness()}")

                if j < self.maxgenerations:
                    allTestFitness[j][i] = result.getGenerationalTestFitness(j).fitness()
                else:
                    print(f"the evolution generation is larger than {self.maxgenerations}")
                    sys.exit(1)

            finish = time.time()
            duration = (finish - start) * 1000  # convert to ms
            print(f"Duration = {duration:.0f} ms.")

            testResults.append(result)

        try:
            with open(csvFile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Run", "Generation", "Size", "UniqueTerminals", "Obj", "TrainFitness", "TestFitness", "Time"])

                for i in range(self.numRuns):
                    result = testResults[i]

                    for j in range(len(result.getGenerationalRules())):
                        rule = result.getGenerationalRule(j)  # LGPIndividual

                        trainFit = result.getGenerationalTrainFitness(j)   # MultiObjectiveFitness
                        testFit = result.getGenerationalTestFitness(j)     # MultiObjectiveFitness

                        numUniqueTerminals = 0

                        if len(self.objectives) == 1:
                            writer.writerow([
                                i, j,
                                rule.getTreesLength(),
                                numUniqueTerminals,
                                0,
                                trainFit.fitness(),
                                testFit.fitness(),
                                0
                            ])
                        else:
                            # row = [i, j, rule.getTreesLength(), numUniqueTerminals]
                            # for k in range(len(self.objectives)):
                            #     row += [k, trainFit.getObjective(k), testFit.getObjective(k)]
                            # row.append(0)
                            # writer.writerow(row)
                            raise ValueError("we do not support multi-objective yet in ruletest4LGP.py")

            allTestFitnessFile = os.path.join(targetPath, self.dataName + "-allTestFitness.csv")
            with open(allTestFitnessFile, "w", newline="") as f:
                writer = csv.writer(f)

                header = ["generation:"] + list(range(self.numRuns))
                writer.writerow(header)

                gen = len(testResults[0].getGenerationalRules())
                for j in range(gen):
                    row = [j] + [allTestFitness[j][i] for i in range(self.numRuns)]
                    writer.writerow(row)

            print(f"Results written to:{allTestFitnessFile}")

        except IOError as e:
            print(e)

    @staticmethod
    def main(args: List[str]):
        idx = 0

        trainPath = args[idx]; idx += 1
        dataPath = args[idx]; idx += 1
        testSetName = args[idx]; idx += 1
        numRuns = int(args[idx]); idx += 1
        numRegs = int(args[idx]); idx += 1
        maxIteration = int(args[idx]); idx += 1
        numObjectives = int(args[idx]); idx += 1

        if numObjectives > 1:
            sys.stderr.write(
                "the basic rule analysis in LGP for SR does not support multi-objective\n"
            )
            sys.exit(1)

        ruleTest = RuleTest4LGP_Classification(
            trainPath, dataPath, testSetName, numRuns, numRegs, maxIteration, numObjectives > 1
        )

        for i in range(numObjectives):
            ruleTest.addObjective(args[idx]); idx += 1

        parameters = None
        try:
            parameters = ParameterDatabase(os.path.abspath(args[idx]), args)
        except Exception as e:
            print(e)
            print(
                f"An exception was generated upon reading the parameter file \"{args[idx]}\".\n"
                f"Here it is:\n{e}"
            )
        idx += 1

        ruleTest.addParamsfile(parameters)
        ruleTest.writeToCSV()


if __name__ == "__main__":
    RuleTest4LGP_Classification.main(sys.argv[1:])