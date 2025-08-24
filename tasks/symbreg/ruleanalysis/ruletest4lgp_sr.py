import os
import sys
import math
import time
import pandas as pd


sys.path.append('D:/data/study/LGP4PY/LGP4PY')
from tasks.symbreg.optimization.gp_symbolic_regression import GPSymbolicRegression
from tasks.symbreg.ruleanalysis.result_file_reader4lgp_sr import TestResult4LGPSR
from src.ec.util import *

class RuleTest4LGPSRMT:
    maxgenerations = 5500

    def __init__(self, trainPath:str, dataPath:str, dataName:str, numRuns:int, numReg:int, maxIter:int, isMO:bool = False):
        self.trainPath = trainPath
        self.dataPath = dataPath
        self.numRuns = numRuns
        self.dataName = dataName
        self.objectives = []
        self.numRegs = numReg
        self.maxIterations = maxIter
        self.isMultiObj = isMO
        self.parameters:ParameterDatabase = None

    def addParamsfile(self, parameters):
        self.parameters = parameters

    def getDataPath(self):
        return self.dataPath

    def getNumRuns(self):
        return self.numRuns

    def getObjectives(self):
        return self.objectives

    def setObjectives(self, objectives):
        self.objectives = objectives

    def addObjective(self, objective):
        self.objectives.append(objective)

    def writeToCSV(self):
        # Placeholder: GPSymbolicRegressionMultiTarget class should be implemented separately
        problem = GPSymbolicRegression(
            self.dataPath, self.dataName, self.objectives[0], False, self.parameters
        )

        targetPath = os.path.join(self.trainPath, "test")
        os.makedirs(targetPath, exist_ok=True)

        csv_path = os.path.join(targetPath, f"{self.dataName}.csv")

        testResults:list[TestResult4LGPSR] = []
        allTestFitness = [[0] * self.numRuns for _ in range(self.maxgenerations)]

        numOutRegs = self.parameters.getInt(
            Parameter("pop.subpop.0.species.ind.num-output-register"), None
        )
        if numOutRegs <= 0:
            raise ValueError("the number of output registers is illegal in RuleTest")

        outputRegs = list(range(numOutRegs))

        for i in range(self.numRuns):
            sourceFile = os.path.join(self.trainPath, f"job.{i}.out.stat")
            if self.numRuns > 1:
                problem.setFoldIndex(i % problem.getFoldNum(), False)

            # Placeholder: This function should parse the result from file
            result:TestResult4LGPSR = TestResult4LGPSR.readFromFile4LGP(
                sourceFile, self.numRegs, self.maxIterations, self.isMultiObj, outputRegs
            )

            start = time.time()

            for j in range(len(result.getGenerationalRules())):
                if (
                    j % math.ceil(len(result.getGenerationalRules()) / 50.0) == 0
                    or j == len(result.getGenerationalRules()) - 1
                ):
                    problem.simpleevaluate(result.getGenerationalRule(j))
                    # fitnesses = [
                    #     result.getGenerationalRule(j).fitness.fitness()
                    #     for _ in self.objectives
                    # ]
                    fitnesses = result.getGenerationalRule(j).fitness.fitness()
                    result.getGenerationalTestFitness(j).setFitness(None, fitnesses)
                else:
                    prev_fit = result.getGenerationalTestFitness(j - 1).fitness()
                    # fitnesses = [prev_fit for _ in self.objectives]
                    result.getGenerationalTestFitness(j).setFitness(None, prev_fit)

                print(
                    f"Generation {j}: test fitness = {result.getGenerationalTestFitness(j).fitness()}"
                )

                if j < self.maxgenerations:
                    allTestFitness[j][i] = result.getGenerationalTestFitness(j).fitness()
                else:
                    raise ValueError(
                        f"the evolution generation is larger than {self.maxgenerations}"
                    )

            duration = time.time() - start
            print(f"Duration = {duration * 1000:.0f} ms.")

            testResults.append(result)

        # Main CSV
        rows = []
        for i in range(self.numRuns):
            result = testResults[i]
            for j in range(len(result.getGenerationalRules())):
                rule = result.getGenerationalRule(j)
                trainFit = result.getGenerationalTrainFitness(j)
                testFit = result.getGenerationalTestFitness(j)
                numUniqueTerminals = 0  # Placeholder

                if len(self.objectives) == 1:
                    rows.append(
                        [
                            i,
                            j,
                            rule.getTreesLength(),
                            numUniqueTerminals,
                            0,
                            trainFit.fitness(),
                            testFit.fitness(),
                            0,
                        ]
                    )
                else:
                    # base = [i, j, rule.getTreesLength(), numUniqueTerminals]
                    # for k in range(len(self.objectives)):
                    #     base.extend(
                    #         [k, trainFit.getObjective(k), testFit.getObjective(k)]
                    #     )
                    # base.append(0)
                    # rows.append(base)
                    raise ValueError("we do not support multi-objective yet in ruletest4LGP.py")

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # allTestFitness CSV
        all_test_path = os.path.join(targetPath, f"{self.dataName}-allTestFitness.csv")
        header = ["generation"] + list(range(self.numRuns))
        df_fitness = pd.DataFrame(
            [[j] + allTestFitness[j][: self.numRuns] for j in range(len(testResults[0].getGenerationalRules()))],
            columns=header,
        )
        df_fitness.to_csv(all_test_path, index=False)

        print(f"Results written to:\n{csv_path}\n{all_test_path}")


if __name__ == "__main__":
    idx = 1  # skip script name

    trainPath = sys.argv[idx]; idx += 1
    dataPath = sys.argv[idx]; idx += 1
    testSetName = sys.argv[idx]; idx += 1
    numRuns = int(sys.argv[idx]); idx += 1

    numRegs = int(sys.argv[idx]); idx += 1
    maxIteration = int(sys.argv[idx]); idx += 1
    numObjectives = int(sys.argv[idx]); idx += 1

    if numObjectives > 1:
        sys.stderr.write("the basic rule analysis in LGP for SR does not support multi-objective\n")
        sys.exit(1)

    ruleTest = RuleTest4LGPSRMT(trainPath, dataPath, testSetName, numRuns, numRegs, maxIteration, numObjectives > 1)

    for i in range(numObjectives):
        ruleTest.addObjective(sys.argv[idx]); idx += 1

    # algorithm-related parameters (placeholder for ParameterDatabase)
    try:
        params_path = os.path.abspath(sys.argv[idx])
        parameters = ParameterDatabase(params_path)
        # parameters = {"params_file": params_path, "args": sys.argv}
    except Exception as e:
        print(f"An exception was generated upon reading the parameter file \"{sys.argv[idx]}\".\nHere it is:\n{e}")
        sys.exit(1)
    idx += 1
    ruleTest.addParamsfile(parameters)

    ruleTest.writeToCSV()