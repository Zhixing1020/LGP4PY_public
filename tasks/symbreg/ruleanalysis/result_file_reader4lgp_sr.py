import pandas as pd
from pathlib import Path
from tasks.symbreg.ruleanalysis.test_result4lgp_sr import TestResult4LGPSR
from tasks.symbreg.individual.lgpindividual4SR import LGPIndividual4SR
from tasks.symbreg.util.lisp_parser4sr import LispParser4SR
from src.ec import *
# Assuming the following classes exist elsewhere in your project:
# from your_module import TestResult4CpxGPSRMT, LGPIndividual4SRMT, Fitness, KozaFitness, MultiObjectiveFitness, GPTree, LispParser4SRMT

class ResultFileReader4LGPSR:

    @staticmethod
    def readTestResultFromFile(file, numRegs, maxIterations, isMultiObjective, outputRegs):
        result = TestResult4LGPSR()
        rule = None
        fitness = None
        tree = None

        file_path = Path(file)

        try:
            with open(file_path, "r", encoding="utf-8") as br:
                for line in br:
                    line = line.strip()
                    if line == "Best Individual of Run:":
                        break

                    if line.startswith("Generation"):
                        rule = LGPIndividual4SR()
                        if outputRegs is None:
                            rule.resetIndividual(numRegs, maxIterations)
                        else:
                            rule.resetIndividual(numRegs, maxIterations, outputRegs)

                        # skip 3 lines
                        next(br)
                        next(br)
                        next(br)

                        # read fitness
                        line = next(br).strip()
                        fitness = ResultFileReader4LGPSR.readFitnessFromLine(line, isMultiObjective)

                        expression = next(br).strip()

                        while not expression.startswith("#"):
                            if expression.startswith("//"):
                                expression = expression[2:]

                            # remove "Ins index"
                            nextWhiteSpaceIdx = expression.find('\t')
                            if nextWhiteSpaceIdx != -1:
                                expression = expression[nextWhiteSpaceIdx + 1:].strip()

                            # parse expression
                            tree = LispParser4SR.parseSymRegRule(expression)
                            rule.addTree(rule.getTreesLength(), tree)

                            expression = next(br).strip()

                        result.addGenerationalRule(rule)
                        result.addGenerationalTrainFitness(fitness)
                        result.addGenerationalValidationFitnesses(fitness.clone())
                        result.addGenerationalTestFitnesses(fitness.clone())

        except IOError as e:
            print(f"Error reading file: {e}")

        if rule is not None:
            result.setBestRule(rule)
            result.setBestTrainingFitness(fitness)

        return result

    @staticmethod
    def readFitnessFromLine(line, isMultiobjective)->Fitness:
        if isMultiobjective:
            # spaceSegments = line.split()
            # equation = spaceSegments[1].split("=")
            # fitness = float(equation[1])
            # f = Fitness()
            # f.setFitness(None, fitness)
            # return f
            raise ValueError("we do not support multi-objective fitness yet")
        else:
            spaceSegments = line.split()
            fitVec = spaceSegments[1].split("[")[1].split("]")[0]
            fitness = float(fitVec)
            f = Fitness()
            f.setFitness(None, fitness)
            return f

    @staticmethod
    def readLispExpressionFromFile4LGP(file, numRegs, maxIterations, isMultiObjective, outputRegs):
        expressions = []
        rule = None
        ruleString = ""
        fitness:Fitness = None
        tree = None

        file_path = Path(file)

        try:
            with open(file_path, "r", encoding="utf-8") as br:
                for line in br:
                    line = line.strip()
                    if line == "Best Individual of Run:":
                        break

                    if line.startswith("Generation"):
                        rule = LGPIndividual4SR()
                        if outputRegs is None:
                            rule.resetIndividual(numRegs, maxIterations)
                        else:
                            rule.resetIndividual(numRegs, maxIterations, outputRegs)

                        ruleString = ""

                        # skip 3 lines
                        next(br)
                        next(br)
                        next(br)

                        # read fitness
                        line = next(br).strip()
                        fitness = ResultFileReader4LGPSR.readFitnessFromLine(line, isMultiObjective)

                        expression = next(br).strip()

                        while not expression.startswith("#"):
                            ruleString += expression + "\n"

                            if expression.startswith("//"):
                                expression = expression[2:]

                            nextWhiteSpaceIdx = expression.find('\t')
                            if nextWhiteSpaceIdx != -1:
                                expression = expression[nextWhiteSpaceIdx + 1:].strip()

                            tree = LispParser4SR.parseSymRegRule(expression)
                            rule.addTree(rule.getTreesLength(), tree)

                            expression = next(br).strip()

                        ruleString += "#\n"
                        expressions.append(ruleString)

        except IOError as e:
            print(f"Error reading file: {e}")

        return expressions
