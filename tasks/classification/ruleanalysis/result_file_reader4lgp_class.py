import sys
import re
from typing import List, Optional

from src.ec.fitness import Fitness
from tasks.classification.individual.lgpindividual4Class import LGPIndividual4Class
from tasks.classification.ruleanalysis.test_result4lgp_class import TestResult4LGPClass
from tasks.classification.util.lisp_parser4Class import LispParser4Class


class ResultFileReader4LGPClass:

    @staticmethod
    def readTestResultFromFile(file, numRegs: int, maxIterations: int,
                               isMultiObjective: bool, outputRegs: List[int]):
        result = TestResult4LGPClass()
        rule = None
        fitness = None
        tree = None

        try:
            with open(file, 'r') as br:
                for line in br:
                    line = line.rstrip('\n')

                    if line == "Best Individual of Run:":
                        break

                    if line.startswith("Generation"):

                        rule = LGPIndividual4Class()
                        if outputRegs is None:
                            rule.resetIndividual(numRegs, maxIterations)
                        else:
                            rule.resetIndividual(numRegs, maxIterations, outputRegs)

                        next(br)  # skip line
                        next(br)  # skip line
                        next(br)  # skip line
                        line = next(br).rstrip('\n')
                        fitness = ResultFileReader4LGPClass.readFitnessFromLine(line, isMultiObjective)
                        expression = next(br).rstrip('\n')

                        while not expression.startswith("#"):
                            if expression.startswith("//"):
                                expression = expression[2:]

                            # remove the "Ins index"
                            nextWhiteSpaceIdx = expression.index('\t')
                            expression = expression[nextWhiteSpaceIdx + 1:]
                            expression.strip()

                            tree = LispParser4Class.parseSymRegRule(expression)
                            rule.addTree(rule.getTreesLength(), tree)
                            expression = next(br).rstrip('\n')

                        result.addGenerationalRule(rule)
                        result.addGenerationalTrainFitness(fitness)
                        result.addGenerationalValidationFitnesses(fitness.clone())
                        result.addGenerationalTestFitnesses(fitness.clone())

        except IOError as e:
            print(e)

        # Set the best rule as the rule in the last generation
        if rule is not None:
            result.setBestRule(rule)
            result.setBestTrainingFitness(fitness)

        return result

    @staticmethod
    def readFitnessFromLine(line: str, isMultiobjective: bool):
        if isMultiobjective:
            # TODO read multi-objective fitness line
            # spaceSegments = line.split()
            # equation = spaceSegments[1].split("=")
            # fitness = float(equation[1])
            # f = KozaFitness()
            # f.setStandardizedFitness(None, fitness)
            # return f
            raise ValueError("we do not support multi-objective fitness yet")
        else:
            spaceSegments = line.split()
            fitVec = re.split(r'\[|\]', spaceSegments[1])
            fitness = float(fitVec[1])
            f = Fitness()
            f.setFitness(None, fitness)
            return f

    @staticmethod
    def readLispExpressionFromFile4LGP(file, numRegs: int, maxIterations: int,
                                       isMultiObjective: bool, outputRegs: List[int]) -> List[str]:
        expressions = []

        rule = None
        ruleString = ""
        fitness = None
        tree = None

        try:
            with open(file, 'r') as br:
                for line in br:
                    line = line.rstrip('\n')

                    if line == "Best Individual of Run:":
                        break

                    if line.startswith("Generation"):

                        rule = LGPIndividual4Class()
                        if outputRegs is None:
                            rule.resetIndividual(numRegs, maxIterations)
                        else:
                            rule.resetIndividual(numRegs, maxIterations, outputRegs)
                        ruleString = ""

                        next(br)  # skip line
                        next(br)  # skip line
                        next(br)  # skip line
                        line = next(br).rstrip('\n')
                        fitness = ResultFileReader4LGPClass.readFitnessFromLine(line, isMultiObjective)
                        expression = next(br).rstrip('\n')

                        while not expression.startswith("#"):

                            ruleString += expression + "\n"

                            if expression.startswith("//"):
                                expression = expression[2:]

                            # remove the "Ins index"
                            nextWhiteSpaceIdx = expression.index('\t')
                            expression = expression[nextWhiteSpaceIdx + 1:]
                            expression.strip()

                            tree = LispParser4Class.parseSymRegRule(expression)
                            rule.addTree(rule.getTreesLength(), tree)

                            expression = next(br).rstrip('\n')

                        ruleString += "#\n"
                        expressions.append(ruleString)

        except IOError as e:
            print(e)

        return expressions