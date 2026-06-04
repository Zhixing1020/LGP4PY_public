import sys
import os
from typing import List, Optional

from src.ec import *
from src.ec.util import *
from src.lgp.individual import LGPIndividual
from src.lgp.individual.primitive import *

from tasks.classification.individual.lgpindividual4Class import LGPIndividual4Class
from tasks.classification.individual.primitives.inputFeature4Class import InputFeature4Class

class LispParser4Class:

    @staticmethod
    def parseSymRegRule(expression: str):
        tree = GPTree()
        expression = expression.strip()
        tree.child = LispParser4Class.parseNode(expression)
        return tree

    @staticmethod
    def parseClassLGPRule(expression: str, numRegs: int, maxIterations: int):
        tree = None
        rule = LGPIndividual4Class()
        rule.resetIndividual(numRegs, maxIterations)

        expression = expression.strip()
        split = expression.split("\n")

        i = 0
        instruction = split[i]; i += 1

        while not instruction.startswith("#"):
            if instruction.startswith("//"):
                instruction = instruction[2:]

            # remove the "Ins index"
            nextWhiteSpaceIdx = instruction.index('\t')
            instruction = instruction[nextWhiteSpaceIdx + 1:]
            instruction.strip()

            tree = LispParser4Class.parseSymRegRule(instruction)
            rule.addTree(rule.getTreesLength(), tree)

            instruction = split[i]; i += 1

        return rule

    @staticmethod
    def parseNode(expression: str):
        node = None

        if expression[0] == '(':
            nextWhiteSpaceIdx = expression.index(' ')
            func = expression[1:nextWhiteSpaceIdx]
            argsString = expression[nextWhiteSpaceIdx + 1: len(expression) - 1]
            args = LispUtil.splitArguments(argsString)

            # ---- LGP-specific prefix patterns --------------------------------

            if func.startswith("R") and func.endswith("="):
                equalIdx = func.index('=')
                indexStr = func[1:equalIdx]
                index = int(indexStr)
                node = WriteRegisterGPNode()
                node.setIndex(index)
                node.children = [None] * 1
                node.children[0] = LispParser4Class.parseNode(args[0])
                node.children[0].parent = node
                node.children[0].argposition = 0

            # elif func.startswith("IF>#"):
            #     NumIdx = func.index('#')
            #     NumStr = func[NumIdx + 1:]
            #     bodylength = int(NumStr)
            #     node = WriteRegisterGPNode()
            #     node.children = [None] * 1
            #     node.children[0] = IFLargerThan()
            #     node.children[0].setMaxBodyLength(bodylength)
            #     node.children[0].setBodyLength(bodylength)
            #     node.children[0].parent = node
            #     node.children[0].argposition = 0
            #     node.children[0].children = [None] * 2
            #     node.children[0].children[0] = LispParser4Class.parseNode(args[0])
            #     node.children[0].children[1] = LispParser4Class.parseNode(args[1])
            #     node.children[0].children[0].parent = node.children[0]
            #     node.children[0].children[1].parent = node.children[0]
            #     node.children[0].children[0].argposition = 0
            #     node.children[0].children[1].argposition = 1

            # elif func.startswith("IF<=#"):
            #     NumIdx = func.index('#')
            #     NumStr = func[NumIdx + 1:]
            #     bodylength = int(NumStr)
            #     node = WriteRegisterGPNode()
            #     node.children = [None] * 1
            #     node.children[0] = IFLessEqual()
            #     node.children[0].setMaxBodyLength(bodylength)
            #     node.children[0].setBodyLength(bodylength)
            #     node.children[0].parent = node
            #     node.children[0].argposition = 0
            #     node.children[0].children = [None] * 2
            #     node.children[0].children[0] = LispParser4Class.parseNode(args[0])
            #     node.children[0].children[1] = LispParser4Class.parseNode(args[1])
            #     node.children[0].children[0].parent = node.children[0]
            #     node.children[0].children[1].parent = node.children[0]
            #     node.children[0].children[0].argposition = 0
            #     node.children[0].children[1].argposition = 1

            # elif func.startswith("WHILE>#"):
            #     NumIdx = func.index('#')
            #     NumStr = func[NumIdx + 1:]
            #     bodylength = int(NumStr)
            #     node = WriteRegisterGPNode()
            #     node.children = [None] * 1
            #     node.children[0] = WhileLargeLoop()
            #     node.children[0].setMaxBodyLength(bodylength)
            #     node.children[0].setBodyLength(bodylength)
            #     node.children[0].parent = node
            #     node.children[0].argposition = 0
            #     node.children[0].children = [None] * 2
            #     node.children[0].children[0] = LispParser4Class.parseNode(args[0])
            #     node.children[0].children[1] = LispParser4Class.parseNode(args[1])
            #     node.children[0].children[0].parent = node.children[0]
            #     node.children[0].children[1].parent = node.children[0]
            #     node.children[0].children[0].argposition = 0
            #     node.children[0].children[1].argposition = 1

            # elif func.startswith("LRF_entity"):
            #     node = LinearRegFunc_EntityNode()
            #     nextLeftBracketIdx = expression.index('[')
            #     nextRightBracketIdx = expression.index(']')
            #     args_str = expression[nextLeftBracketIdx: nextRightBracketIdx + 1]
            #     node.setFromString(args_str)
            #     num_child = node.getArguments().getMaxLength() - 1
            #     node.children = [None] * num_child
            #     for c in range(num_child):
            #         node.children[c] = LispParser4Class.parseNode(args[c])
            #         node.children[c].parent = node
            #         node.children[c].argposition = c

            # elif func.startswith("Temp"):
            #     node = Temperature()
            #     nextLeftBracketIdx = expression.index('[')
            #     nextRightBracketIdx = expression.index(']')
            #     args_str = expression[nextLeftBracketIdx: nextRightBracketIdx + 1]
            #     node.setFromString(args_str)
            #     num_child = node.expectedChildren()
            #     node.children = [None] * num_child
            #     for c in range(num_child):
            #         node.children[c] = LispParser4Class.parseNode(args[c])
            #         node.children[c].parent = node
            #         node.children[c].argposition = c

            # elif func.startswith("RadRF_entity"):
            #     node = RadRegFunc()
            #     nextLeftBracketIdx = expression.index('[')
            #     nextRightBracketIdx = expression.index(']')
            #     args_str = expression[nextLeftBracketIdx: nextRightBracketIdx + 1]
            #     node.setFromString(args_str)
            #     num_child = node.getArguments().getMaxLength() - 3
            #     node.children = [None] * num_child
            #     for c in range(num_child):
            #         node.children[c] = LispParser4Class.parseNode(args[c])
            #         node.children[c].parent = node
            #         node.children[c].argposition = c

            # elif func.startswith("SinRF_entity"):
            #     node = SinRegFunc()
            #     nextLeftBracketIdx = expression.index('[')
            #     nextRightBracketIdx = expression.index(']')
            #     args_str = expression[nextLeftBracketIdx: nextRightBracketIdx + 1]
            #     node.setFromString(args_str)
            #     num_child = node.expectedChildren()
            #     node.children = [None] * num_child
            #     for c in range(num_child):
            #         node.children[c] = LispParser4Class.parseNode(args[c])
            #         node.children[c].parent = node
            #         node.children[c].argposition = c

            # elif func.startswith("PowRF_entity"):
            #     node = PowRegFunc()
            #     nextLeftBracketIdx = expression.index('[')
            #     nextRightBracketIdx = expression.index(']')
            #     args_str = expression[nextLeftBracketIdx: nextRightBracketIdx + 1]
            #     node.setFromString(args_str)
            #     num_child = node.expectedChildren()
            #     node.children = [None] * num_child
            #     for c in range(num_child):
            #         node.children[c] = LispParser4Class.parseNode(args[c])
            #         node.children[c].parent = node
            #         node.children[c].argposition = c

            # elif func.startswith("ExpoRF_entity"):
            #     node = ExpoRegFunc()
            #     nextLeftBracketIdx = expression.index('[')
            #     nextRightBracketIdx = expression.index(']')
            #     args_str = expression[nextLeftBracketIdx: nextRightBracketIdx + 1]
            #     node.setFromString(args_str)
            #     num_child = node.expectedChildren()
            #     node.children = [None] * num_child
            #     for c in range(num_child):
            #         node.children[c] = LispParser4Class.parseNode(args[c])
            #         node.children[c].parent = node
            #         node.children[c].argposition = c

            # elif func.startswith("AvgHub"):
            #     node = AvgHub()
            #     num_child = 5
            #     node.children = [None] * num_child
            #     for c in range(num_child):
            #         node.children[c] = LispParser4Class.parseNode(args[c])
            #         node.children[c].parent = node
            #         node.children[c].argposition = c

            # elif func.startswith("MaxHub"):
            #     node = MaxHub()
            #     num_child = 5
            #     node.children = [None] * num_child
            #     for c in range(num_child):
            #         node.children[c] = LispParser4Class.parseNode(args[c])
            #         node.children[c].parent = node
            #         node.children[c].argposition = c

            # elif func.startswith("MinHub"):
            #     node = MinHub()
            #     num_child = 5
            #     node.children = [None] * num_child
            #     for c in range(num_child):
            #         node.children[c] = LispParser4Class.parseNode(args[c])
            #         node.children[c].parent = node
            #         node.children[c].argposition = c

            # ---- Standard function switch ------------------------------------

            else:
                if func in ("+", "add"):
                    node = Add()
                    node.children = [None] * 2
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[1] = LispParser4Class.parseNode(args[1])
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1

                elif func in ("-", "sub"):
                    node = Sub()
                    node.children = [None] * 2
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[1] = LispParser4Class.parseNode(args[1])
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1

                elif func in ("*", "mul"):
                    node = Mul()
                    node.children = [None] * 2
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[1] = LispParser4Class.parseNode(args[1])
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1

                elif func in ("/", "div"):
                    node = Div()
                    node.children = [None] * 2
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[1] = LispParser4Class.parseNode(args[1])
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1

                elif func == "max":
                    node = Max()
                    node.children = [None] * 2
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[1] = LispParser4Class.parseNode(args[1])
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1

                elif func == "min":
                    node = Min()
                    node.children = [None] * 2
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[1] = LispParser4Class.parseNode(args[1])
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1

                # elif func == "if":
                #     node = IF()
                #     node.children = [None] * 3
                #     node.children[0] = LispParser4Class.parseNode(args[0])
                #     node.children[1] = LispParser4Class.parseNode(args[1])
                #     node.children[2] = LispParser4Class.parseNode(args[2])
                #     node.children[0].parent = node
                #     node.children[1].parent = node
                #     node.children[2].parent = node
                #     node.children[0].argposition = 0
                #     node.children[1].argposition = 1
                #     node.children[2].argposition = 2

                elif func == "sin":
                    node = Sin()
                    node.children = [None] * 1
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[0].parent = node
                    node.children[0].argposition = 0

                elif func == "cos":
                    node = Cos()
                    node.children = [None] * 1
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[0].parent = node
                    node.children[0].argposition = 0

                elif func == "ln":
                    node = Ln()
                    node.children = [None] * 1
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[0].parent = node
                    node.children[0].argposition = 0

                elif func == "sqr":
                    node = Sqrt()
                    node.children = [None] * 1
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[0].parent = node
                    node.children[0].argposition = 0

                elif func == "exp":
                    node = Exp()
                    node.children = [None] * 1
                    node.children[0] = LispParser4Class.parseNode(args[0])
                    node.children[0].parent = node
                    node.children[0].argposition = 0

                # elif func == "pow2":
                #     node = Pow2()
                #     node.children = [None] * 1
                #     node.children[0] = LispParser4Class.parseNode(args[0])
                #     node.children[0].parent = node
                #     node.children[0].argposition = 0

                # elif func == "tanh":
                #     node = Tanh()
                #     node.children = [None] * 1
                #     node.children[0] = LispParser4Class.parseNode(args[0])
                #     node.children[0].parent = node
                #     node.children[0].argposition = 0

                # elif func == "relu":
                #     node = ReLu()
                #     node.children = [None] * 1
                #     node.children[0] = LispParser4Class.parseNode(args[0])
                #     node.children[0].parent = node
                #     node.children[0].argposition = 0

        # ---- Terminal branch -------------------------------------------------
        else:
            try:
                node = ConstantGPNode(float(expression))
            except ValueError:
                # if expression.startswith("Rad_entity"):
                #     node = Radius_EntityNode()
                #     node.setFromString(expression)

                if expression.startswith("R"):
                    indexStr = expression[1:]
                    index = int(indexStr)
                    node = ReadRegisterGPNode(index)

                elif expression.startswith("In"):
                    indexStr = expression[2:]
                    index = int(indexStr)
                    node = InputFeature4Class(index)

                # elif expression.startswith("Avg_Entity"):
                #     node = Avg_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("Std_Entity"):
                #     node = Std_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("Fluctuate_Entity"):
                #     node = Fluctuate_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("NegSlope_Entity"):
                #     node = NegSlope_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("PosSlope_Entity"):
                #     node = PosSlope_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("Peak_Entity"):
                #     node = Peak_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("Valley_Entity"):
                #     node = Valley_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("Magnitude_Entity"):
                #     node = Magnitude_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("PeakNum_Entity"):
                #     node = PeakNum_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("PeakLoc_Entity"):
                #     node = PeakLoc_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("Width_Entity"):
                #     node = Width_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("LR_entity"):
                #     node = LR_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("LDA_entity"):
                #     node = LDA_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("PLSR_entity"):
                #     node = PLSR_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("1stD_LR_entity"):
                #     node = FirstDerivativeLR_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("1stD_LDA_entity"):
                #     node = FirstDerivativeLDA_EntityNode()
                #     node.setFromString(expression)

                # elif expression.startswith("CondLR_entity"):
                #     node = CondLR_EntityNode()
                #     node.setFromString(expression)

            node.children = []

        return node

    @staticmethod
    def main(args: List[str]):
        path = "D:/zhixing/科研/plant_N_food/result/"
        algo = "LGP-TP-maxrange"
        scenario = "InGaARaman_V4_SNV_DA-RSE-0"

        sourcePath = path + algo + "/" + scenario + "/"

        numRuns = 13
        numRegs = 30
        maxIterations = 100

        outputRegs: List[int] = [0]

        from tasks.classification.ruleanalysis.result_file_reader4lgp_class import ResultFileReader4LGPClass

        for run in range(12, numRuns):
            sourceFile = os.path.join(sourcePath, f"job.{run}.out.stat")
            outFile = os.path.join(sourcePath, f"job.{run}.bestrule.dot")

            expressions = ResultFileReader4LGPClass.readLispExpressionFromFile4LGP(
                sourceFile, numRegs, maxIterations, False, outputRegs
            )

            bestExpression = expressions[-1]
            rule = LispParser4Class.parseClassLGPRule(bestExpression, numRegs, maxIterations)
            bestGraphVizTree = rule.makeGraphvizRule(outputRegs)

            try:
                with open(os.path.abspath(outFile), "w") as writer:
                    writer.write(bestGraphVizTree)
            except IOError as e:
                print(e)


if __name__ == "__main__":
    LispParser4Class.main(sys.argv[1:])