import os
import re
from typing import List
from numbers import Number
from abc import ABC, abstractmethod
from src.ec import *
from src.ec.util import *
from src.lgp.individual import LGPIndividual
from src.lgp.individual.primitive import *

# Assuming these imports have equivalent Python implementations
# from ec.gp import GPNode, GPTree
# from zhixing.symbreg_multitarget.individual import LGPIndividual4SRMT
# from zhixing.symbreg_multitarget.individual.primitive import InputFeature4SRMT
# and all other primitive classes...

class LispParser4SR:
    @staticmethod
    def parseSymRegRule(expression: str) -> GPTree:
        tree = GPTree()
        expression = expression.strip()
        tree.child = LispParser4SR.parseNode(expression)
        return tree
    
    @staticmethod
    def parseSRLGPRule(expression: str, numRegs: int, maxIterations: int) -> LGPIndividual:
        tree = None
        expression = expression.strip()
        lines = expression.split('\n')
        
        rule = LGPIndividual()
        rule.resetIndividual(numRegs, maxIterations)
        
        i = 0
        instruction = lines[i]
        i += 1
        
        while not instruction.startswith("#"):
            if instruction.startswith("//"):
                instruction = instruction[2:]
            
            # Remove the "Ins index"
            next_ws_idx = instruction.find('\t')
            instruction = instruction[next_ws_idx + 1:].strip()
            
            tree = LispParser4SR.parseSymRegRule(instruction)
            rule.addTree(rule.getTreesLength(), tree)
            
            if i < len(lines):
                instruction = lines[i]
                i += 1
            else:
                break
        
        return rule
    
    @staticmethod
    def parseNode(expression: str) -> GPNode:
        node = None
        
        if expression.startswith('('):
            next_ws_idx = expression.find(' ')
            func = expression[1:next_ws_idx]
            args_str = expression[next_ws_idx + 1:-1]
            args = LispUtil.splitArguments(args_str)
            
            # LGP-specific nodes
            if func.startswith("R") and func.endswith("="):
                equal_idx = func.index('=')
                index = int(func[1:equal_idx])
                node = WriteRegisterGPNode()
                node.setIndex(index)
                node.children = [LispParser4SR.parseNode(args[0])]
                node.children[0].parent = node
                node.children[0].argposition = 0
            
            # elif func.startswith("IF>#"):
            #     bodylength = int(func[4:])
            #     node = WriteRegisterGPNode()
            #     node.children = [IFLargerThan()]
            #     node.children[0].setMaxBodyLength(bodylength)
            #     node.children[0].setBodyLength(bodylength)
            #     node.children[0].parent = node
            #     node.children[0].argposition = 0
            #     node.children[0].children = [
            #         LispParser4SR.parseNode(args[0]),
            #         LispParser4SR.parseNode(args[1])
            #     ]
            #     node.children[0].children[0].parent = node.children[0]
            #     node.children[0].children[1].parent = node.children[0]
            #     node.children[0].children[0].argposition = 0
            #     node.children[0].children[1].argposition = 1
            
            # elif func.startswith("IF<=#"):
            #     bodylength = int(func[5:])
            #     node = WriteRegisterGPNode()
            #     node.children = [IFLessEqual()]
            #     node.children[0].setMaxBodyLength(bodylength)
            #     node.children[0].setBodyLength(bodylength)
            #     node.children[0].parent = node
            #     node.children[0].argposition = 0
            #     node.children[0].children = [
            #         LispParser4SR.parseNode(args[0]),
            #         LispParser4SR.parseNode(args[1])
            #     ]
            #     node.children[0].children[0].parent = node.children[0]
            #     node.children[0].children[1].parent = node.children[0]
            #     node.children[0].children[0].argposition = 0
            #     node.children[0].children[1].argposition = 1
            
            # elif func.startswith("WHILE>#"):
            #     bodylength = int(func[7:])
            #     node = WriteRegisterGPNode()
            #     node.children = [WhileLargeLoop()]
            #     node.children[0].setMaxBodyLength(bodylength)
            #     node.children[0].setBodyLength(bodylength)
            #     node.children[0].parent = node
            #     node.children[0].argposition = 0
            #     node.children[0].children = [
            #         LispParser4SR.parseNode(args[0]),
            #         LispParser4SR.parseNode(args[1])
            #     ]
            #     node.children[0].children[0].parent = node.children[0]
            #     node.children[0].children[1].parent = node.children[0]
            #     node.children[0].children[0].argposition = 0
            #     node.children[0].children[1].argposition = 1
            
            # elif func.startswith("LRF_entity"):
            #     node = LinearRegFunc_EntityNode()
            #     left_bracket = expression.index('[')
            #     right_bracket = expression.index(']')
            #     args_str = expression[left_bracket:right_bracket+1]
            #     node.setFromString(args_str)
            #     num_child = node.getArguments().getMaxLength() - 1
            #     node.children = [LispParser4SR.parseNode(args[c]) for c in range(num_child)]
            #     for c in range(num_child):
            #         node.children[c].parent = node
            #         node.children[c].argposition = c
            
            # elif func.startswith("Temp"):
            #     node = Temperature()
            #     left_bracket = expression.index('[')
            #     right_bracket = expression.index(']')
            #     args_str = expression[left_bracket:right_bracket+1]
            #     node.setFromString(args_str)
            #     num_child = node.expectedChildren()
            #     node.children = [LispParser4SR.parseNode(args[c]) for c in range(num_child)]
            #     for c in range(num_child):
            #         node.children[c].parent = node
            #         node.children[c].argposition = c
            
            # Similar handling for other specialized nodes...
            # RadRF_entity, SinRF_entity, TanhRF_entity, PowRF_entity, ExpoRF_entity
            # AvgHub, MaxHub, MinHub
            
            else:
                # Standard operators
                if func in ["+", "add"]:
                    node = Add()
                    node.children = [
                        LispParser4SR.parseNode(args[0]),
                        LispParser4SR.parseNode(args[1])
                    ]
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1
                
                elif func in ["-", "sub"]:
                    node = Sub()
                    node.children = [
                        LispParser4SR.parseNode(args[0]),
                        LispParser4SR.parseNode(args[1])
                    ]
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1
                
                elif func in ["*", "mul"]:
                    node = Mul()
                    node.children = [
                        LispParser4SR.parseNode(args[0]),
                        LispParser4SR.parseNode(args[1])
                    ]
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1
                
                elif func in ["/", "div"]:
                    node = Div()
                    node.children = [
                        LispParser4SR.parseNode(args[0]),
                        LispParser4SR.parseNode(args[1])
                    ]
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1
                
                elif func == "max":
                    node = Max()
                    node.children = [
                        LispParser4SR.parseNode(args[0]),
                        LispParser4SR.parseNode(args[1])
                    ]
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1
                
                elif func == "min":
                    node = Min()
                    node.children = [
                        LispParser4SR.parseNode(args[0]),
                        LispParser4SR.parseNode(args[1])
                    ]
                    node.children[0].parent = node
                    node.children[1].parent = node
                    node.children[0].argposition = 0
                    node.children[1].argposition = 1
                
                # elif func == "if":
                #     node = IF()
                #     node.children = [
                #         LispParser4SR.parseNode(args[0]),
                #         LispParser4SR.parseNode(args[1]),
                #         LispParser4SR.parseNode(args[2])
                #     ]
                #     node.children[0].parent = node
                #     node.children[1].parent = node
                #     node.children[2].parent = node
                #     node.children[0].argposition = 0
                #     node.children[1].argposition = 1
                #     node.children[2].argposition = 2
                
                elif func == "sin":
                    node = Sin()
                    node.children = [LispParser4SR.parseNode(args[0])]
                    node.children[0].parent = node
                    node.children[0].argposition = 0
                
                elif func == "cos":
                    node = Cos()
                    node.children = [LispParser4SR.parseNode(args[0])]
                    node.children[0].parent = node
                    node.children[0].argposition = 0
                
                elif func == "ln":
                    node = Ln()
                    node.children = [LispParser4SR.parseNode(args[0])]
                    node.children[0].parent = node
                    node.children[0].argposition = 0
                
                elif func == "sqr":
                    node = Sqrt()
                    node.children = [LispParser4SR.parseNode(args[0])]
                    node.children[0].parent = node
                    node.children[0].argposition = 0
                
                elif func == "exp":
                    node = Exp()
                    node.children = [LispParser4SR.parseNode(args[0])]
                    node.children[0].parent = node
                    node.children[0].argposition = 0
                
                # elif func == "pow2":
                #     node = Pow2()
                #     node.children = [LispParser4SR.parseNode(args[0])]
                #     node.children[0].parent = node
                #     node.children[0].argposition = 0
                
                # elif func == "tanh":
                #     node = Tanh()
                #     node.children = [LispParser4SR.parseNode(args[0])]
                #     node.children[0].parent = node
                #     node.children[0].argposition = 0
                
                # elif func == "relu":
                #     node = ReLu()
                #     node.children = [LispParser4SR.parseNode(args[0])]
                #     node.children[0].parent = node
                #     node.children[0].argposition = 0
        
        else:
            if LispParser4SR.is_number(expression):
                node = ConstantGPNode(float(expression))
            # elif expression.startswith("Rad_entity"):
            #     node = Radius_EntityNode()
            #     node.setFromString(expression)
            elif expression.startswith("R"):
                index = int(expression[1:])
                node = ReadRegisterGPNode(index)
            elif expression.startswith("In"):
                index = int(expression[2:])
                node = InputFeatureGPNode(index)
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
            # elif expression.startswith("Element_LR"):
            #     node = Element_LR()
            #     node.setFromString(expression)
            # elif expression.startswith("PLSR_entity"):
            #     node = PLSR_EntityNode()
            #     node.setFromString(expression)
            # elif expression.startswith("1stD_LR_entity"):
            #     node = FirstDerivativeLR_EntityNode()
            #     node.setFromString(expression)
            # elif expression.startswith("CondLR_entity"):
            #     node = CondLR_EntityNode()
            #     node.setFromString(expression)
            
            node.children = []
        
        return node
    
    @staticmethod
    def main():
        path = "D:/zhixing/科研/plant_N_food/result/"
        algo = "LGP-TP-maxrange"
        scenario = "InGaARaman_V4_SNV_DA-RSE-0"
        
        source_path = os.path.join(path, algo, scenario)
        
        num_runs = 13
        num_regs = 30
        max_iterations = 100
        
        output_regs = [0]
        
        for run in range(12, num_runs):
            source_file = os.path.join(source_path, f"job.{run}.out.stat")
            out_file = os.path.join(source_path, f"job.{run}.bestrule.dot")
            
            expressions = ResultFileReader4LGPSRMT.readLispExpressionFromFile4LGP(
                source_file, num_regs, max_iterations, False, output_regs)
            
            best_expression = expressions[-1]
            rule = LispParser4SR.parseSRLGPRule(best_expression, num_regs, max_iterations)
            best_graphviz_tree = rule.makeGraphvizRule(output_regs)
            
            try:
                with open(out_file, 'w') as writer:
                    writer.write(best_graphviz_tree)
            except IOError as e:
                print(e)

    @staticmethod
    def is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
    
if __name__ == "__main__":
    LispParser4SR.main()