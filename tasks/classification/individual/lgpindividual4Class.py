
from operator import index

from src.ec.evolution_state import EvolutionState
from src.lgp.individual.lgp_individual import LGPIndividual, AtomicInteger, GPTreeStruct
from src.lgp.individual.primitive.readRegisterGPNode import ReadRegisterGPNode
import numpy as np
from sklearn.linear_model import Ridge
from typing import override
from src.ec import *
from tasks.problem import Problem
import sys

from sklearn.linear_model import LogisticRegression

class LGPIndividual4Class(LGPIndividual):

    CLASSNUM_P = "class_num"
    initVal = 0.0

    def __init__(self):
        super().__init__()
        self.dataindex = 0

    def setup(self, state, base):
        """Set up the individual prototype."""
        super().setup(state, base)

    @override
    def execute(self, state:EvolutionState, thread:int, input:GPData, individual:'LGPIndividual4Class', problem:Problem, with_wrap:bool = False):
        return super().execute(state, thread, input, individual, problem, with_wrap)
    
    def postExecution(self, state:EvolutionState, thread:int):
        return
    
    def getDataIndex(self): return self.dataindex

    def setDataIndex(self, index:int):
        self.dataindex = index

    # -------------------------------------------------------------------------
    # From LGPInterface4Class
    # -------------------------------------------------------------------------

    def execute_outs(self, state:EvolutionState, thread:int, input:GPData, individual:'LGPIndividual4Class', problem:Problem):
        ind = individual  # LGPIndividual

        ind.resetRegisters(problem, self.initVal, ind)

        for index in range(ind.getTreesLength()):
            tree = ind.getTree(index)
            if tree.status:
                tree.child.eval(state, thread, input, ind, problem)

        res = [None] * len(ind.getOutputRegisters())
        for d in range(len(ind.getOutputRegisters())):
            res[d] = ind.getRegisters()[ind.getOutputRegisters()[d]]
        return res

    def execute_outs_wrap(self, state:EvolutionState, thread:int, input:GPData, individual:'LGPIndividual4Class', problem:Problem):
        ind = individual  # LGPIndividual

        ind.resetRegisters(problem, self.initVal, ind)

        for index in range(ind.getTreesLength()):
            tree = ind.getTree(index)
            if tree.status:
                tree.child.eval(state, thread, input, ind, problem)

        if ind.IsWrap():
            for w in range(len(ind.getWrapper())):
                tree = ind.getWrapper()[w]
                tree.child.eval(state, thread, input, ind, problem)

        res = [None] * len(ind.getOutputRegisters())
        for d in range(len(ind.getOutputRegisters())):
            res[d] = ind.getRegisters()[ind.getOutputRegisters()[d]]
        return res

    def resetRegisters(self, problem, val, ind: 'LGPIndividual4Class'):
        for i in range(ind.getRegisters().__len__()):
            ind.setRegister(i, val)

    def execute(self, state:EvolutionState, thread:int, input:GPData, individual:'LGPIndividual4Class', problem:Problem):
        sys.stderr.write(
            "individuals for classification problems should not use this function "
            "to execute programs, try to use \"execute_outs(...)\"\n"
        )
        sys.exit(1)
        return 0

    def getAdjacencyTable(self, state:EvolutionState, start: int, end: int):
        ind = self  # LGPIndividual

        if start < 0 or start >= end or end > len(ind.getTreelist()):
            sys.stderr.write(
                "illegal arguments in getAdjacencyTable() of LGPIndividual4Graph\n"
            )
            sys.exit(1)

        res = []  # list of Pair<String, ArrayList<String>>

        for i in range(end - 1, start - 1, -1):
            tree = ind.getTreelist()[i]

            if not tree.status:
                continue

            check = str(tree.child.children[0])
            slibings = []

            for j in range(tree.child.children[0].expectedChildren()):
                node = tree.child.children[0].children[j]

                if (not isinstance(node, ReadRegisterGPNode)
                        and node.expectedChildren() == 0):
                    slibings.append(str(node))
                else:
                    slibings.append(None)
                    k = i - 1
                    while k >= 0:
                        if (tree.child.children[0].children[j].getIndex() ==
                                ind.getTreelist()[k].child.getIndex()):  # WriteRegisterGPNode index check
                            slibings[j] = str(ind.getTreelist()[k].child.children[0])
                            break
                        k -= 1
                    if k < 0:
                        # term_ind = tree.child.children[0].children[j].getIndex()
                        slibings[j] = None

            res.append((check, slibings))  # equivalent to Pair<String, ArrayList<String>>

        return res

    def makeGraphvizRule(self, outputRegs: list) -> str:
        ind = self  # LGPIndividual

        usedTerminals = [""] * ind.getNumRegs()
        for j in range(ind.getNumRegs()):
            usedTerminals[j] = str(self.initVal)

        ClassInputs = set()

        nodeSpec = ""
        for i in range(len(ind.getTreelist())):
            tree = ind.getTreelist()[i]

            if not tree.status:
                continue

            nodeSpec += (
                "" + str(i) + "[label=\"" + tree.child.children[0].toGraphvizString() + "\"];\n"
            )

        connection = ""
        notUsed = set(outputRegs)

        cntindex = [len(ind.getTreelist())]  # mutable int equivalent to AtomicInteger

        for i in range(len(ind.getTreelist()) - 1, -1, -1):
            tree = ind.getTreelist()[i]

            if not tree.status:
                continue

            connection += ind.makeGraphvizInstr(i, ClassInputs, usedTerminals, notUsed, cntindex)

        result = (
            "digraph g {\n"
            "nodesep=0.2;\n"
            "ranksep=0;\n"
            "node[fixedsize=true,width=1.3,height=0.6,fontsize=\"30\",fontname=\"times-bold\","
            "style=filled, fillcolor=lightgrey];\n"
            "edge[fontsize=\"25.0\",fontname=\"times-bold\"];\n"
            + nodeSpec
            + connection
            + "}\n"
        )

        return result

    # -------------------------------------------------------------------------
    # From LGPIndividual4Class (original)
    # -------------------------------------------------------------------------
    
    def wrapper(self, predict_list: list, target_list: list, state: EvolutionState, thread: int, problem: Problem) -> list:
        """This wrapper function mappes the raw outputs of the LGP individual to class labels using a linear classifier.
        predict_list: 2D list of raw outputs from the LGP individual, each element is a list of outputs from all registers.
        target_list: 2D list of true class labels corresponding to the inputs, each element is a list with one element (the class label).
        """

        MAX_SAMPLE = 128

        classprob = problem  # LGPClassification

        num_samples = len(predict_list)
        num_features = len(predict_list[0])
        num_classes = len(classprob.getClassLabels())
        class_labels = np.asarray(classprob.getClassLabels(), dtype=float)

        # Keep a mutable matrix mirroring predict_list so updates after each fitted classifier
        # are immediately visible to the next classifier, matching the Java flow.
        predict_np = np.asarray(predict_list, dtype=float)

        if num_samples > MAX_SAMPLE:
            indices = np.asarray(state.random[thread].sample(range(num_samples), MAX_SAMPLE), dtype=int)
        else:
            indices = np.arange(num_samples, dtype=int)

        target = np.asarray([target_list[i][0] for i in indices], dtype=float)

        wrapWeights = np.zeros((num_classes, num_features + 1), dtype=float)

        def build_design_matrix(cur_predict: np.ndarray, idx: np.ndarray) -> np.ndarray:
            sampled = cur_predict[idx, :num_features]
            design = np.ones((idx.shape[0], num_features + 1), dtype=float)
            design[:, 1:] = sampled
            return design

        def fit_one_label(design: np.ndarray, y_binary: np.ndarray) -> np.ndarray:
            # One linear classifier per class label (one-vs-rest).
            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                fit_intercept=False,
            )
            # Degenerate sampled targets can contain only one class.
            if np.unique(y_binary).shape[0] < 2:
                return np.zeros(design.shape[1], dtype=float)
            clf.fit(design, y_binary)
            return np.asarray(clf.coef_[0], dtype=float)

        def apply_weights(cur_predict: np.ndarray, w: np.ndarray, out_col: int):
            vals = w[0] + cur_predict[:, :num_features] @ w[1:]
            np.clip(vals, -1e6, 1e6, out=vals)
            cur_predict[:, out_col] = vals

        self.wraplist.clear()
        for label in range(num_classes):
            design = build_design_matrix(predict_np, indices)
            y_binary = (target == class_labels[label]).astype(int)
            w = fit_one_label(design, y_binary)

            instr = self.constructInstr(self.outputRegister[label], w.tolist())
            self.wraplist.append(instr)

            wrapWeights[label, :] = w

            # Update predict_list after each label fit, as in the Java implementation.
            apply_weights(predict_np, w, label)


        # output the updated prediction
        if num_samples <= MAX_SAMPLE:
            newpred = predict_np[:, :num_classes].tolist()
        else:
            newpred_np = np.zeros((num_samples, num_classes), dtype=float)

            used_mask = np.zeros(num_samples, dtype=bool)
            used_mask[indices] = True

            # already updated
            newpred_np[used_mask, :] = predict_np[used_mask, :num_classes]

            # recompute for non-updated samples using learned wrapWeights
            non_used_mask = ~used_mask
            if np.any(non_used_mask):
                non_used_pred = predict_np[non_used_mask, :num_features]
                for label in range(num_classes):
                    w = wrapWeights[label]
                    vals = w[0] + non_used_pred @ w[1:]
                    np.clip(vals, -1e6, 1e6, out=vals)
                    newpred_np[non_used_mask, label] = vals

            newpred = newpred_np.tolist()

        return newpred


    def copyLGPproperties(self, obj: 'LGPIndividual4Class'):
        """Copy LGP properties from another individual."""
        super().copyLGPproperties(obj)
        self.dataindex = obj.dataindex