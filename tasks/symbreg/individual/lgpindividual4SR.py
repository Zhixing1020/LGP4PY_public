from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from src.ec.evolution_state import EvolutionState
from src.lgp.individual.lgp_individual import LGPIndividual, AtomicInteger
import numpy as np
from sklearn.linear_model import Ridge
from typing import override
from src.ec import *
from tasks.problem import Problem

class LGPIndividual4SR(LGPIndividual):

    def __init__(self):
        super().__init__()
        self.dataindex = 0

    @override
    def execute(self, state, thread, input, individual, problem, with_wrap:bool = False):
        return super().execute(state, thread, input, individual, problem, with_wrap)
    
    def postExecution(self, state, thread):
        return
    
    def getDataIndex(self): return self.dataindex

    def setDataIndex(self, index:int):
        self.dataindex = index

    @override
    def makeGraphvizRule(self, outputRegs: list[int]) -> str:
        """
        Convert the LGPIndividual into a Graphviz-compatible string,
        visualizing only effective instructions and connections.
        """
        # collect terminal names
        usedTerminals = [str(self.initVal) for _ in range(self.getNumRegs())]
        
        # set of seen terminals (used for deduplication)
        SRInputs = set()

        # specify nodes (effective instructions)
        nodeSpec = ""
        for i, tree in enumerate(self.getTreelist()):
            if not tree.status:
                continue
            nodeSpec += f'{i}[label="{tree.child.children[0].toGraphvizString()}"];\n'

        # prepare to collect connections
        connection = ""
        notUsed = set(outputRegs)
        cntindex = AtomicInteger(len(self.getTreelist()))

        for i in reversed(range(len(self.getTreelist()))):
            tree = self.getTree(i)
            if not tree.status:
                continue
            connection += self.makeGraphvizInstr(i, SRInputs, usedTerminals, notUsed, cntindex)

        # assemble graph
        result = (
            "digraph g {\n"
            "nodesep=0.2;\n"
            "ranksep=0;\n"
            "node[fixedsize=true,width=1.3,height=0.6,fontsize=\"30\",fontname=\"times-bold\",style=filled, fillcolor=lightgrey];\n"
            "edge[fontsize=\"25.0\",fontname=\"times-bold\"];\n"
            f"{nodeSpec}"
            f"{connection}"
            "}\n"
        )

        return result

    def wrapper(self, predict_array: np.ndarray, target_array: np.ndarray, 
            state: EvolutionState, thread: int, problem: Problem):
        
        #predict_array.shape = (N, F) where N = number of instances (samples) F = number of features
        #target_array.shape = (N, O) where O = number of outputs (targets)

        MAX_SAMPLE = self.wrap_max_sample

        num_samples, num_features = predict_array.shape
        num_outputs = target_array.shape[1]
        sample_size = min(num_samples, MAX_SAMPLE)

        # sample indices
        if num_samples > MAX_SAMPLE:
            indices = np.array(
                # state.random[thread].randint(0, num_samples - 1)
                # for _ in range(MAX_SAMPLE)
                state.random[thread].sample(range(num_samples), MAX_SAMPLE)
            )
        else:
            indices = np.arange(sample_size)

        # sampled predictors and targets
        predict = predict_array[indices, :]         # shape (sample_size, num_features)
        self.wraplist.clear()

        for tar in range(num_outputs):
            # build target vector for this output
            target = target_array[indices, tar]     # shape (sample_size,)

            # fit linear regression
            lr = Ridge(alpha=0.1, solver="lsqr", fit_intercept=True, max_iter=1000)
            lr.fit(predict, target)

            # combine intercept + coefficients
            W = np.concatenate(([lr.intercept_], lr.coef_))

            # construct instruction
            instr = self.constructInstr(self.outputRegister[tar], W)
            self.wraplist.append(instr)

            # update all predictions inplace
            tmp = W[0] + predict_array @ W[1:]   # shape (num_samples,)
            tmp = np.clip(tmp, -1e6, 1e6)
            predict_array[:, tar] = tmp

        # return updated predictions
        return predict_array.copy()
    
    def getWrapNorm(self, predict_array: np.ndarray, target_array: np.ndarray,
                    state: EvolutionState, thread: int, problem: Problem) -> float:

        
        norm_result = 0.0

        num_samples, num_features = predict_array.shape
        # MAX_SAMPLE = max(int(self.wrap_max_sample / num_features), 100)
        # num_outputs = target_array.shape[1]
        # sample_size = min(num_samples, MAX_SAMPLE)

        # # sample indices
        # if num_samples > MAX_SAMPLE:
        #     indices = np.array(
        #         # state.random[thread].randint(0, num_samples - 1)
        #         # for _ in range(MAX_SAMPLE)
        #         state.random[thread].sample(range(num_samples), MAX_SAMPLE)
        #     )
        # else:
        #     indices = np.arange(sample_size)

        # # sampled predictors and targets
        # predict = predict_array[indices, :]         # shape (sample_size, num_features)
        
        # # fit linear regression
        # lr = Ridge(alpha=0.1, solver="lsqr", fit_intercept=True, max_iter=100)
        
        # for tar in range(num_outputs):
        #     # build target vector for this output
        #     target = target_array[indices, tar]     # shape (sample_size,)

        #     for p in range(num_features):
        #         lr.fit(predict[:, p:p+1], target)
        #         y_pred = lr.predict(predict[:, p:p+1])
        #         norm_result += (1 - r2_score(target, y_pred))


        # return norm_result / (num_outputs * num_features)


        unique_cols = np.unique(predict_array, axis=1)
        num_unique = unique_cols.shape[1]

        return (num_features - num_unique) / num_features
    
    def copyLGPproperties(self, obj: 'LGPIndividual4SR'):
        """Copy LGP properties from another individual."""
        super().copyLGPproperties(obj)
        self.dataindex = obj.dataindex