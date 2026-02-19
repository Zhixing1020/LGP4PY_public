import numpy as np
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector

class norm2Q:
    """norm2Q: ||G_j - G_i||_2, G_j=theta_j * I"""

    @staticmethod
    def subQ_vec(a, b):
        """Vectorized version of subQ using NumPy."""
        # a and b are numpy arrays
        res = np.zeros_like(a, dtype=float)
        
        # Case 1: both >= 0
        both_pos = (a >= 0) & (b >= 0)
        res[both_pos] = a[both_pos] - b[both_pos]
        
        # Case 2: either > 0 (and not handled by Case 1)
        either_pos = (~both_pos) & ((a > 0) | (b > 0))
        res[either_pos] = 1.0
        
        return res

    @staticmethod
    def Q(g1:GenoVector, g2:GenoVector):

        if g1.length != g2.length:
            raise ValueError(f"Inconsistent GenoVector when calculating Q: {g1.length} vs {g2.length}")

        # Ensure they are numpy arrays
        G1 = np.array(g1.G)
        G2 = np.array(g2.G)
        
        # Calculate mask for where both are negative (the 'break' condition)
        # In Python vectorization, we mask the valid indices
        valid = ~((G1 < 0) & (G2 < 0))
        
        diffs = norm2Q.subQ_vec(G1[valid], G2[valid])
        return np.sum(diffs**2)

    @staticmethod
    def pQpI(GA1_G:list[GenoVector], theta1, GA2_G:list[GenoVector], theta2, gj, gi, l):
        """
        get the partial Q partial I_l
		gj, gi: indexes of individuals,  l: index of index list 

        Note: Passing raw arrays (GA1_G) instead of objects is faster.
        """
        g1 = GA1_G[gj]
        g2 = GA2_G[gi]
        
        valid = ~((g1 < 0) & (g2 < 0))
        
        sub = norm2Q.subQ_vec(g1[valid], g2[valid])
        # theta shape is assumed to be (individual, gene_k, index_l)
        diff_theta = theta1[gj][valid, l] - theta2[gi][valid, l]
        
        return np.sum(sub * 2 * diff_theta)