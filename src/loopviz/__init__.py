"""loopviz: a cyclic playlist operator whose free subspace is optimized for aesthetics.

Pipeline: youtube links -> audio matrix X -> factored operator A = A0 + Z P_perp
-> rendered image of A -> aesthetic metric loss -> ES over Z/render params
-> pairwise comparisons -> Bradley-Terry fit of personal metric weights.
"""

__version__ = "0.1.0"
