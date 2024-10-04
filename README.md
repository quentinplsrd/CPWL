# 1D CPWL
This python code is an implementation of the LP models, MILP models, and algorithms described in the article
"Piecewise linear approximation with minimum number of linear segments and minimum error: A fast approach to tighten and warm start the hierarchical mixed integer formulation"
https://doi.org/10.1016/j.ejor.2023.11.017

The models and algorithm allows for a fast calculation of the optimal continuous piecewise linear (CPWL) approximation of a 2D dataset.

The code requires the Google OR-Tools python libraries and modules: https://pypi.org/project/ortools/

The LP solver GLOP is open-source and included in mathopt.

For the MILP solver, Gurobi is the default solver.
If you do not have a Gurobi license, the MILP solver GSCIP is open-source and included in mathopt.

The main function is "fast_MILP(data, err)":
"data" should be a (N,2) numpy array that represents the dataset to be fitted.
"err" should be a float that represents the maximum fitting error between the linear segments and the y-value of the dataset.
