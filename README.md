# CPWL
Fitting of Continuous Piecewise Linear (CPWL) functions

The code requires the package mathopt.

The LP solver GLOP is open-source and included in mathopt.

For the MILP solver, Gurobi is the default solver.
If you do not have a Gurobi license, the MILP solver GSCIP is open-source and included in mathopt.

The main function is "fast_MILP(data, err)":
  "data" should be a (N,2) numpy array that represents the dataset to be fitted.
  "err" should be a float that represents the maximum fitting error between the linear segments and the y-value of the dataset.
