# CPWL
Fitting of Continuous Piecewise Linear (CPWL) functions

The code requires the package mathopt.

The LP solver GLOP is open-source and included in mathopt.

For the MILP solver, Gurobi is the default solver.
If you do not have a Gurobi license, the MILP solver GSCIP is open-source and included in mathopt.

The main function is "fast_MILP(data, err)"
