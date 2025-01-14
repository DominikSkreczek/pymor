from pymor.basic import *
from functools import partial
import numpy as np
from time import perf_counter
import trust_region_method
from scipy.sparse import coo_matrix, csc_matrix, dia_matrix
from pymor.algorithms.newton import newton
from discretize_cg_with_nonlinear_reactionoperator import discretize_stationary_cg as discretizer
from discretize_cg_with_nonlinear_reactionoperator import element_NonlinearReactionOperator
from discretize_cg_with_nonlinear_reactionoperator import quadratic_functional, element_quadratic_functional
import stationary_problem
from scipy.optimize import linprog

set_log_levels({'pymor': 'INFO'})


domain = RectDomain(([0,0], [1,1]))
l = ExpressionFunction('100 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', dim_domain = 2)
parameters = Parameters({'reaction': 2})
diffusion = ConstantFunction(1,2)

diameter = 1/40

nonlinear_reaction_coefficient = ConstantFunction(1,2)
test_nonlinearreaction = ExpressionFunction('reaction[0] * (exp(reaction[1] * u[0]) - 1) / reaction[1]', dim_domain = 1, parameters = parameters, variable = 'u')
test_nonlinearreaction_derivative = ExpressionFunction('reaction[0] * exp(reaction[1] * u[0])', dim_domain = 1, parameters = parameters, variable = 'u')
problem = stationary_problem.StationaryProblem(domain = domain, rhs = l, diffusion = diffusion,  nonlinear_reaction_coefficient = nonlinear_reaction_coefficient,
                                               nonlinear_reaction = test_nonlinearreaction, nonlinear_reaction_derivative = test_nonlinearreaction_derivative)
grid, boundary_info = discretize_domain_default(problem.domain, diameter=diameter)
print('Anzahl Element', grid.size(0))
print('Anzahl DoFs', grid.size(2))
fom, data = discretizer(problem, diameter = diameter)
u= newton(fom.operator, fom.rhs.as_range_array(), mu = problem.parameters.parse([0.01, 0.01]))[0]
fom.visualize(u, title = 'cg')
# u= newton(fom.operator, fom.rhs.as_range_array(), mu = problem.parameters.parse([10, 10]))[0]
# fom.visualize(u, title = 'cg')
problem_fv = StationaryProblem(domain = domain, rhs = l, diffusion = diffusion, nonlinear_reaction = test_nonlinearreaction, nonlinear_reaction_derivative = test_nonlinearreaction_derivative)
fom_fv, data_fv = discretize_stationary_fv(problem_fv, diameter = diameter)
u_fv = fom_fv.solve([0.01, 0.01])
fom_fv.visualize(u_fv, title = 'fv')