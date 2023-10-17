from pymor.basic import *
from pymor.discretizers.builtin.cg import discretize_stationary_cg as discretizer
from pymor.analyticalproblems.elliptic import StationaryProblem
import numpy as np

set_log_levels({'pymor': 'INFO'})

domain = RectDomain(([0,0], [1,1]))
l = ExpressionFunction('100 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', dim_domain = 2)
parameters = Parameters({'reaction': 2})
diffusion = ConstantFunction(1,2)

diameter = 1/36  # comparable to original paper 
ei_snapshots = 12  # same as paper (creates 12x12 grid)
ei_size = 25  # maximum number of bases in EIM
rb_size = 25  # maximum number of bases in RBM
test_snapshots = 15 # same as paper (creates 15x15 grid)

nonlinear_reaction_coefficient = ConstantFunction(1,2)
test_nonlinearreaction = ExpressionFunction('reaction[0] * (exp(reaction[1] * u[0]) - 1) / reaction[1]', dim_domain = 1, parameters = parameters, variable = 'u')
test_nonlinearreaction_derivative = ExpressionFunction('reaction[0] * exp(reaction[1] * u[0])', dim_domain = 1, parameters = parameters, variable = 'u')
problem = StationaryProblem(domain = domain, rhs = l, diffusion = diffusion, nonlinear_reaction_coefficient = nonlinear_reaction_coefficient,
                            nonlinear_reaction = test_nonlinearreaction, nonlinear_reaction_derivative = test_nonlinearreaction_derivative)
grid, boundary_info = discretize_domain_default(problem.domain, diameter=diameter)
print('Anzahl Element', grid.size(0))
print('Anzahl DoFs', grid.size(2))
fom, data = discretizer(problem, diameter = diameter)

# cache_id = (f'pymordemos.nonlinear_reaction {ei_snapshots} {test_snapshots}')
fom.enable_caching('memory')

parameter_space = fom.parameters.space((0.01, 10))

# Training set
parameter_sample = parameter_space.sample_uniformly(ei_snapshots)
nonlin_op = fom.operator.operators[2]
evaluations = nonlin_op.range.empty()
for mu in parameter_sample:
    U = fom.solve(mu)
    evaluations.append(nonlin_op.apply(U, mu=mu))

# Test set
test_sample = parameter_space.sample_uniformly(test_snapshots)
u_max_norm = -1
for mu in test_sample:
    U = fom.solve(mu)
    u_norm = U.norm(fom.h1_0_semi_product)
    if u_norm > u_max_norm: u_max_norm = u_norm
u_max_norm = u_max_norm.item()

dofs, basis, data = ei_greedy(evaluations, copy=False,
                            error_norm=fom.l2_norm,
                            max_interpolation_dofs=ei_size)
ei_op = EmpiricalInterpolatedOperator(nonlin_op, dofs, basis, triangular=True)  #False for DEIM
new_ops = [ei_op if i == 2 else op for i, op in enumerate(fom.operator.operators)]
fom_ei = fom.with_(operator=fom.operator.with_(operators=new_ops))

print('RB generation ...')

reductor = StationaryRBReductor(fom_ei)

greedy_data = rb_greedy(fom, reductor, parameter_sample,
                        use_error_estimator=False,
                        error_norm=lambda U: np.max(fom.h1_0_semi_norm(U)),
                        max_extensions=rb_size)

rom = greedy_data['rom']

# print('Testing ROM...')

# max_err = -1
# for mu in test_sample:
#     u_fom = fom.solve(mu)
#     u_rom = rom.solve(mu)
#     this_diff = u_fom - reductor.reconstruct(u_rom)
#     this_err = this_diff.norm(fom.h1_0_semi_product)[0]
#     if this_err > max_err: max_err = this_err

# rel_error = max_err.item()/u_max_norm
print(f'RB size N: {len(reductor.bases["RB"])}')
# print(f'max. rel. error: {rel_error:2.5e}')

# test_sample = parameter_space.sample_uniformly(test_snapshots)
# abs_errs = []
# rel_errs = []
# max_abs_err = -1
# max_rel_err = -1
# max_abs_diff = None
# max_rel_diff = None
# for mu in test_sample:
#     u_fom = fom.solve(mu)
#     u_rom = rom.solve(mu)
#     this_diff = u_fom - reductor.reconstruct(u_rom)
#     this_abs_err = this_diff.norm(fom.h1_0_semi_product)[0]
#     this_rel_err = this_diff.norm(fom.h1_0_semi_product)[0]/u_fom.norm(fom.h1_0_semi_product)[0]
#     abs_errs.append(this_abs_err)
#     rel_errs.append(this_rel_err)
#     if this_abs_err > max_abs_err:
#         max_abs_err = this_abs_err
#         max_abs_diff = this_diff
#     if this_rel_err > max_rel_err:
#         max_rel_err = this_rel_err
#         max_rel_diff = this_diff

# print(f'max. abs. err.: {max_abs_err:2.5e}')
# print(f'max. rel. err.: {max_rel_err:2.5e}')

# fom.visualize((max_abs_diff, max_rel_diff), legend = ('Maximum Absolute Test Error', 'Maximum Relative Test Error'), separate_colorbars=True)