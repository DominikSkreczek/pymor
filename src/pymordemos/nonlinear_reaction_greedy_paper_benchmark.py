from time import perf_counter
from pymor.basic import *
from pymor.discretizers.builtin.cg import discretize_stationary_cg as discretizer
from pymor.analyticalproblems.elliptic import StationaryProblem
import numpy as np

def benchmark_problem(fom, parameter_sample, test_sample, M, N):

    # Training set
    nonlin_op = fom.operator.operators[2]
    evaluations = nonlin_op.range.empty()
    for mu in parameter_sample:
        U = fom.solve(mu)
        evaluations.append(nonlin_op.apply(U, mu=mu))

    # Test set
    # u_max_norm = -1
    # for mu in test_sample:
    #     U = fom.solve(mu)
    #     u_norm = U.norm(fom.h1_0_semi_product)
    #     if u_norm > u_max_norm: u_max_norm = u_norm
    # u_max_norm = u_max_norm.item()

    tic = perf_counter()

    dofs, basis, data = ei_greedy(evaluations, copy=False,
                                error_norm=fom.l2_norm,
                                max_interpolation_dofs=M)
    ei_op = EmpiricalInterpolatedOperator(nonlin_op, dofs, basis, triangular=True)  #False for DEIM
    new_ops = [ei_op if i == 2 else op for i, op in enumerate(fom.operator.operators)]
    fom_ei = fom.with_(operator=fom.operator.with_(operators=new_ops))

    print('RB generation ...')

    reductor = StationaryRBReductor(fom_ei)

    greedy_data = rb_greedy(fom, reductor, parameter_sample,
                            use_error_estimator=False,
                            error_norm=lambda U: np.max(fom.h1_0_semi_norm(U)),
                            max_extensions=N)

    rom = greedy_data['rom']

    toc = perf_counter()
    calctime = toc - tic

    print('Testing ROM...')

    max_err = -1
    u_max_norm = -1
    for mu in test_sample:
        u_fom = fom.solve(mu)
        u_rom = rom.solve(mu)
        this_diff = u_fom - reductor.reconstruct(u_rom)
        this_err = this_diff.norm(fom.h1_0_semi_product)[0]
        if this_err > max_err: max_err = this_err
        u_norm = U.norm(fom.h1_0_semi_product)
        if u_norm > u_max_norm: u_max_norm = u_norm

    rel_err = max_err.item()/u_max_norm.item()

    print(f'\nBenchmark done for M={M} and N={N}.\n')

    return max_err, rel_err, calctime


set_log_levels({'pymor': 'INFO'})

domain = RectDomain(([0,0], [1,1]))
l = ExpressionFunction('100 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', dim_domain = 2)
parameters = Parameters({'reaction': 2})
diffusion = ConstantFunction(1,2)

diameter = 1/36  # comparable to original paper 
ei_snapshots = 12  # same as paper (creates 12x12 grid)
test_snapshots = 15 # same as paper (creates 15x15 grid)
ei_sizes = [25] # , 10, 15, 20, 25]  # maximum number of bases in EIM
Nmax = [22]# 10, 16, 17, 22]  # corresponding maximum number of bases in RBM

assert(len(ei_sizes)==len(Nmax))

nonlinear_reaction_coefficient = ConstantFunction(1,2)
test_nonlinearreaction = ExpressionFunction('reaction[0] * (exp(reaction[1] * u[0]) - 1) / reaction[1]', dim_domain = 1, parameters = parameters, variable = 'u')
test_nonlinearreaction_derivative = ExpressionFunction('reaction[0] * exp(reaction[1] * u[0])', dim_domain = 1, parameters = parameters, variable = 'u')
problem = StationaryProblem(domain = domain, rhs = l, diffusion = diffusion, nonlinear_reaction_coefficient = nonlinear_reaction_coefficient,
                            nonlinear_reaction = test_nonlinearreaction, nonlinear_reaction_derivative = test_nonlinearreaction_derivative)
grid, boundary_info = discretize_domain_default(problem.domain, diameter=diameter)
print('Anzahl Element', grid.size(0))
print('Anzahl DoFs', grid.size(2))
fom, data = discretizer(problem, diameter = diameter)

cache_id = (f'pymordemos.nonlinear_reaction {ei_snapshots} {test_snapshots}')
fom.enable_caching('persistent',cache_id)

parameter_space = fom.parameters.space((0.01, 10))
parameter_sample = parameter_space.sample_uniformly(ei_snapshots)
test_sample = parameter_space.sample_uniformly(test_snapshots)

abs_errors = np.empty((len(ei_sizes), np.max(Nmax)+1))
abs_errors[:] = np.nan
rel_errors = np.copy(abs_errors)
calctimes = np.copy(abs_errors)

for i in range(len(ei_sizes)):
    for N in range(1,Nmax[i]+1):

        M = ei_sizes[i]

        this_abs_err, this_rel_err, this_calctime = benchmark_problem(fom, parameter_sample, test_sample, M, N)
        abs_errors[i,N] = this_abs_err
        rel_errors[i,N] = this_rel_err
        calctimes[i,N] = this_calctime

np.savetxt(f"benchmark_M{ei_sizes[0]}_abs_errors.csv", abs_errors, delimiter=",")
np.savetxt(f"benchmark_M{ei_sizes[0]}_rel_errors.csv", rel_errors, delimiter=",")
np.savetxt(f"benchmark_M{ei_sizes[0]}_calctimes.csv", calctimes, delimiter=",")
#np.savetxt(f"benchmark_M.csv", ei_sizes, delimiter=",")



