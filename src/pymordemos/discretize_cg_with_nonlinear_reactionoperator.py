# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module provides some operators for continuous finite element discretizations."""

from functools import partial

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix

from pymor.algorithms.preassemble import preassemble as preassemble_
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper, ImplicitEulerTimeStepper
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import Function, ConstantFunction, LincombFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.discretizers.builtin.domaindiscretizers.default import discretize_domain_default
from pymor.discretizers.builtin.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.discretizers.builtin.grids.referenceelements import line, triangle, square
from pymor.discretizers.builtin.gui.visualizers import PatchVisualizer, OnedVisualizer
from pymor.models.basic import StationaryModel, InstationaryModel
from pymor.operators.constructions import LincombOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixBasedOperator, NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


LagrangeShapeFunctions = {
    line: {1: [lambda X: 1 - X[..., 0],
               lambda X: X[..., 0]]},
    square: {1: [lambda X: (1 - X[..., 0]) * (1 - X[..., 1]),
                 lambda X: (1 - X[..., 1]) * (X[..., 0]),
                 lambda X:     (X[..., 0]) * (X[..., 1]),
                 lambda X:     (X[..., 1]) * (1 - X[..., 0])]},
    triangle: {1: [lambda X: 1 - X[..., 0] - X[..., 1],
                   lambda X: X[..., 0],
                   lambda X: X[..., 1]]},
}

LagrangeShapeFunctionsGrads = {
    line: {1: np.array(([-1.],
                        [1., ]))},
    square: {1: lambda X: np.array(([X[..., 1] - 1., X[..., 0] - 1.],
                                    [1. - X[..., 1], - X[..., 0]],
                                    [X[..., 1], X[..., 0]],
                                    [-X[..., 1], 1. - X[..., 0]]))},
    triangle: {1: np.array(([-1., -1.],
                            [1., 0.],
                            [0., 1.]))},
}


def CGVectorSpace(grid, id='STATE'):
    return NumpyVectorSpace(grid.size(grid.dim), id)


class L2ProductFunctionalP1(NumpyMatrixBasedOperator):
    """Linear functional representing the inner product with an L2-|Function|.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product.
    dirichlet_clear_dofs
        If `True`, set dirichlet boundary DOFs to zero.
    boundary_info
        |BoundaryInfo| determining the Dirichlet boundaries in case
        `dirichlet_clear_dofs` is set to `True`.
    name
        The name of the functional.
    """

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, grid, function, dirichlet_clear_dofs=False, boundary_info=None, name=None):
        assert grid.reference_element(0) in {line, triangle}
        assert function.shape_range == ()
        assert not dirichlet_clear_dofs or boundary_info
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at element centers
        F = self.function(g.centers(0), mu=mu)

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=1)
        SF = LagrangeShapeFunctions[g.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF_INTS = np.einsum('e,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFs
        # FIXME This implementation is horrible, find a better way!
        SF_I = g.subentities(0, g.dim).ravel()
        I = coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        if self.dirichlet_clear_dofs and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            I[DI] = 0

        return I.reshape((-1, 1))


class BoundaryL2ProductFunctional(NumpyMatrixBasedOperator):
    """Linear functional representing the inner product with an L2-|Function| on the boundary.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product.
    boundary_type
        The type of domain boundary (e.g. 'neumann') on which to assemble the functional.
        If `None` the functional is assembled over the whole boundary.
    dirichlet_clear_dofs
        If `True`, set dirichlet boundary DOFs to zero.
    boundary_info
        If `boundary_type` is specified or `dirichlet_clear_dofs` is `True`, the
        |BoundaryInfo| determining which boundary entity belongs to which physical boundary.
    name
        The name of the functional.
    """

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_type=None, dirichlet_clear_dofs=False, boundary_info=None, name=None):
        assert grid.reference_element(0) in {line, triangle, square}
        assert function.shape_range == ()
        assert not (boundary_type or dirichlet_clear_dofs) or boundary_info
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        NI = bi.boundaries(self.boundary_type, 1) if self.boundary_type else g.boundaries(1)
        if g.dim == 1:
            I = np.zeros(self.range.dim)
            I[NI] = self.function(g.centers(1)[NI])
        else:
            F = self.function(g.centers(1)[NI], mu=mu)
            q, w = line.quadrature(order=1)
            # remove last dimension of q, as line coordinates are one dimensional
            q = q[:, 0]
            SF = np.array([1 - q, q])
            SF_INTS = np.einsum('e,pi,e,i->ep', F, SF, g.integration_elements(1)[NI], w).ravel()
            SF_I = g.subentities(1, 2)[NI].ravel()
            I = coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        if self.dirichlet_clear_dofs and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            I[DI] = 0

        return I.reshape((-1, 1))


class BoundaryDirichletFunctional(NumpyMatrixBasedOperator):
    """Linear functional for enforcing Dirichlet boundary values.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    dirichlet_data
        |Function| providing the Dirichlet boundary values.
    boundary_info
        |BoundaryInfo| determining the Dirichlet boundaries.
    name
        The name of the functional.
    """

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, grid, dirichlet_data, boundary_info, name=None):
        assert grid.reference_element(0) in {line, triangle, square}
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        I = np.zeros(self.range.dim)
        DI = bi.dirichlet_boundaries(g.dim)
        I[DI] = self.dirichlet_data(g.centers(g.dim)[DI], mu=mu)

        return I.reshape((-1, 1))


class L2ProductFunctionalQ1(NumpyMatrixBasedOperator):
    """Bilinear functional representing the inner product with an L2-|Function|.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product.
    dirichlet_clear_dofs
        If `True`, set dirichlet boundary DOFs to zero.
    boundary_info
        |BoundaryInfo| determining the Dirichlet boundaries in case
        `dirichlet_clear_dofs` is set to `True`.
    name
        The name of the functional.
    """

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, grid, function, dirichlet_clear_dofs=False, boundary_info=None, name=None):
        assert grid.reference_element(0) in {square}
        assert function.shape_range == ()
        assert not dirichlet_clear_dofs or boundary_info
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at all quadrature points ->
        #   shape = (g.size(0), number of quadrature points)
        F = self.function(g.centers(0), mu=mu)

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=1)
        SF = LagrangeShapeFunctions[g.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF_INTS = np.einsum('e,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFs
        # FIXME This implementation is horrible, find a better way!
        SF_I = g.subentities(0, g.dim).ravel()
        I = coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        if self.dirichlet_clear_dofs and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            I[DI] = 0

        return I.reshape((-1, 1))


class L2ProductP1(NumpyMatrixBasedOperator):
    """|Operator| representing the L2-product between linear finite element functions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the product.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    dirichlet_clear_rows
        If `True`, set the rows of the system matrix corresponding to Dirichlet boundary
        DOFs to zero.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise, if either `dirichlet_clear_rows` or `dirichlet_clear_columns` is
        `True`, the diagonal entries are set to one.
    coefficient_function
        Coefficient |Function| for product with `shape_range == ()`.
        If `None`, constant one is assumed.
    solver_options
        The |solver_options| for the operator.
    name
        The name of the product.
    """

    sparse = True

    def __init__(self, grid, boundary_info, dirichlet_clear_rows=True, dirichlet_clear_columns=False,
                 dirichlet_clear_diag=False, coefficient_function=None, solver_options=None, name=None):
        assert grid.reference_element in (line, triangle)
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate the shape functions on the quadrature points
        q, w = g.reference_element.quadrature(order=2)
        SF = LagrangeShapeFunctions[g.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))

        self.logger.info('Integrate the products of the shape functions on each element')
        # -> shape = (g.size(0), number of shape functions ** 2)
        if self.coefficient_function is not None:
            C = self.coefficient_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('iq,jq,q,e,e->eij', SF, SF, w, g.integration_elements(0), C).ravel()
            del C
        else:
            SF_INTS = np.einsum('iq,jq,q,e->eij', SF, SF, w, g.integration_elements(0)).ravel()

        del SF

        self.logger.info('Determine global dofs ...')
        SF_I0 = np.repeat(g.subentities(0, g.dim), g.dim + 1, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, g.dim + 1]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            if self.dirichlet_clear_rows:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I1], 0, SF_INTS)
            if not self.dirichlet_clear_diag and (self.dirichlet_clear_rows or self.dirichlet_clear_columns):
                SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(g.dim).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(g.dim)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(g.dim)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        del SF_INTS, SF_I0, SF_I1
        A.eliminate_zeros()
        A = csc_matrix(A).copy()  # See DiffusionOperatorP1 for why copy() is necessary

        return A


class L2ProductQ1(NumpyMatrixBasedOperator):
    """|Operator| representing the L2-product between bilinear finite element functions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the product.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    dirichlet_clear_rows
        If `True`, set the rows of the system matrix corresponding to Dirichlet boundary
        DOFs to zero.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise, if either `dirichlet_clear_rows` or `dirichlet_clear_columns`
        is `True`, the diagonal entries are set to one.
    coefficient_function
        Coefficient |Function| for product with `shape_range == ()`.
        If `None`, constant one is assumed.
    solver_options
        The |solver_options| for the operator.
    name
        The name of the product.
    """

    sparse = True

    def __init__(self, grid, boundary_info, dirichlet_clear_rows=True, dirichlet_clear_columns=False,
                 dirichlet_clear_diag=False, coefficient_function=None, solver_options=None, name=None):
        assert grid.reference_element in {square}
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate the shape functions on the quadrature points
        q, w = square.quadrature(order=2)
        SF = LagrangeShapeFunctions[g.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))

        self.logger.info('Integrate the products of the shape functions on each element')
        # -> shape = (g.size(0), number of shape functions ** 2)
        if self.coefficient_function is not None:
            C = self.coefficient_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('iq,jq,q,e,e->eij', SF, SF, w, g.integration_elements(0), C).ravel()
            del C
        else:
            SF_INTS = np.einsum('iq,jq,q,e->eij', SF, SF, w, g.integration_elements(0)).ravel()

        del SF

        self.logger.info('Determine global dofs ...')
        SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            if self.dirichlet_clear_rows:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I1], 0, SF_INTS)
            if not self.dirichlet_clear_diag and (self.dirichlet_clear_rows or self.dirichlet_clear_columns):
                SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(g.dim).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(g.dim)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(g.dim)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        del SF_INTS, SF_I0, SF_I1
        A.eliminate_zeros()
        A = csc_matrix(A).copy()  # See DiffusionOperatorP1 for why copy() is necessary

        return A


class DiffusionOperatorP1(NumpyMatrixBasedOperator):
    """Diffusion |Operator| for linear finite elements.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ d(x) ∇ u(x) ]

    The function `d` can be scalar- or matrix-valued.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    diffusion_function
        The |Function| `d(x)` with `shape_range == ()` or
        `shape_range = (grid.dim, grid.dim)`. If `None`, constant one is
        assumed.
    diffusion_constant
        The constant `c`. If `None`, `c` is set to one.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise they are set to one.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {triangle, line}, 'A simplicial grid is expected!'
        assert diffusion_function is None \
            or (isinstance(diffusion_function, Function)
                and diffusion_function.dim_domain == grid.dim
                and diffusion_function.shape_range == ()
                or diffusion_function.shape_range == (grid.dim,) * 2)
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # gradients of shape functions
        SF_GRAD = LagrangeShapeFunctionsGrads[g.reference_element][1]

        self.logger.info('Calculate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)

        self.logger.info('Calculate all local scalar products between gradients ...')
        if self.diffusion_function is not None and self.diffusion_function.shape_range == ():
            D = self.diffusion_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('epi,eqi,e,e->epq', SF_GRADS, SF_GRADS, g.volumes(0), D).ravel()
            del D
        elif self.diffusion_function is not None:
            D = self.diffusion_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('epi,eqj,e,eij->epq', SF_GRADS, SF_GRADS, g.volumes(0), D).ravel()
            del D
        else:
            SF_INTS = np.einsum('epi,eqi,e->epq', SF_GRADS, SF_GRADS, g.volumes(0)).ravel()

        del SF_GRADS

        if self.diffusion_constant is not None:
            SF_INTS *= self.diffusion_constant

        self.logger.info('Determine global dofs ...')
        SF_I0 = np.repeat(g.subentities(0, g.dim), g.dim + 1, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, g.dim + 1]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I1], 0, SF_INTS)

            if not self.dirichlet_clear_diag:
                SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(g.dim).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(g.dim)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(g.dim)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        del SF_INTS, SF_I0, SF_I1
        A.eliminate_zeros()
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class DiffusionOperatorQ1(NumpyMatrixBasedOperator):
    """Diffusion |Operator| for bilinear finite elements.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ d(x) ∇ u(x) ]

    The function `d` can be scalar- or matrix-valued.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    diffusion_function
        The |Function| `d(x)` with `shape_range == ()` or
        `shape_range = (grid.dim, grid.dim)`. If `None`, constant one is
        assumed.
    diffusion_constant
        The constant `c`. If `None`, `c` is set to one.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise they are set to one.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {square}, 'A square grid is expected!'
        assert diffusion_function is None \
            or (isinstance(diffusion_function, Function)
                and diffusion_function.dim_domain == grid.dim
                and diffusion_function.shape_range == ()
                or diffusion_function.shape_range == (grid.dim,) * 2)
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # gradients of shape functions
        q, w = g.reference_element.quadrature(order=2)
        SF_GRAD = LagrangeShapeFunctionsGrads[g.reference_element][1]
        SF_GRAD = SF_GRAD(q)

        self.logger.info('Calculate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pjc->epic', g.jacobian_inverse_transposed(0), SF_GRAD)

        self.logger.info('Calculate all local scalar products between gradients ...')
        if self.diffusion_function is not None and self.diffusion_function.shape_range == ():
            D = self.diffusion_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('epic,eqic,c,e,e->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D
        elif self.diffusion_function is not None:
            D = self.diffusion_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('epic,eqjc,c,e,eij->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D
        else:
            SF_INTS = np.einsum('epic,eqic,c,e->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0)).ravel()

        del SF_GRADS

        if self.diffusion_constant is not None:
            SF_INTS *= self.diffusion_constant

        self.logger.info('Determine global dofs ...')

        SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I1], 0, SF_INTS)

            if not self.dirichlet_clear_diag:
                SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(g.dim).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(g.dim)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(g.dim)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        del SF_INTS, SF_I0, SF_I1
        A.eliminate_zeros()
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class AdvectionOperatorP1(NumpyMatrixBasedOperator):
    """Linear advection |Operator| for linear finite elements.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ v(x) u(x) ]

    The function `v` has to be vector-valued.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    advection_function
        The |Function| `v(x)` with `shape_range = (grid.dim, )`.
        If `None`, constant one is assumed.
    advection_constant
        The constant `c`. If `None`, `c` is set to one.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise they are set to one.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, advection_function=None, advection_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {triangle, line}, 'A simplicial grid is expected!'

        advection_function = advection_function or ConstantFunction(np.ones((grid.dim,)), grid.dim)
        assert isinstance(advection_function, Function)
        assert advection_function.dim_domain == grid.dim
        assert advection_function.shape_range == (grid.dim,)
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        q, w = g.reference_element.quadrature(order=1)
        SF = LagrangeShapeFunctions[g.reference_element][1]

        self.logger.info('Calculate gradients of shape functions transformed by reference map ...')
        SF_GRAD = LagrangeShapeFunctionsGrads[g.reference_element][1]
        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)
        # SF_GRADS(element, function, component)

        SFQ = np.array(tuple(f(q) for f in SF))
        # SFQ(function, quadraturepoint)

        self.logger.info('Calculate all local scalar products between gradients ...')
        D = self.advection_function(self.grid.centers(0), mu=mu)
        SF_INTS = - np.einsum('pc,eqi,c,e,ei->eqp', SFQ, SF_GRADS, w, g.integration_elements(0), D).ravel()
        del D
        del SF_GRADS

        if self.advection_constant is not None:
            SF_INTS *= self.advection_constant

        self.logger.info('Determine global dofs ...')
        SF_I0 = np.repeat(g.subentities(0, g.dim), g.dim + 1, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, g.dim + 1]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I1], 0, SF_INTS)

            if not self.dirichlet_clear_diag:
                SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(g.dim).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(g.dim)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(g.dim)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        del SF_INTS, SF_I0, SF_I1
        A.eliminate_zeros()
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class AdvectionOperatorQ1(NumpyMatrixBasedOperator):
    """Linear advection |Operator| for bilinear finite elements.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ v(x) u(x) ]

    The function `v` has to be vector-valued.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    advection_function
        The |Function| `v(x)` with `shape_range = (grid.dim, )`.
        If `None`, constant one is assumed.
    advection_constant
        The constant `c`. If `None`, `c` is set to one.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise they are set to one.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, advection_function=None, advection_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {square}, 'A square grid is expected!'
        assert advection_function is None \
            or (isinstance(advection_function, Function)
                and advection_function.dim_domain == grid.dim
                and advection_function.shape_range == (grid.dim,))
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        self.logger.info('Calculate gradients of shape functions transformed by reference map ...')
        q, w = g.reference_element.quadrature(order=2)
        SF_GRAD = LagrangeShapeFunctionsGrads[g.reference_element][1](q)
        SF_GRADS = np.einsum('eij,pjc->epic', g.jacobian_inverse_transposed(0), SF_GRAD)
        # SF_GRADS(element,function,component,quadraturepoint)

        SF = LagrangeShapeFunctions[g.reference_element][1]
        SFQ = np.array(tuple(f(q) for f in SF))
        # SFQ(function, quadraturepoint)

        self.logger.info('Calculate all local scalar products between gradients ...')

        D = self.advection_function(self.grid.centers(0), mu=mu)
        SF_INTS = - np.einsum('pc,eqic,c,e,ei->eqp', SFQ, SF_GRADS, w, g.integration_elements(0), D).ravel()
        del D
        del SF_GRADS

        if self.advection_constant is not None:
            SF_INTS *= self.advection_constant

        self.logger.info('Determine global dofs ...')

        SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I1], 0, SF_INTS)

            if not self.dirichlet_clear_diag:
                SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(g.dim).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(g.dim)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(g.dim)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        del SF_INTS, SF_I0, SF_I1
        A.eliminate_zeros()
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class RobinBoundaryOperator(NumpyMatrixBasedOperator):
    """Robin boundary |Operator| for linear finite elements.

    The operator represents the contribution of Robin boundary conditions to the
    stiffness matrix, where the boundary condition is supposed to be given in the
    form ::

        -[ d(x) ∇u(x) ] ⋅ n(x) = c(x) (u(x) - g(x))

    `d` and `n` are the diffusion function (see :class:`DiffusionOperatorP1`) and
    the unit outer normal in `x`, while `c` is the (scalar) Robin parameter
    function and `g` is the (also scalar) Robin boundary value function.

    Parameters
    ----------
    grid
        The |Grid| over which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    robin_data
        Tuple providing two |Functions| that represent the Robin parameter and boundary
        value function. If `None`, the resulting operator is zero.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, robin_data=None, solver_options=None, name=None):
        assert robin_data is None or (isinstance(robin_data, tuple) and len(robin_data) == 2)
        assert robin_data is None or all([isinstance(f, Function)
                                          and f.dim_domain == grid.dim
                                          and (f.shape_range == ()
                                               or f.shape_range == (grid.dim,))
                                          for f in robin_data])
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        if g.dim > 2:
            raise NotImplementedError

        if bi is None or not bi.has_robin or self.robin_data is None:
            return coo_matrix((g.size(g.dim), g.size(g.dim))).tocsc()

        RI = bi.robin_boundaries(1)
        if g.dim == 1:
            robin_c = self.robin_data[0](g.centers(1)[RI], mu=mu)
            I = coo_matrix((robin_c, (RI, RI)), shape=(g.size(g.dim), g.size(g.dim)))
            I.eliminate_zeros()
            return csc_matrix(I).copy()
        else:
            xref = g.centers(1)[RI]
            # xref(robin-index, quadraturepoint-index)
            if self.robin_data[0].shape_range == ():
                robin_c = self.robin_data[0](xref, mu=mu)
            else:
                robin_elements = g.superentities(1, 0)[RI, 0]
                robin_indices = g.superentity_indices(1, 0)[RI, 0]
                normals = g.unit_outer_normals()[robin_elements, robin_indices]
                robin_values = self.robin_data[0](xref, mu=mu)
                robin_c = np.einsum('ei,eqi->eq', normals, robin_values)

            # robin_c(robin-index, quadraturepoint-index)
            q, w = line.quadrature(order=2)
            # remove last dimension of q, as line coordinates are one dimensional
            q = q[:, 0]
            SF = np.array([1 - q, q])
            SF_INTS = np.einsum('e,pi,pj,e,p->eij', robin_c, SF, SF, g.integration_elements(1)[RI], w).ravel()
            SF_I0 = np.repeat(g.subentities(1, g.dim)[RI], 2).ravel()
            SF_I1 = np.tile(g.subentities(1, g.dim)[RI], [1, 2]).ravel()
            I = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
            I.eliminate_zeros()
            return csc_matrix(I).copy()


class InterpolationOperator(NumpyMatrixBasedOperator):
    """Vector-like Lagrange interpolation |Operator| for continuous finite element spaces.

    Parameters
    ----------
    grid
        The |Grid| on which to interpolate.
    function
        The |Function| to interpolate.
    """

    source = NumpyVectorSpace(1)
    linear = True

    def __init__(self, grid, function):
        assert function.dim_domain == grid.dim
        assert function.shape_range == ()
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)

    def _assemble(self, mu=None):
        return self.function.evaluate(self.grid.centers(self.grid.dim), mu=mu).reshape((-1, 1))


class NonlinearReactionOperator(Operator):
    """ The operator is of the form::

    L(u,mu)(x) = q(x, mu) * c_nl(u(x, mu))

    reaction_coefficient
        The function 'q'
    reaction_function
        The function 'c_nl'
    """
    linear = False

    def __init__(self, grid, boundary_info, reaction_coefficient, reaction_function, reaction_function_derivative, space_id = 'STATE', name = None):
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid, space_id)


    def apply(self, U, mu = None, element_contribution = False, element_contribution_operator = False, rho = None):
        U = U.to_numpy().ravel()
        q, w = self.grid.reference_element.quadrature(order=2)
        SF = LagrangeShapeFunctions[self.grid.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))
        C = self.reaction_coefficient(self.grid.centers(0), mu=mu)
        # C = reaction_coefficient(q, mu = mu) #Warum nehme ich nicht die Qaudraturpunkte?
        subentities = self.grid.subentities(0, self.grid.dim)
        c_nl = np.zeros(np.shape(subentities))
        # Damit bin ich noch nicht zufrieden!
        for e in range(self.grid.size(0)):
            u_dofs = U[subentities[e]]
            wert = np.dot(u_dofs, SF)
            wert = np.reshape(wert, (3, 1))
            c_nl[e] = self.reaction_function(wert, mu = mu)
        SF_INTS = np.einsum('ji,ei,e,e,i->ej', SF, c_nl, C, self.grid.volumes(0), w).ravel()

        del C, c_nl, SF
        A = coo_matrix((SF_INTS, (subentities.ravel(), np.zeros_like(subentities.ravel()))),
                       shape=(self.grid.size(self.grid.dim), 1)).toarray().ravel()

        # if element_contribution_operator and self.boundary_info.has_dirichlet:
        #     assert rho is not None, 'rho muss übergeben werden'
        #     A_e_op = coo_matrix(([],([],[])), shape = (self.grid.size(self.grid.dim), 1))
        #     NonZeroIndices = np.where(rho != 0)[0]
        #     for e in NonZeroIndices:
        #         A_e_op = A_e_op + coo_matrix((SF_INTS[3*e:3*(e+1)], (SF_I[3*e:3*(e+1)], [0,0,0])), shape = (self.grid.size(self.grid.dim), 1))
        #     DI = self.boundary_info.dirichlet_boundaries(self.grid.dim)
        #     A_e_op[DI] = 0
        #     return self.range.make_array(A_e_op.toarray().ravel())

        #geht das auch, wenn die Dim des Problems größer als 2 ist?
        if element_contribution and self.boundary_info.has_dirichlet:
            A_e_list = []
            DI = self.boundary_info.dirichlet_boundaries(self.grid.dim)
            SF_I = subentities.ravel()
            #geht das auch anders als eine for-Schleife?
            for e in range(self.grid.size(0)):
                A_e = coo_matrix((SF_INTS[3*e:3*(e+1)], (SF_I[3*e:3*(e+1)], [0,0,0])), shape = (self.grid.size(self.grid.dim), 1)).toarray().ravel()
                A_e[DI] = 0
                A_e_list.append(self.range.make_array(A_e))
            A[DI] = 0
            del subentities, SF_INTS
            return A, A_e_list

        del subentities, SF_INTS

        if self.boundary_info.has_dirichlet:
            DI = self.boundary_info.dirichlet_boundaries(self.grid.dim)
            A[DI] = 0
        #return self.range.make_array(A.reshape((-1,1)))
        return self.range.make_array(A)

    def jacobian(self, U, mu = None, element_contribution = False, element_contribution_operator = False, rho = None):
        U = U.to_numpy().ravel()
        q, w = self.grid.reference_element.quadrature(order=2)
        SF = LagrangeShapeFunctions[self.grid.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))
        C = self.reaction_coefficient(self.grid.centers(0), mu=mu)
        # C = reaction_coefficient(q, mu = mu) #Warum nehme ich nicht die Qaudraturpunkte?
        subentities = self.grid.subentities(0, self.grid.dim)
        SF_I0 = np.repeat(self.grid.subentities(0, self.grid.dim), self.grid.dim + 1, axis=1).ravel()
        SF_I1 = np.tile(self.grid.subentities(0, self.grid.dim), [1, self.grid.dim + 1]).ravel()
        c_nl_prime = np.zeros(np.shape(subentities))
        # Damit bin ich noch nicht zufrieden!
        for e in range(self.grid.size(0)):
            u_dofs = U[subentities[e]]
            wert = np.dot(u_dofs, SF)
            wert = np.reshape(wert, (3, 1))
            c_nl_prime[e] = self.reaction_function_derivative(wert, mu = mu)
        SF_INTS = np.einsum('pi,qi,ei,e,e,i->epq', SF, SF, c_nl_prime, C, self.grid.volumes(0), w).ravel()

        del C, c_nl_prime, SF

        if self.boundary_info.has_dirichlet:
            SF_INTS = np.where(self.boundary_info.dirichlet_mask(self.grid.dim)[SF_I0], 0, SF_INTS)
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(self.grid.size(self.grid.dim), self.grid.size(self.grid.dim)))

        # if element_contribution_operator and self.boundary_info.has_dirichlet:
        #     assert rho is not None, 'rho muss übergeben werden'
        #     A_e_op = coo_matrix(([],([],[])), shape = (self.grid.size(self.grid.dim), self.grid.size(self.grid.dim)))
        #     NonZeroIndices = np.where(rho != 0)[0]
        #     for e in NonZeroIndices:
        #         A_e_op = A_e_op + coo_matrix((SF_INTS[9*e:9*(e+1)], (SF_I0[9*e:9*(e+1)], SF_I1[9*e:9*(e+1)])), shape = (self.grid.size(self.grid.dim), self.grid.size(self.grid.dim)))
        #     DI = self.boundary_info.dirichlet_boundaries(self.grid.dim)
        #     A_e_op[DI] = 0
        #     A_e_op.eliminate_zeros()
        #     A_e_op = csc_matrix(A_e_op).copy()
        #     return NumpyMatrixOperator(A_e_op, source_id = self.source.id, range_id = self.range.id)

        #geht das auch, wenn die Dimension der PDE größer als 2 ist?
        if element_contribution:
            A_e_list = []
            for e in range(self.grid.size(0)):
                A_e = coo_matrix((SF_INTS[9*e:9*(e+1)],(SF_I0[9*e:9*(e+1)], SF_I1[9*e:9*(e+1)])),  shape = (self.grid.size(self.grid.dim), self.grid.size(self.grid.dim)))
                A_e.eliminate_zeros()
                A_e = csc_matrix(A_e).copy()
                A_e_list.append(NumpyMatrixOperator(A_e, source_id = self.source.id, range_id = self.range.id))
        del SF_INTS, SF_I0, SF_I1

        A.eliminate_zeros()
        A = csc_matrix(A).copy()
        if element_contribution:
            return NumpyMatrixOperator(A, source_id = self.source.id, range_id = self.range.id), A_e_list
        else:
            return NumpyMatrixOperator(A, source_id = self.source.id, range_id = self.range.id)


class element_NonlinearReactionOperator(Operator):
    """ The operator is of the form::

    L(u,mu)(x) = q(x, mu) * c_nl(u(x,mu))

    reaction_coefficient
        The function 'q'
    reaction_function
        The function 'c_nl'
    Only for special Elements
    """
    linear = False

    def __init__(self, grid, boundary_info, reaction_coefficient, reaction_function, reaction_function_derivative, rho, space_id = 'STATE', name = None):
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid, space_id)

    def apply(self, U, mu=None):
        #macht es hier mehr Sinn wie test vorzugehen?
        NonZeroIndices = np.where(self.rho != 0)[0]
        U = U.to_numpy().ravel()
        q, w = self.grid.reference_element.quadrature(order=2)
        SF = LagrangeShapeFunctions[self.grid.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))
        #test1 = self.reaction_coefficient(self.grid.centers(0)[NonZeroIndices], mu = mu)
        C = self.reaction_coefficient(self.grid.centers(0), mu=mu)
        # C = reaction_coefficient(q, mu = mu) #Warum nehme ich nicht die Qaudraturpunkte?
        subentities = self.grid.subentities(0, self.grid.dim)
        #test2 = self.grid.subentities(0, self.grid.dim)[NonZeroIndices]
        c_nl = np.zeros(np.shape(subentities))
        #test3 = np.zeros((np.shape(test2)))
        # Damit bin ich noch nicht zufrieden!
        for e in NonZeroIndices:
            u_dofs = U[subentities[e]]
            wert = np.dot(u_dofs, SF)
            wert = np.reshape(wert, (3, 1))
            c_nl[e] = self.reaction_function(wert, mu = mu)
        SF_INTS = np.einsum('ji,ei,e,e,i->ej', SF, c_nl, C, self.grid.volumes(0), w).ravel()

        # for e in range(len(NonZeroIndices)):
        #     u_dofs = U[test2[e]]
        #     wert = np.dot(u_dofs, SF)
        #     wert = np.reshape(wert, (3, 1))
        #     test3[e] = self.reaction_function(wert)
        # test4 =  np.einsum('ji,ei,e,e,i->ej', SF, test3, test1, self.grid.volumes(0)[NonZeroIndices], w).ravel()

        del C, c_nl, SF
        A_e = coo_matrix(([], ([], [])), shape=(self.grid.size(self.grid.dim), 1))
        #test5 = coo_matrix(([], ([], [])), shape=(self.grid.size(self.grid.dim), 1))
        SF_I = subentities.ravel()
        for e in NonZeroIndices:
            A_e = A_e + self.rho[e] * coo_matrix((SF_INTS[3 * e:3 * (e + 1)], (SF_I[3 * e:3 * (e + 1)], [0, 0, 0])),
                                                 shape=(self.grid.size(self.grid.dim), 1)).toarray()
        # test6 = test2.ravel()
        # for e in range(len(NonZeroIndices)):
        #     test5 = test5 + self.rho[NonZeroIndices[e]] * coo_matrix((test4[3 * e:3 * (e + 1)], (test6[3 * e:3 * (e + 1)], [0, 0, 0])),
        #                                          shape=(self.grid.size(self.grid.dim), 1)).toarray()
        DI = self.boundary_info.dirichlet_boundaries(self.grid.dim)
        A_e[DI] = 0

        del subentities, SF_INTS

        return self.range.make_array(A_e.ravel())


    def jacobian(self, U, mu = None):
        U = U.to_numpy().ravel()
        q, w = self.grid.reference_element.quadrature(order=2)
        SF = LagrangeShapeFunctions[self.grid.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))
        C = self.reaction_coefficient(self.grid.centers(0), mu=mu)
        # C = reaction_coefficient(q, mu = mu) #Warum nehme ich nicht die Qaudraturpunkte?
        subentities = self.grid.subentities(0, self.grid.dim)
        SF_I0 = np.repeat(self.grid.subentities(0, self.grid.dim), self.grid.dim + 1, axis=1).ravel()
        SF_I1 = np.tile(self.grid.subentities(0, self.grid.dim), [1, self.grid.dim + 1]).ravel()
        c_nl_prime = np.zeros(np.shape(subentities))
        # Damit bin ich noch nicht zufrieden!
        for e in range(self.grid.size(0)):
            u_dofs = U[subentities[e]]
            wert = np.dot(u_dofs, SF)
            wert = np.reshape(wert, (3, 1))
            c_nl_prime[e] = self.reaction_function_derivative(wert, mu = mu)
        SF_INTS = np.einsum('pi,qi,ei,e,e,i->epq', SF, SF, c_nl_prime, C, self.grid.volumes(0), w).ravel()

        del C, c_nl_prime, SF

        if self.boundary_info.has_dirichlet:
            SF_INTS = np.where(self.boundary_info.dirichlet_mask(self.grid.dim)[SF_I0], 0, SF_INTS)

        #A_e_op = lil_matrix((self.grid.size(self.grid.dim), self.grid.size(self.grid.dim)))
        A_e_op = coo_matrix(([],([],[])), shape = (self.grid.size(self.grid.dim), self.grid.size(self.grid.dim)))
        NonZeroIndices = np.where(self.rho != 0)[0]
        for e in NonZeroIndices:
            A_e_op = A_e_op + self.rho[e] * coo_matrix((SF_INTS[9*e:9*(e+1)], (SF_I0[9*e:9*(e+1)], SF_I1[9*e:9*(e+1)])), shape = (self.grid.size(self.grid.dim), self.grid.size(self.grid.dim)))
        DI = self.boundary_info.dirichlet_boundaries(self.grid.dim)
        A_e_op.eliminate_zeros()
        A_e_op = csc_matrix(A_e_op).copy()

        del SF_INTS, SF_I0, SF_I1

        return NumpyMatrixOperator(A_e_op, source_id = self.source.id, range_id = self.range.id)

class quadratic_functional(Operator):
    linear = False
    """The following Operator

    j(u,mu) = 1/2 * ||u - u_d||_{L^2}^2 + 1/2 ||mu - mu_d||_2^2

    where mu_d is a given parameter and u_d = u(mu_d) is the solution of a parametrized PDE
    for mu_d

    """

    def __init__(self, grid, u_d, mu_d, name = None):
        self.__auto_init(locals())
        self.source = CGVectorSpace(grid)
        self.range = NumpyVectorSpace(1, id = 'STATE')

    def apply(self, U, mu = None, element_contribution = False):
        U = U.to_numpy().ravel()
        U_d = self.u_d.to_numpy().ravel()
        q, w = self.grid.reference_element.quadrature(order=2)
        SF = LagrangeShapeFunctions[self.grid.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))
        subentities = self.grid.subentities(0, self.grid.dim)
        from pymor.analyticalproblems.functions import ExpressionFunction
        quadatric_function = ExpressionFunction('u[0] * u[0]', dim_domain = 1, variable = 'u')
        F = np.zeros(np.shape(subentities))
        # Damit bin ich noch nicht zufrieden!
        dofs = U - U_d
        for e in range(self.grid.size(0)):
            u_dofs = dofs[subentities[e]]
            wert = np.dot(u_dofs, SF)
            wert = np.reshape(wert, (3, 1))
            F[e] = quadatric_function(wert)
        SF_INTS = np.einsum('ei,e,i->e', F, self.grid.volumes(0), w).ravel()
        del F, SF, quadatric_function
        A = 0.5 * sum(SF_INTS) + 0.5 * np.linalg.norm(mu.to_numpy() - self.mu_d.to_numpy())**2

        if element_contribution:
            A_e_list = []
            grid_volumes = self.grid.volumes(0)
            volume_grid = sum(grid_volumes)
            norm_parameter = np.linalg.norm(mu.to_numpy() - self.mu_d.to_numpy())**2
            for e in range(self.grid.size(0)):
                A_e = 0.5 * SF_INTS[e] + 0.5 * grid_volumes[e]/volume_grid * norm_parameter
                A_e_list.append(self.range.make_array(A_e))
            del subentities, SF_INTS, grid_volumes, volume_grid, norm_parameter
            return A, A_e_list

        del SF_INTS, subentities
        return A

    def jacobian(self, U, mu=None, element_contribution = False):
        U = U.to_numpy().ravel()
        U_d = self.u_d.to_numpy().ravel()
        q, w = self.grid.reference_element.quadrature(order=2)
        SF = LagrangeShapeFunctions[self.grid.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))
        subentities = self.grid.subentities(0, self.grid.dim)
        from pymor.analyticalproblems.functions import ExpressionFunction
        quadatric_function_derivative = ExpressionFunction('2 * u[0]', dim_domain = 1, variable = 'u')
        F = np.zeros(np.shape(subentities))
        # Damit bin ich noch nicht zufrieden!
        dofs = U - U_d
        for e in range(self.grid.size(0)):
            u_dofs = dofs[subentities[e]]
            wert = np.dot(u_dofs, SF)
            wert = np.reshape(wert, (3, 1))
            F[e] = quadatric_function_derivative(wert)
        SF_INTS = np.einsum('ji,ei,e,i->ej', SF, F, self.grid.volumes(0), w).ravel()
        del F, SF, quadatric_function_derivative

        A = coo_matrix((SF_INTS, (subentities.ravel(), np.zeros_like(subentities.ravel()))),
                       shape=(self.grid.size(self.grid.dim), 1)).toarray().ravel()

        if element_contribution:
            A_e_list = []
            SF_I = subentities.ravel()
            # geht das auch anders als eine for-Schleife?
            for e in range(self.grid.size(0)):
                A_e = coo_matrix((SF_INTS[3 * e:3 * (e + 1)], (SF_I[3 * e:3 * (e + 1)], [0, 0, 0])),
                                     shape=(self.grid.size(self.grid.dim), 1)).toarray().ravel()
                A_e_list.append(NumpyMatrixOperator(A_e, source_id = self.source.id, range_id = self.range.id).H)
            del SF_INTS, subentities, SF_I
            return NumpyMatrixOperator(A, source_id = self.source.id, range_id = self.range.id).H, A_e_list

        del SF_INTS, subentities
        return NumpyMatrixOperator(A, source_id = self.source.id, range_id = self.range.id).H

    def d_mu(self, mu, index = 0, element_contribution = False):
        if element_contribution:
            #Hier ignoriere ich mal kurz den Index!!!!
            element_contribution_grid = self.grid.volumes(0)
            grid_volume = sum(element_contribution_grid)
            diff = mu.to_numpy() - self.mu_d.to_numpy()
            A_e_list = []
            for e in range(self.grid.size(0)):
                A_e = element_contribution_grid[e]/grid_volume * diff
                A_e_list.append(A_e)
            return mu.to_numpy()[index] - self.mu_d.to_numpy()[index], A_e_list
        return mu.to_numpy()[index] - self.mu_d.to_numpy()[index]


class element_quadratic_functional(Operator):
    """The following Operator

    j(u,mu) = 1/2 * ||u - u_d||_{L^2}^2 + 1/2 ||mu - mu_d||_2^2

    where mu_d is a given parameter and u_d = u(mu_d) is the solution of a parametrized PDE
    for mu_d in Element_form

    """

    def __init__(self, grid, u_d, mu_d, rho = None, name = None):
        self.__auto_init(locals())
        self.source = CGVectorSpace(grid)
        self.range = NumpyVectorSpace(1, id='STATE')

    def apply(self, U, mu = None):
        NonZeroIndices = np.where(self.rho != 0)[0]
        U = U.to_numpy().ravel()
        U_d = self.u_d.to_numpy().ravel()
        q, w = self.grid.reference_element.quadrature(order=2)
        SF = LagrangeShapeFunctions[self.grid.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))
        subentities = self.grid.subentities(0, self.grid.dim)[NonZeroIndices]
        from pymor.analyticalproblems.functions import ExpressionFunction
        quadatric_function = ExpressionFunction('u[0] * u[0]', dim_domain = 1, variable = 'u')
        F = np.zeros(np.shape(subentities))
        # Damit bin ich noch nicht zufrieden!
        dofs = U - U_d
        for e in range(len(NonZeroIndices)):
            u_dofs = dofs[subentities[e]]
            wert = np.dot(u_dofs, SF)
            wert = np.reshape(wert, (3, 1))
            F[e] = quadatric_function(wert)
        SF_INTS = np.einsum('ei,e,i->e', F, self.grid.volumes(0)[NonZeroIndices], w).ravel()
        del F, SF, quadatric_function
        #da ||mu - mu_d|| keine Darstellung über die Summe aller Elemente besitzt, erzwinge ich diese durch
        # || mu - mu_d || = \sum |\Omega_e| / |Omega| * || mu - mu_d ||

        grid_volumes = self.grid.volumes(0)
        element_contribution_grid = grid_volumes[NonZeroIndices]
        volume_grid = sum(grid_volumes)
        norm_parameter = np.linalg.norm(mu.to_numpy() - self.mu_d.to_numpy())**2
        A_e = (self.rho[NonZeroIndices]).dot(0.5 * SF_INTS + 0.5 * element_contribution_grid/volume_grid * norm_parameter)
        del SF_INTS, subentities, grid_volumes, element_contribution_grid, norm_parameter
        return A_e

    def jacobian(self, U, mu=None):
        NonZeroIndices = np.where(self.rho != 0)[0]
        U = U.to_numpy().ravel()
        U_d = self.u_d.to_numpy().ravel()
        q, w = self.grid.reference_element.quadrature(order=2)
        SF = LagrangeShapeFunctions[self.grid.reference_element][1]
        SF = np.array(tuple(f(q) for f in SF))
        subentities = self.grid.subentities(0, self.grid.dim)[NonZeroIndices]
        from pymor.analyticalproblems.functions import ExpressionFunction
        quadatric_function_derivative = ExpressionFunction('2 * u[0]', dim_domain=1, variable='u')
        F = np.zeros(np.shape(subentities))
        # Damit bin ich noch nicht zufrieden!
        dofs = U - U_d
        for e in range(len(NonZeroIndices)):
            u_dofs = dofs[subentities[e]]
            wert = np.dot(u_dofs, SF)
            wert = np.reshape(wert, (3, 1))
            F[e] = quadatric_function_derivative(wert)
        SF_INTS = np.einsum('ji,ei,e,i->ej', SF, F, self.grid.volumes(0)[NonZeroIndices], w).ravel()
        del F, SF, quadatric_function_derivative
        A_e = coo_matrix(([], ([],[])), shape = (self.grid.size(2), 1))

        for e in range(len(NonZeroIndices)):
            A_e = A_e + self.rho[NonZeroIndices[e]] * coo_matrix((SF_INTS[3*e:3*(e+1)], (subentities.ravel()[3*e:3*(e+1)], [0,0,0])), shape = (self.grid.size(self.grid.dim), 1))
        del SF_INTS, subentities
        return NumpyMatrixOperator(A_e.toarray().ravel(), source_id = self.source.id, range_id = self.range.id).H

    def d_mu(self, mu, index = 0):
        NonZeroIndices = np.where(self.rho != 0)[0]
        element_contribution_grid = self.grid.volumes(0)[NonZeroIndices]
        return sum(element_contribution_grid)/sum(self.grid.volumes(0)) * (mu.to_numpy()[index] - self.mu_d.to_numpy()[index])

def discretize_stationary_cg(analytical_problem, diameter=None, domain_discretizer=None,
                             grid_type=None, grid=None, boundary_info=None,
                             preassemble=True, mu_energy_product=None):
    """Discretizes a |StationaryProblem| using finite elements.

    Parameters
    ----------
    analytical_problem
        The |StationaryProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed as an argument to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If `None`, |discretize_domain_default| is used.
    grid_type
        If not `None`, this parameter is forwarded to `domain_discretizer` to specify
        the type of the generated |Grid|.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.
    preassemble
        If `True`, preassemble all operators in the resulting |Model|.
    mu_energy_product
        If not `None`, |parameter values| for which to assemble the symmetric part of the
        |Operator| of the resulting |Model| `fom` (ignoring the advection part). Thus,
        assuming no advection and a symmetric diffusion tensor, `fom.products['energy']`
        is equal to `fom.operator.assemble(mu)`, except for the fact that the former has
        cleared Dirichlet rows and columns, while the latter only
        has cleared Dirichlet rows).

    Returns
    -------
    m
        The |Model| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
            :unassembled_m:  In case `preassemble` is `True`, the generated |Model|
                             before preassembling operators.
    """
    #das wurde geändert
    #assert isinstance(analytical_problem, StationaryProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert grid_type is None or grid is None

    p = analytical_problem

    #das wurde geändert
    # if not (p.nonlinear_advection
    #         == p.nonlinear_advection_derivative
    #         == p.nonlinear_reaction
    #         == p.nonlinear_reaction_derivative
    #         is None):
    #     raise NotImplementedError

    if not (p.nonlinear_advection
            == p.nonlinear_advection_derivative
            is None):
        raise NotImplementedError

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if grid_type:
            domain_discretizer = partial(domain_discretizer, grid_type=grid_type)
        if diameter is None:
            grid, boundary_info = domain_discretizer(p.domain)
        else:
            grid, boundary_info = domain_discretizer(p.domain, diameter=diameter)

    assert grid.reference_element in (line, triangle, square)

    if grid.reference_element is square:
        DiffusionOperator = DiffusionOperatorQ1
        AdvectionOperator = AdvectionOperatorQ1
        ReactionOperator  = L2ProductQ1
        L2Functional = L2ProductFunctionalQ1
        BoundaryL2Functional = BoundaryL2ProductFunctional
    else:
        DiffusionOperator = DiffusionOperatorP1
        AdvectionOperator = AdvectionOperatorP1
        ReactionOperator  = L2ProductP1
        L2Functional = L2ProductFunctionalP1
        BoundaryL2Functional = BoundaryL2ProductFunctional

    Li = [DiffusionOperator(grid, boundary_info, diffusion_constant=0, name='boundary_part')]
    if mu_energy_product:
        eLi = [DiffusionOperator(grid, boundary_info, dirichlet_clear_columns=True, diffusion_constant=0)]
    coefficients = [1.]

    # diffusion part
    if isinstance(p.diffusion, LincombFunction):
        Li += [DiffusionOperator(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                                 name=f'diffusion_{i}')
               for i, df in enumerate(p.diffusion.functions)]
        if mu_energy_product:
            eLi += [DiffusionOperator(grid, boundary_info, diffusion_function=p.diffusion, dirichlet_clear_diag=True,
                                      dirichlet_clear_columns=True)]
        coefficients += list(p.diffusion.coefficients)
    elif p.diffusion is not None:
        Li += [DiffusionOperator(grid, boundary_info, diffusion_function=p.diffusion,
                                 dirichlet_clear_diag=True, name='diffusion')]
        if mu_energy_product:
            eLi += [DiffusionOperator(grid, boundary_info, diffusion_function=p.diffusion,
                                      dirichlet_clear_diag=True, dirichlet_clear_columns=True)]
        coefficients.append(1.)

    # advection part
    if isinstance(p.advection, LincombFunction):
        Li += [AdvectionOperator(grid, boundary_info, advection_function=af, dirichlet_clear_diag=True,
                                 name=f'advection_{i}')
               for i, af in enumerate(p.advection.functions)]
        coefficients += list(p.advection.coefficients)
    elif p.advection is not None:
        Li += [AdvectionOperator(grid, boundary_info, advection_function=p.advection,
                                 dirichlet_clear_diag=True, name='advection')]
        coefficients.append(1.)

    # reaction part
    if isinstance(p.reaction, LincombFunction):
        Li += [ReactionOperator(grid, boundary_info, coefficient_function=rf, dirichlet_clear_diag=True,
                                name=f'reaction_{i}')
               for i, rf in enumerate(p.reaction.functions)]
        if mu_energy_product:
            eLi += [ReactionOperator(grid, boundary_info, coefficient_function=p.reaction, dirichlet_clear_diag=True,
                                     dirichlet_clear_columns=True)]
        coefficients += list(p.reaction.coefficients)
    elif p.reaction is not None:
        Li += [ReactionOperator(grid, boundary_info, coefficient_function=p.reaction,
                                dirichlet_clear_diag=True, name='reaction')]
        if mu_energy_product:
            eLi += [ReactionOperator(grid, boundary_info, coefficient_function=p.reaction, dirichlet_clear_columns=True,
                                     dirichlet_clear_diag=True)]
        coefficients.append(1.)
    # nonlinear reaction part
    if p.nonlinear_reaction is not None:
        Li += [NonlinearReactionOperator(grid, boundary_info, reaction_coefficient = p.nonlinear_reaction_coefficient,
                                        reaction_function = p.nonlinear_reaction, reaction_function_derivative = p.nonlinear_reaction_derivative)]
        coefficients += [1.]

    # robin boundaries
    if p.robin_data is not None:
        assert isinstance(p.robin_data, tuple) and len(p.robin_data) == 2
        if isinstance(p.robin_data[0], LincombFunction):
            for i, rd in enumerate(p.robin_data[0].functions):
                robin_tuple = (rd, p.robin_data[1])
                Li += [RobinBoundaryOperator(grid, boundary_info, robin_data=robin_tuple, name=f'robin_{i}')]
            coefficients += list(p.robin_data[0].coefficients)
            if mu_energy_product:
                eLi += [RobinBoundaryOperator(grid, boundary_info, robin_data=p.robin_data)]
        else:
            Li += [RobinBoundaryOperator(grid, boundary_info, robin_data=p.robin_data, name='robin')]
            if mu_energy_product:
                eLi += [RobinBoundaryOperator(grid, boundary_info, robin_data=p.robin_data)]
            coefficients.append(1.)

    L = LincombOperator(operators=Li, coefficients=coefficients, name='ellipticOperator')
    if mu_energy_product:
        eL = LincombOperator(operators=eLi, coefficients=[1.]*len(eLi), name='ellipticEnergyProduct')

    # right-hand side
    rhs = p.rhs or ConstantFunction(0., dim_domain=p.domain.dim)
    Fi = []
    coefficients_F = []
    if isinstance(p.rhs, LincombFunction):
        Fi += [L2Functional(grid, rh, dirichlet_clear_dofs=True, boundary_info=boundary_info, name=f'rhs_{i}')
               for i, rh in enumerate(p.rhs.functions)]
        coefficients_F += list(p.rhs.coefficients)
    else:
        Fi += [L2Functional(grid, rhs, dirichlet_clear_dofs=True, boundary_info=boundary_info, name='rhs')]
        coefficients_F.append(1.)

    if p.neumann_data is not None and boundary_info.has_neumann:
        if isinstance(p.neumann_data, LincombFunction):
            Fi += [BoundaryL2Functional(grid, -ne, boundary_info=boundary_info,
                                        boundary_type='neumann', dirichlet_clear_dofs=True, name=f'neumann_{i}')
                   for i, ne in enumerate(p.neumann_data.functions)]
            coefficients_F += list(p.neumann_data.coefficients)
        else:
            Fi += [BoundaryL2Functional(grid, -p.neumann_data, boundary_info=boundary_info,
                                        boundary_type='neumann', dirichlet_clear_dofs=True)]
            coefficients_F.append(1.)

    if p.robin_data is not None and boundary_info.has_robin:
        if isinstance(p.robin_data[0], LincombFunction):
            Fi += [BoundaryL2Functional(grid, rob * p.robin_data[1], boundary_info=boundary_info,
                                        boundary_type='robin', dirichlet_clear_dofs=True, name=f'robin_{i}')
                   for i, rob in enumerate(p.robin_data[0].functions)]
            coefficients_F += list(p.robin_data[0].coefficients)
        else:
            Fi += [BoundaryL2Functional(grid, p.robin_data[0] * p.robin_data[1], boundary_info=boundary_info,
                                        boundary_type='robin', dirichlet_clear_dofs=True)]
            coefficients_F.append(1.)

    if p.dirichlet_data is not None and boundary_info.has_dirichlet:
        if isinstance(p.dirichlet_data, LincombFunction):
            Fi += [BoundaryDirichletFunctional(grid, di, boundary_info, name=f'dirichlet{i}')
                   for i, di in enumerate(p.dirichlet_data.functions)]
            coefficients_F += list(p.dirichlet_data.coefficients)
        else:
            Fi += [BoundaryDirichletFunctional(grid, p.dirichlet_data, boundary_info)]
            coefficients_F.append(1.)

    F = LincombOperator(operators=Fi, coefficients=coefficients_F, name='rhsOperator')

    if grid.reference_element in (triangle, square):
        visualizer = PatchVisualizer(grid=grid, codim=2)
    elif grid.reference_element is line:
        visualizer = OnedVisualizer(grid=grid, codim=1)
    else:
        visualizer = None

    Prod = L2ProductQ1 if grid.reference_element is square else L2ProductP1
    empty_bi = EmptyBoundaryInfo(grid)
    l2_product = Prod(grid, empty_bi, name='l2')
    l2_0_product = Prod(grid, boundary_info, dirichlet_clear_columns=True, name='l2_0')
    h1_semi_product = DiffusionOperator(grid, empty_bi, name='h1_semi')
    h1_0_semi_product = DiffusionOperator(grid, boundary_info, dirichlet_clear_columns=True, name='h1_0_semi')
    products = {'h1': l2_product + h1_semi_product,
                'h1_semi': h1_semi_product,
                'l2': l2_product,
                'h1_0': l2_0_product + h1_0_semi_product,
                'h1_0_semi': h1_0_semi_product,
                'l2_0': l2_0_product}

    # assemble additional output functionals
    if p.outputs:
        if any(v[0] not in ('l2', 'l2_boundary') for v in p.outputs):
            raise NotImplementedError
        outputs = []
        for v in p.outputs:
            if v[0] == 'l2':
                if isinstance(v[1], LincombFunction):
                    ops = [L2Functional(grid, vv, dirichlet_clear_dofs=False).H
                           for vv in v[1].functions]
                    outputs.append(LincombOperator(ops, v[1].coefficients))
                else:
                    outputs.append(L2Functional(grid, v[1], dirichlet_clear_dofs=False).H)
            else:
                if isinstance(v[1], LincombFunction):
                    ops = [BoundaryL2Functional(grid, vv, dirichlet_clear_dofs=False).H
                           for vv in v[1].functions]
                    outputs.append(LincombOperator(ops, v[1].coefficients))
                else:
                    outputs.append(BoundaryL2Functional(grid, v[1], dirichlet_clear_dofs=False).H)
        if len(outputs) > 1:
            from pymor.operators.block import BlockColumnOperator
            from pymor.operators.constructions import NumpyConversionOperator
            output_functional = BlockColumnOperator(outputs)
            output_functional = NumpyConversionOperator(output_functional.range) @ output_functional
        else:
            output_functional = outputs[0]
    else:
        output_functional = None

    # assemble additional product
    if mu_energy_product:
        if preassemble:
            # mu_energy_product is the |Parameter| with which we build the energy product (s.a.)
            eL = eL.assemble(mu_energy_product)
        else:
            from pymor.operators.constructions import FixedParameterOperator
            eL = FixedParameterOperator(eL, mu=mu_energy_product)
        if p.diffusion is not None:
            scalar_diffusion = len(p.diffusion.shape_range) == 0
            if scalar_diffusion:
                products['energy'] = eL
            else:
                eL_unassembled = 0.5*eL + 0.5*eL.H
                products['energy'] = eL_unassembled.assemble() if preassemble else eL_unassembled
        else:
            products['energy'] = eL

    m  = StationaryModel(L, F, output_functional=output_functional, products=products, visualizer=visualizer,
                         name=f'{p.name}_CG')

    data = {'grid': grid, 'boundary_info': boundary_info}

    if preassemble:
        data['unassembled_m'] = m
        m = preassemble_(m)

    return m, data