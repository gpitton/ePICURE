from interfaces.vector_space import *
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import lil_matrix, dok_matrix, csc_matrix, csr_matrix


def interpolation_matrix(vs, points = None, d = 0):
    """Compute the Interpolation Matrix associated with the given vector space
    and the given points. This matrix can be used to solve Interpolation and
    Least Squares approximations. The matrix which is returned is defined as

    f(x) = sum(coeffs[i]*vs.basis_der(i,d)(x)), then M*coeffs = f(points)

    If points is not specified, use by default Gaussian quadrature points
    with degree = 2*vs.n_dofs - 1
    """
    # M_ij := v_j(q_i)
    if points is None:
        points, _ = np.polynomial.legendre.leggauss(2*vs.n_dofs - 1)
    col = points.reshape((-1,1))
    M = np.zeros((len(points), vs.n_dofs), order='F')
    for i in range(vs.n_dofs):
        # advanced slicing... only compute those points which are
        # different from zero
        ia, ib = vs.basis_span(i)
        a = vs.cells[ia]
        b = vs.cells[ib]
        ids = (a <= points) & (points <= b)
        if d == 0:
            M[ids, i] = vs.basis(i)(points[ids])
        else:
            M[ids, i] = vs.basis_der(i, d)(points[ids])
    return np.asmatrix(M)


def mass_matrix(vs, nq, quadfun=np.polynomial.legendre.leggauss, matrix_format="FULL", internal="DOK"):
    """A MassMatrix assembler, computing the sparse matrix 
    
    $M_{ij} =\int_0^1 v_i(x) v_j(x) dx$
    vs: A vector_space type object.
    nq: Number of quadrature points used for each individual integration.
    quadfun: a function that receives a number and returns a tuple of
                numpy arrays with the values of quadrature points and 
                the weights in the interval [-1;1]
    matrix_format: output matrix format. Can be CSR, CSC or FULL (default).
    This function returns a scipy.sparse.lil.lil_matrix that can be converted to
    python array, ndarrays or equivalents usually with only 1 call.
    Arithmetic operations are also supported.
    see: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.lil_matrix.html
    
    """
    (qst, wst) = quadfun(nq)
    cum = {"DOK": lambda: dok_matrix((vs.n_dofs, vs.n_dofs)), "LIL": lambda: lil_matrix((vs.n_dofs, vs.n_dofs))}[internal]()
    for lcell in range(vs.n_cells):
        a, b = vs.cells[lcell], vs.cells[lcell + 1]
        q, w = ((a+b) + (b-a)*qst)/2, (b-a)*wst/2
        fnts = vs.cell_span(lcell)
        nfnts = len(fnts)
        fq = np.zeros((nfnts,nq))
        for i in range(nfnts):                                         #funtion evaluation is a heavy operation do it one time
            fq[i] = vs.basis(fnts[i])(q)
        #out = np.dot(fq*w, fq.transpose())                             #all in only one step
        out = np.einsum('iq, jq, q -> ij', fq, fq, w)
        for rout, rcum in zip(out, fnts):
            for vout, ccum in zip(rout, fnts):
                cum[rcum, ccum] += vout
                
    fun = {"FULL": lambda: cum.toarray(), "CSC": lambda: cum.tocsc(), "CSR": lambda: cum.tocsr()}

    return fun[matrix_format]()


def stiffness_matrix(vs, nq, quadfun=np.polynomial.legendre.leggauss, matrix_format="FULL", internal="DOK"):
    """A StiffnessMatrix assembler, computing the sparse matrix

    $M_{ij} =\int_0^1 v'_i(x) v'_j(x) dx$
    vs: A vector_space type object.
    nq: Number of quadrature points used for each individual integration.
    quadfun: a function that receives a number and returns a tuple of
                numpy arrays with the values of quadrature points and
                the weights in the interval [-1;1]
    matrix_format: output matrix format. Can be CSR, CSC or FULL (default).
    This function returns a scipy.sparse.lil.lil_matrix that can be converted to
    python array, ndarrays or equivalents usually with only 1 call.
    Arithmetic operations are also supported.
    see: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.lil_matrix.html

    """
    (qst, wst) = quadfun(nq)
    cum = {"DOK": lambda: dok_matrix((vs.n_dofs, vs.n_dofs)), "LIL": lambda: lil_matrix((vs.n_dofs, vs.n_dofs))}[
        internal]()
    for lcell in range(vs.n_cells):
        a, b = vs.cells[lcell], vs.cells[lcell + 1]
        q, w = ((a + b) + (b - a) * qst) / 2, (b - a) * wst / 2
        fnts = vs.cell_span(lcell)
        nfnts = len(fnts)
        fq = np.zeros((nfnts, nq))
        for i in range(nfnts):  # funtion evaluation is a heavy operation do it one time
            fq[i] = vs.basis_der(fnts[i], 1)(q)
        # out = np.dot(fq*w, fq.transpose())                             #all in only one step
        out = np.einsum('iq, jq, q -> ij', fq, fq, w)
        for rout, rcum in zip(out, fnts):
            for vout, ccum in zip(rout, fnts):
                cum[rcum, ccum] += vout

    fun = {"FULL": lambda: cum.toarray(), "CSC": lambda: cum.tocsc(), "CSR": lambda: cum.tocsr()}

    return fun[matrix_format]()