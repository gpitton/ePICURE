import numpy as np
from interfaces import *
from utilities import *
from igakit.cad import line, revolve, refine




def compare_stiffness_2d(srf):
    knots = srf.knots

    # list of vector spaces
    vs = []
    # quadrature points
    q = []
    # quadrature weights
    w = []
    for d in range(2):
        vs.append(BsplineVectorSpace(knots = knots[d]))
        tmp_q, tmp_w = np.polynomial.legendre.leggauss(2*vs[d].n_dofs - 1)
        q.append(tmp_q)
        w.append(tmp_w)


    # stiffness matrix. Computation by cycling over the cells and manual assembly
    stiffness_partial = []
    for d in range(2):
        stiffness_partial.append(np.zeros((vs[d].n_dofs,vs[d].n_dofs)))

    for acell in range(vs[0].n_cells):
        a, b = vs[0].cells[acell], vs[0].cells[acell + 1]
        qa, wa = ((a + b) + (b - a) * q[0]) / 2, (b - a) * w[0] / 2
        aidx = vs[0].cell_span(acell)
        a_idx = len(aidx)
        fq = np.zeros((a_idx, len(qa)))
        for i in range(a_idx):
            fq[i] = vs[0].basis_der(aidx[i], 1)(qa)
        out_a = np.einsum('iq, jq, q -> ij', fq, fq, wa)
        for rout_a, rc_a in zip(out_a, aidx):
            for v_a, cc_a in zip(rout_a, aidx):
                stiffness_partial[0][rc_a, cc_a] += v_a

    for bcell in range(vs[1].n_cells):
        a, b = vs[1].cells[bcell], vs[1].cells[bcell + 1]
        qb, wb = ((a + b) + (b - a) * q[1]) / 2, (b - a) * w[1] / 2
        bidx = vs[1].cell_span(bcell)
        b_idx = len(bidx)
        fq = np.zeros((b_idx, len(qb)))
        for i in range(b_idx):
            fq[i] = vs[1].basis_der(bidx[i], 1)(qb)
        out_b = np.einsum('iq, jq, q -> ij', fq, fq, wb)
        for rout_b, rc_b in zip(out_b, bidx):
            for v_b, cc_b in zip(rout_b, bidx):
                stiffness_partial[1][rc_b, cc_b] += v_b

    stiffness_manual = np.einsum('ik, jl -> ijkl', stiffness_partial[0], stiffness_partial[1])
    n = (stiffness_manual.shape[0], stiffness_manual.shape[1])
    stiffness_manual = stiffness_manual.reshape((np.prod(n), np.prod(n)))



    # stiffness matrix. Computation by using built-in functions and exploiting
    # tensor product structure of basis functions
    stiffness_1d = []
    for d in range(2):
        stiffness_1d.append(stiffness_matrix(vs[d], 2*vs[d].n_dofs))

    stiffness_tensor = np.einsum('ik, jl -> ijkl', stiffness_1d[0], stiffness_1d[1])
    n = (stiffness_tensor.shape[0], stiffness_tensor.shape[1])
    stiffness_tensor = stiffness_tensor.reshape((np.prod(n), np.prod(n)))

    np.testing.assert_array_almost_equal(stiffness_manual, stiffness_tensor)


def test_h_refinement():
    # set up geometry: 2d annulus
    R1 = 0.5
    R2 = 1.

    c1 = line([R1, 0], [R2, 0])
    srf = revolve(c1, point = [0, 0], axis = 2, angle = [0, np.pi/2.])
    for el_x in range(1, 6):
        for el_y in range(1, 6):
            refined_surface = refine(srf.copy(), factor = (el_x, el_y))
            compare_stiffness_2d(refined_surface)


def test_p_refinement():
    # set up geometry: 2d annulus
    R1 = 0.5
    R2 = 1.

    c1 = line([R1, 0], [R2, 0])
    srf = revolve(c1, point = [0, 0], axis = 2, angle = [0, np.pi/2.])
    for px in range(8):
        for py in range(8):
            refined_surface = refine(srf.copy(), degree = (px, py))
            compare_stiffness_2d(refined_surface)
