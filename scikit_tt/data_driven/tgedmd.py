import numpy as np
import multiprocessing as mp
from scikit_tt import TT
from scikit_tt.data_driven.transform import basis_decomposition, Function, hocur


def amuset_hosvd(data_matrix, basis_list, b, sigma, num_eigvals=np.infty, threshold=1e-2, max_rank=np.infty,
                 return_option='eigentensors', chunk_size=None):
    """
    AMUSE algorithm for calculation of eigenvalues of the Koopman generator.

    The tensor-trains are created using the exact TT decomposition, whose ranks are reduced using SVDs.

    Parameters
    ----------
    data_matrix : np.ndarray
        snapshot matrix, shape (d, m)
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)
    num_eigvals : int, optional
        number of eigenvalues and eigentensors that are returned
        default: return all calculated eigenvalues and eigentensors
    threshold : float, optional
        threshold for svd of psi and dpsi
    max_rank : int, optional
        maximal rank of TT representations of psi and dpsi after svd/ortho
    return_option : {'eigentensors', 'eigenfunctionevals'}
        'eigentensors': return a list of the eigentensors of the koopman generator
        'eigenfunctionevals': return the evaluations of the eigenfunctions of the koopman generator at all snapshots
    chunk_size : int or None, optional
        if a chunk_size is specified, M in AMUSEt is built in chunks

    Returns
    -------
    eigvals : np.ndarray
        eigenvalues of Koopman generator
    eigtensors : list[TT] or np.ndarray
        eigentensors of Koopman generator or evaluations of eigenfunctions at snapshots (shape (*, m))
        (cf. return_option)
    """

    print('calculating psi...')
    psi = basis_decomposition(data_matrix, basis_list)
    p = psi.order - 1
    # SVD of psi
    u, s, v = psi.svd(p, threshold=threshold, max_rank=max_rank, ortho_l=True, ortho_r=False)
    psi = u.rank_tensordot(np.diag(s))
    psi.concatenate(v, overwrite=True)  # rank reduced version
    print(psi)

    print('calculating M in AMUSEt')
    if chunk_size is None:
        dpsi = tt_decomposition(data_matrix, basis_list, b, sigma)
        dpsi = dpsi.ortho_left(threshold=threshold, max_rank=max_rank)
        s_inv = np.diag(1.0 / s)
        u.rank_tensordot(s_inv, mode='last', overwrite=True)
        M = _amuset(u, v, dpsi)
    else:
        M = _amuset_chunks(u, s, v, data_matrix, basis_list, b, sigma, threshold, max_rank, chunk_size)

    print('calculating eigenvalues and eigentensors...')
    # calculate eigenvalues of M
    eigvals, eigvecs = np.linalg.eig(M)

    sorted_indices = np.argsort(-eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    if not (eigvals < 0).all():
        print('WARNING: there are eigenvalues >= 0')

    if len(eigvals > num_eigvals):
        eigvals = eigvals[:num_eigvals]
        eigvecs = eigvecs[:, :num_eigvals]

    # calculate eigentensors
    if return_option == 'eigentensors':
        eigvecs = eigvecs[:, :, np.newaxis]
        eigtensors = []
        for i in range(eigvals.shape[0]):
            eigtensor = u.copy()
            eigtensor.rank_tensordot(eigvecs[:, i, :], overwrite=True)
            eigtensors.append(eigtensor)

        return eigvals, eigtensors
    else:
        u.rank_tensordot(eigvecs, overwrite=True)
        u.tensordot(psi, p, mode='first-first', overwrite=True)
        u = u.cores[0][0, :, 0, :].T
        return eigvals, u


def tt_decomposition_chunks(x, basis_list, b, sigma, threshold=1e-2, max_rank=np.infty, chunk_size=100):
    """
    Calculate dPsi(X) in chunks.

    The data is divided in chunks. dPsi(X) is calculated for the first and second chunk and the two tensors
    added and rank-compressed. Then dPsi(X) is calculated for the third chunk, added and the resulting tensor is
    again rank compressed. And so on...

    Parameters
    ----------
    x : np.ndarray
        snapshot matrix of size d x m
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)
    threshold : float, optional
        threshold for compression
    max_rank : int, optional
        maximal rank after compression
    chunk_size : int, optional
        how much data is used in one chunk

    Returns
    -------
    TT
        tensor train of basis function evaluations
    """

    m = x.shape[1]
    start_chunk = 0
    end_chunk = min(m, start_chunk + chunk_size)

    dPsi = _tt_decomposition_one_chunk(x[:, start_chunk:end_chunk], basis_list, b, sigma, start_chunk, m)
    dPsi.ortho_left(threshold=threshold, max_rank=max_rank)

    while end_chunk < m:
        start_chunk = end_chunk
        end_chunk = min(m, start_chunk + chunk_size)
        new = _tt_decomposition_one_chunk(x[:, start_chunk:end_chunk], basis_list, b, sigma, start_chunk, m)
        dPsi += new
        dPsi.ortho_left(threshold=threshold, max_rank=max_rank)

    return dPsi


def _amuset_chunks(u, s, v, x, basis_list, b, sigma, threshold=1e-2, max_rank=np.infty, chunk_size=100):
    """
    Construct the Matrix M in AMUSEt in chunks.

    If chunk_size = 1, M can be constructed using the special tensordot. In this case no compression takes places,
    and threshold, max_rank are therefore ignored.

    Parameters
    ----------
    u : TT
    s : np.ndarray
    v : TT
    x : np.ndarray
        snapshot matrix of size d x m
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)
    threshold : float, optional
        threshold for compression
    max_rank : int, optional
        maximal rank after compression
    chunk_size : int, optional
        how much data is used in one chunk

    Returns
    -------
    np.ndarray
        matrix M from AMUSEt
    """
    m = x.shape[1]
    start_chunk = 0
    end_chunk = min(m, start_chunk + chunk_size)
    print('amuset: chunk {} - {}'.format(start_chunk, end_chunk))

    # if the chunk_size is one we can use the _amuset_special
    if chunk_size == 1:
        amuse_fun = _amuset_special
    else:
        amuse_fun = _amuset

    s_inv = np.diag(1.0 / s)
    u.rank_tensordot(s_inv, mode='last', overwrite=True)

    dPsi = _tt_decomposition_one_chunk(x[:, start_chunk:end_chunk], basis_list, b, sigma, start_chunk, m)
    if chunk_size > 1:  # for chunk_size = 1 the structure mustnt change
        dPsi.ortho_left(threshold=threshold, max_rank=max_rank)
    M = amuse_fun(u, v, dPsi)

    while end_chunk < m:
        start_chunk = end_chunk
        end_chunk = min(m, start_chunk + chunk_size)
        print('amuset: chunk {} - {}'.format(start_chunk, end_chunk))
        dPsi = _tt_decomposition_one_chunk(x[:, start_chunk:end_chunk], basis_list, b, sigma, start_chunk, m)
        if chunk_size > 1:  # for chunk_size = 1 the structure mustnt change
            dPsi.ortho_left(threshold=threshold, max_rank=max_rank)
        M += amuse_fun(u, v, dPsi)

    return M


def calc_M(this_chunk, x, basis_list, b, sigma, m, chunk_size, threshold, max_rank, amuse_fun, u, v):
    start_chunk, end_chunk = this_chunk
    dPsi = _tt_decomposition_one_chunk(x[:, start_chunk:end_chunk], basis_list, b, sigma, start_chunk, m)
    if chunk_size > 1:  # for chunk_size = 1 the structure mustnt change
        dPsi.ortho_left(threshold=threshold, max_rank=max_rank)
    return amuse_fun(u, v, dPsi)


def _amuset_chunks_parallel(u, s, v, x, basis_list, b, sigma, threshold=1e-2, max_rank=np.infty, chunk_size=100):
    """
    Construct the Matrix M in AMUSEt in chunks using parallel processing.

    If chunk_size = 1, M can be constructed using the special tensordot. In this case no compression takes places,
    and threshold, max_rank are therefore ignored.

    Parameters
    ----------
    u : TT
    s : np.ndarray
    v : TT
    x : np.ndarray
        snapshot matrix of size d x m
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)
    threshold : float, optional
        threshold for compression
    max_rank : int, optional
        maximal rank after compression
    chunk_size : int, optional
        how much data is used in one chunk

    Returns
    -------
    np.ndarray
        matrix M from AMUSEt
    """

    s_inv = np.diag(1.0 / s)
    u.rank_tensordot(s_inv, mode='last', overwrite=True)

    m = x.shape[1]
    chunks = [(0, chunk_size)]
    while True:
        if chunks[-1][1] >= m:
            chunks[-1] = (chunks[-1][0], m)
            break
        chunks.append((chunks[-1][1], chunks[-1][1] + chunk_size))

    # if the chunk_size is one we can use the _amuset_special
    if chunk_size == 1:
        amuse_fun = _amuset_special
    else:
        amuse_fun = _amuset

    pool = mp.Pool(mp.cpu_count())
    results = []
    for chunk in chunks:
        results.append(pool.apply_async(calc_M, args=(chunk, x, basis_list, b, sigma, m, chunk_size, threshold, max_rank, amuse_fun, u, v)))
    results = [r.get() for r in results]
    pool.close()
    pool.join()

    return np.sum(results, axis=0)


def _tt_decomposition_one_chunk(x, basis_list, b, sigma, start_chunk, m_total):
    """
    Calculate the exact tt_decomposition of a chunk of dPsi(X).

    Parameters
    ----------
    x : np.ndarray
        snapshot matrix of size d x m
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)
    start_chunk : int
        index of the first snapshot in the chunk
    m_total : int
        total number of snapshots (necessary to build the unit-vectors in the last core)

    Returns
    -------
    TT
        tt_decomposition of a chunk of dPsi(X)
    """
    # number of snapshots
    m = x.shape[1]
    # dimension
    d = x.shape[0]
    # number of modes
    p = len(basis_list)
    # mode dimensions
    n = [len(basis_list[i]) for i in range(p)]

    # define cores 1,...,p as a list of empty arrays
    cores = [np.zeros([1, n[0], 1, m * (d + 2)])] + \
            [np.zeros([m * (d + 2), n[i], 1, m * (d + 2)]) for i in range(1, p - 1)] + \
            [np.zeros([m * (d + 2), n[p - 1], 1, m])]

    # insert elements of core 1
    cores[0] = np.concatenate([dPsix(basis_list[0], x[:, k], b, sigma, position='first') for k in range(m)],
                              axis=3)

    # insert elements of cores 2,...,p-1
    for i in range(1, p - 1):
        for k in range(m):
            cores[i][k * (d + 2): (k + 1) * (d + 2), :, :, k * (d + 2): (k + 1) * (d + 2)] = dPsix(
                basis_list[i], x[:, k], b, sigma, position='middle')

    # insert elements of core p
    for k in range(m):
        cores[p - 1][k * (d + 2): (k + 1) * (d + 2), :, :, k:k + 1] = dPsix(basis_list[p - 1], x[:, k], b, sigma,
                                                                            position='last')

    # append core containing unit vectors
    last_core = np.zeros((m, m_total, 1, 1))
    for k in range(m):
        last_core[k, start_chunk + k, 0, 0] = 1

    cores.append(last_core)

    return TT(cores)


def tt_decomposition(x, basis_list, b, sigma):
    """
    Calculates exact tt-decomposition of dPsi(X).

    Parameters
    ----------
    x : np.ndarray
        snapshot matrix of size d x m
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)

    Returns
    -------
    TT
        tensor train of basis function evaluations
    """

    # number of snapshots
    m = x.shape[1]
    # dimension
    d = x.shape[0]
    # number of modes
    p = len(basis_list)
    # mode dimensions
    n = [len(basis_list[i]) for i in range(p)]

    # define cores 1,...,p as a list of empty arrays
    cores = [np.zeros([1, n[0], 1, m * (d + 2)])] + \
            [np.zeros([m * (d + 2), n[i], 1, m * (d + 2)]) for i in range(1, p - 1)] + \
            [np.zeros([m * (d + 2), n[p - 1], 1, m])]

    # insert elements of core 1
    cores[0] = np.concatenate([dPsix(basis_list[0], x[:, k], b, sigma, position='first') for k in range(m)],
                              axis=3)

    # insert elements of cores 2,...,p-1
    for i in range(1, p - 1):
        for k in range(m):
            cores[i][k * (d + 2): (k + 1) * (d + 2), :, :, k * (d + 2): (k + 1) * (d + 2)] = dPsix(
                basis_list[i], x[:, k], b, sigma, position='middle')

    # insert elements of core p
    for k in range(m):
        cores[p - 1][k * (d + 2): (k + 1) * (d + 2), :, :, k:k + 1] = dPsix(basis_list[p - 1], x[:, k], b, sigma,
                                                                            position='last')

    # append core containing unit vectors
    cores.append(np.eye(m).reshape(m, m, 1, 1))

    return TT(cores)


def dPsix(psi_k, x, b, sigma, position='middle'):
    """
    Computes the k-th core of dPsi(x).

    Parameters
    ----------
    psi_k : list[Function]
        [psi_{k,1}, ... , psi_{k, n_k}]
    x : np.ndarray
        shape (d,)
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)
    position : {'first', 'middle', 'last'}, optional
        first core: k = 1
        middle core: 2 <= k <= p-1
        last core: k = p

    Returns
    -------
    np.ndarray
        k-th core of dPsi(x)
    """

    d = x.shape[0]
    nk = len(psi_k)
    psi_kx = [fun(x) for fun in psi_k]
    a = sigma(x) @ sigma(x).T

    partial_psi_kx = np.zeros((nk, d))

    for i in range(nk):
        partial_psi_kx[i, :] = psi_k[i].gradient(x)

    if position == 'middle':
        core = np.zeros((d + 2, nk, 1, d + 2))

        # diagonal
        for i in range(d + 2):
            core[i, :, 0, i] = psi_kx

        # 1. row
        core[0, :, 0, 1] = [_generator(fun, x, b, sigma) for fun in psi_k]
        core[0, :, 0, 2:] = partial_psi_kx

        # 2. column
        for i in range(2, d + 2):
            core[i, :, 0, 1] = [np.inner(a[i - 2, :], partial_psi_kx[row, :]) for row in range(nk)]

    elif position == 'first':
        core = np.zeros((1, nk, 1, d + 2))
        core[0, :, 0, 0] = psi_kx
        core[0, :, 0, 1] = [_generator(fun, x, b, sigma) for fun in psi_k]
        core[0, :, 0, 2:] = partial_psi_kx

    else:  # position == 'last'
        core = np.zeros((d + 2, nk, 1, 1))
        core[0, :, 0, 0] = [_generator(fun, x, b, sigma) for fun in psi_k]
        core[1, :, 0, 0] = psi_kx

        for i in range(2, d + 2):
            core[i, :, 0, 0] = [np.inner(a[i - 2, :], partial_psi_kx[row, :]) for row in range(nk)]

    return core


def generator_on_product(basis_list, s, x, b, sigma):
    """
    Evaluate the Koopman generator operating on the following function
    f = basis_list[1][s[1]] * ... * basis_list[p][s[p]]
    in x.

    Parameters
    ----------
    basis_list : list[list[Function]]
    s : tuple
        indices of basis functions
    x : np.ndarray
        shape(d,)
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)

    Returns
    -------
    float
    """

    p = len(s)
    a = sigma(x) @ sigma(x).T

    out = 0
    for j in range(p):
        product = 1
        for l in range(p):
            if l == j:
                continue
            product *= basis_list[l][s[l]](x)
        out += product * _generator(basis_list[j][s[j]], x, b, sigma)

        for v in range(j + 1, p):
            product = 1
            for l in range(p):
                if l == j or l == v:
                    continue
                product *= basis_list[l][s[l]](x)
            out += product * _frob_inner(a, np.outer(basis_list[v][s[v]].gradient(x), basis_list[j][s[j]].gradient(x)))
    return out


def _frob_inner(a, b):
    """
    Frobenius inner product of matrices a and b.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray

    Returns
    -------
    np.ndarray
    """

    return np.trace(np.inner(a, b))


def _generator(f, x, b, sigma):
    """
    Infinitesimal Koopman Generator applied to f.

    Computes Lf(x) = b(x) cdot nabla f(x) + 0.5 a(x) : nabla^2 f(x).

    Parameters
    ----------
    f : Function
    x : np.ndarray
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)

    Returns
    -------
    float
    """

    a = sigma(x) @ sigma(x).T
    return np.inner(b(x), f.gradient(x)) + 0.5 * _frob_inner(a, f.hessian(x))


def _amuset(us, v, dpsi):
    """
    Calculate the Matrix M in AMUSEt.

    Parameters
    ----------
    us : TT
        tensor u.rank_tensordot(s_inv, mode='last'), where u, s from svd of transformed data tensor
    v : TT
        tensor v from svd of transformed data tensor
    dpsi : TT

    Returns
    -------
    np.ndarray
        Matrix M in AMUSEt
    """
    p = dpsi.order - 1

    M = dpsi.tensordot(v.rank_transpose(), 1, mode='last-first')
    M.tensordot(us, p, mode='first-first', overwrite=True)
    M = M.cores[0][:, 0, 0, :]
    return M


def _amuset_special(us, v, dpsi):
    """
    Calculate the Matrix M in AMUSEt using the special tensordot.

    DOES ONLY WORK FOR CHUNK_SIZE=1 !!!

    Parameters
    ----------
    us : TT
        tensor u.rank_tensordot(s_inv, mode='last'), where u, s from svd of transformed data tensor
    v : TT
        tensor v from svd of transformed data tensor
    dpsi : TT

    Returns
    -------
    np.ndarray
        Matrix M in AMUSEt
        """
    vdpsi = dpsi.tensordot(v.rank_transpose(), 1, mode='last-first')
    return _contract_dPsi_u(vdpsi, us)


def _contract_dPsi_u(vdpsi, us):
    """
    Index contraction between (V^T dPsi^T) and (U Sigma^-1) in AMUSE.

    DOES ONLY WORK FOR CHUNK_SIZE=1 !!!
    Same as vdpsi.tensordot(us, p, mode='first-first') but using _special_tensordot and _special_kron.
    As it is a complete contraction over both, we return the resulting Matrix M and not a TT.

    Parameters
    ----------
    vdpsi : TT
    us : TT

    Returns
    -------
    np.ndarray
    """
    num_axes = vdpsi.order

    # calculate the contraction
    M = _special_kron(vdpsi.cores[0], us.cores[0])
    # M.shape (r_0, r_1, s_0, s_1)

    for i in range(1, num_axes - 1):
        M_new = _special_kron(vdpsi.cores[i], us.cores[i])
        # M_new.shape (r_{i-1}, r_i, s_{i-1}, s_i)
        M = _special_tensordot(M, M_new)

    # in the last core we cannot use the special tensordot because the last core of (V^T dPsi^T) has rank r
    M_new = np.tensordot(vdpsi.cores[-1], us.cores[-1], axes=([1, 2], [1, 2]))
    # M_new.shape (r_{i-1}, r_i, s_{i-1}, s_i)
    M = np.tensordot(M, M_new, axes=([1, 3], [0, 2]))  # shape (r_0, s_0, r_i, s_i)
    M = np.transpose(M, [0, 2, 1, 3])  # shape (r_0, r_i, s_0, s_i)

    return M[0, :, 0, :]


def _special_tensordot(A, B):
    """
    Tensordot between arrays A and B with special structure.

    A and B have the structure that arises in the cores of dPsi. As A and B can be the result of a Kronecker product,
    the entries of A and B can be matrices themselves. Thus A and B are modeled as 4D Arrays where the first and second
    index refer to the rows and columns of A and B. The third and fourth index refer to the rows and colums of the
    entries of A and B.
    All nonzero elements of A and B are
    in the diagonal [i,i,:,:], in the first row [0,:,:,:] and in the second column [:,1,:,:].
    Furthermore A and B are quadratic (A.shape[0] = A.shape[1]) or A has only one row or B only one column.
    The tensordot is calculated along both column dimensions of A (1,3) and both row dimensions of B (0,2).
    The resulting array has the same structure as A and B.

    Parameters
    ----------
    A : np.ndarray
    B : np.ndarray

    Returns
    -------
    np.ndarray
        tensordot between A and B along the axis ((1,3), (0,2))

    """
    C = np.zeros((A.shape[0], B.shape[1], A.shape[2], B.shape[3]))

    if B.shape[1] > 1:  # B is a matrix
        # diagonal
        for i in range(min(A.shape[0], B.shape[1])):
            C[i, i, :, :] = A[i, i, :, :] @ B[i, i, :, :]

        # entry(0, 1)
        for i in range(A.shape[1]):
            C[0, 1, :, :] += A[0, i, :, :] @ B[i, 1, :, :]

        # first row
        for i in range(2, B.shape[0]):
            C[0, i, :, :] = A[0, 0, :, :] @ B[0, i, :, :] + A[0, i, :, :] @ B[i, i, :, :]

        # second column
        for i in range(2, A.shape[0]):
            C[i, 1, :, :] = A[i, 1, :, :] @ B[1, 1, :, :] + A[i, i, :, :] @ B[i, 1, :, :]

    else:  # B is a vector
        # entry(0, 0)
        for i in range(A.shape[1]):
            C[0, 0, :, :] += A[0, i, :, :] @ B[i, 0, :, :]

        if A.shape[0] > 1:
            # entry(1, 0)
            C[1, 0, :, :] = A[1, 1, :, :] @ B[1, 0, :, :]

            # others
            for i in range(2, A.shape[0]):
                C[i, 0, :, :] = A[i, 1, :, :] @ B[1, 0, :, :] + A[i, i, :, :] @ B[i, 0, :, :]

    return C


def _special_kron(dPsi, B):
    """
    Build Matrix M_new from AMUSE.

    dPsi and B are TT cores (np.ndarray with order 4).
    dPsi needs to have the special structure of the dPsi(x) cores.

    dPsi : np.ndarray
    B : np.ndarray

    Returns
    -------
    np.ndarray
        shape = (dPsi.shape[0], dPsi.shape[3], B.shape[0], B.shape[3])
    """

    C = np.zeros((dPsi.shape[0], dPsi.shape[3], B.shape[0], B.shape[3]))

    if dPsi.shape[0] == 1 or dPsi.shape[3] == 1:  # dPsi is a vector
        for i in range(dPsi.shape[0]):
            for j in range(dPsi.shape[3]):
                C[i, j, :, :] = np.tensordot(dPsi[i, :, 0, j], B[:, :, 0, :], axes=(0, 1))
    else:  # dPsi is a matrix
        # diagonal
        psi = dPsi[0, :, 0, 0]
        psi = np.tensordot(psi, B[:, :, 0, :], axes=(0, 1))
        for i in range(C.shape[0]):
            C[i, i, :, :] = psi

        # first row
        for i in range(1, C.shape[1]):
            C[0, i, :, :] = np.tensordot(dPsi[0, :, 0, i], B[:, :, 0, :], axes=(0, 1))

        # 2nd column
        for i in range(2, C.shape[0]):
            C[i, 1, :, :] = np.tensordot(dPsi[i, :, 0, 1], B[:, :, 0, :], axes=(0, 1))

    return C


def _is_special(A):
    """
    Check if ndarray A has the special structure from the dpsi cores.

    A.shape = (outer rows, outer coumns, inner rows, inner colums). (not a TT core!)

    Parameters
    ----------
    A : np.ndarray
       A.ndim = 4

    Returns
    -------
    bool
    """
    if not A.ndim == 4:
        raise ValueError('A should have ndim = 4')

    if A.shape[0] == A.shape[1]:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == j or i == 0 or j == 1:
                    continue
                if not (A[i, j, :, :] == 0).all():
                    return False

    elif not (A.shape[0] == 1 or A.shape[1] == 1):
        return False

    return True
