import numpy as np
from scikit_tt import TT
from scikit_tt.data_driven.transform import basis_decomposition, Function


def amuset_reversible_exact(data_matrix, basis_list, sigma, threshold=1e-2):
    """
    AMUSE algorithm for calculation of eigenvalues of the Koopman generator. Assuming reversible dynamics.
    The tensor-trains are created using the exact TT decomposition.

    Parameters
    ----------
    data_matrix: ndarray
        snapshot matrix, shape (d, m)
    basis_list: list of lists of instances of function-classes
        list of basis functions in every mode, classes need to have methods "__call" and "partial"
    sigma: diffusion
        sigma(data_matrix[:, k]) should have shape (d, d)
    threshold: float
        threshold for svd of psi

    Returns
    -------
    eigvals: ndarray
        eigenvalues of Koopman generator
    eigtensors: list of instances of TT class
        eigentensors of Koopman generator
    """

    # calculate psi and dpsi
    print('calculating psi...')
    psi = basis_decomposition(data_matrix, basis_list)
    print('calculating dpsi...')
    dpsi = tt_decomposition_reversible(data_matrix, basis_list, sigma)
    p = psi.order - 1
    dpsi.ortho(threshold)
    print('dpsi cores:')
    for core in dpsi.cores:
        print(core.shape)
    print('')

    # SVD of psi
    print('calculating svd of psi...')
    u, s, v = psi.svd(psi.order - 1, threshold=threshold)
    s_inv = 1.0 / s
    s = np.diag(s)
    s_inv = np.diag(s_inv)

    # calculate M
    print('calculating matrix M for AMUSE...')
    u.rank_tensordot(s_inv, overwrite=True)
    M = -0.5 * u.tensordot(dpsi, p, mode='first-first')
    M.rank_transpose(overwrite=True)
    for core in M.cores:
        print(core.shape)

    M.tensordot(dpsi, 2, mode='last-last', overwrite=True)  # MemoryError: cannot resize list (m=5000)
    M.rank_transpose(overwrite=True)
    M.tensordot(u, p, mode='first-first', overwrite=True)
    M = M.cores[0]
    M = np.squeeze(M)

    print('calculating eigenvalues and eigentensors...')
    # calculate eigenvalues of M
    eigvals, eigvecs = np.linalg.eig(M)

    # calculate eigentensors
    eigvecs = eigvecs[:, :, np.newaxis]
    eigtensors = []
    for i in range(eigvals.shape[0]):
        eigtensor = u.copy()
        eigtensor.cores[-1] = np.tensordot(eigtensor.cores[-1], eigvecs[:, i, :], axes=([3], [0]))
        eigtensor.ranks = [eigtensor.cores[i].shape[0] for i in range(eigtensor.order)] + [eigtensor.cores[-1].shape[3]]
        eigtensors.append(eigtensor)

    return eigvals, eigtensors


def tt_decomposition(x, basis_list, b, sigma):
    """
    Calculates dPsi(X).

    Parameters
    ----------
    x: np.ndarray
        snapshot matrix of size d x m
    basis_list: list of lists of instances of function-classes
        list of basis functions in every mode, classes need to have methods "__call__", "partial" and "hessian"
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)

    Returns
    -------
    dPsiX: TT
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
        cores[p - 1][k * (d + 2): (k + 1) * (d + 2), :, :, k] = dPsix(basis_list[p - 1], x[:, k], b, sigma,
                                                                      position='last')

    # append core containing unit vectors
    cores.append(np.eye(m).reshape(m, m, 1, 1))

    dPsiX = TT(cores)
    return dPsiX


def dPsix(psi_k, x, b, sigma, position='middle'):
    """
    Computes the k-th core of dPsi(x).

    Parameters
    ----------
    psi_k: list of instances of function-classes
        [psi_{k,1}, ... , psi_{k, n_k}]
    x: np.ndarray
        shape (d,)
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)
    position: 'first', 'middle' or 'last', default=None
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
    for i in range(d):
        partial_psi_kx[:, i] = [fun.partial(x, i) for fun in psi_k]

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


def tt_decomposition_reversible(x, basis_list, sigma):
    """
    Calculates dPsi(X).

    Parameters
    ----------
    x: np.ndarray
        snapshot matrix of size d x m
    basis_list: list of lists of instances of function-classes
        list of basis functions in every mode, classes need to have methods "__call" and "partial"
    sigma: function
        diffusion, sigma(x[:,k]) should have shape (d, d)

    Returns
    -------
    dPsiX: TT
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

    # define cores 1,...,(p+1) as a list of empty arrays
    cores = [np.zeros([1, n[0], 1, m * (d + 1)])] + \
            [np.zeros([m * (d + 1), n[i], 1, m * (d + 1)]) for i in range(1, p - 1)] + \
            [np.zeros([m * (d + 1), n[p - 1], 1, m * d])] + \
            [np.zeros([m * d, d, 1, m])]

    # insert elements of core 1
    cores[0] = np.concatenate([dPsix_reversible(basis_list[0], x[:, k], position='first') for k in range(m)],
                              axis=3)

    # insert elements of cores 2,...,p-1
    for i in range(1, p - 1):
        for k in range(m):
            cores[i][k * (d + 1): (k + 1) * (d + 1), :, :, k * (d + 1): (k + 1) * (d + 1)] = dPsix_reversible(
                basis_list[i], x[:, k], position='middle')

    # insert elements of core p
    for k in range(m):
        cores[p - 1][k * (d + 1): (k + 1) * (d + 1), :, :, k * d: (k + 1) * d] = dPsix_reversible(
            basis_list[p - 1], x[:, k], position='last')

    # insert elements of core p + 1
    for k in range(m):
        cores[p][k * d: (k + 1) * d, :, 0, k] = sigma(x[:, k])

    # append core containing unit vectors
    cores.append(np.eye(m).reshape(m, m, 1, 1))

    dPsiX = TT(cores)
    return dPsiX


def dPsix_reversible(psi_k, x, position='middle'):
    """
    Computes the k-th core of dPsi(x).

    Parameters
    ----------
    psi_k: list of instances of function-classes
        [psi_{k,1}, ... , psi_{k, n_k}]
    x: np.ndarray
        shape (d,)
    position: 'first', 'middle' or 'last', default=None
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

    if position == 'middle':
        core = np.zeros((d + 1, nk, 1, d + 1))

        # diagonal
        for i in range(d + 1):
            core[i, :, 0, i] = psi_kx

        # partials
        for i in range(1, d + 1):
            core[0, :, 0, i] = [fun.partial(x, i - 1) for fun in psi_k]

    elif position == 'first':
        core = np.zeros((1, nk, 1, d + 1))
        core[0, :, 0, 0] = psi_kx
        for i in range(1, d + 1):
            core[0, :, 0, i] = [fun.partial(x, i - 1) for fun in psi_k]

    else:
        core = np.zeros((d + 1, nk, 1, d))
        for i in range(d):
            core[0, :, 0, i] = [fun.partial(x, i) for fun in psi_k]
            core[i + 1, :, 0, i] = psi_kx

    return core


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
