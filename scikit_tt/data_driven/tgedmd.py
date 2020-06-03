import numpy as np
from scikit_tt import TT
from scikit_tt.data_driven.transform import basis_decomposition, Function
from scikit_tt.data_driven.transform import __hocur_find_li_cols, __hocur_maxvolume
import scikit_tt.utils as utl


def amuset_hosvd(data_matrix, basis_list, b, sigma, num_eigvals=np.infty, threshold=1e-2, max_rank=np.infty, return_option='eigentensors',
                 chunk_size=None):
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
        threshold for svd of psi
    max_rank : int, optional
        maximal rank of TT representations of psi and dpsi after svd/ortho
    return_option : {'eigentensors', 'eigenfunctionevals'}
        'eigentensors': return a list of the eigentensors of the koopman generator
        'eigenfunctionevals': return the evaluations of the eigenfunctions of the koopman generator at all snapshots
     chunk_size : int or None, optional
        if a chunk_size is specified, dPsi(X) is built in chunks

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
    # SVD of psi
    u, s, v = psi.svd(psi.order - 1, threshold=threshold, max_rank=max_rank, ortho_l=True, ortho_r=False)
    s_inv = 1.0 / s
    s = np.diag(s)
    s_inv = np.diag(s_inv)
    psi = u.rank_tensordot(s)
    psi.concatenate(v, overwrite=True)  # rank reduced version

    print('calculating dpsi...')
    if chunk_size is None:
        dpsi = tt_decomposition(data_matrix, basis_list, b, sigma)
        dpsi = dpsi.ortho_left(threshold=threshold, max_rank=max_rank)
    else:
        dpsi = tt_decomposition_chunks(data_matrix, basis_list, b, sigma, threshold, max_rank, chunk_size)
    p = dpsi.order - 1

    # calculate M
    print('calculating matrix M for AMUSE...')
    M = dpsi.tensordot(v.rank_transpose(), 1, mode='last-first')

    u.rank_tensordot(s_inv, mode='last', overwrite=True)
    M.tensordot(u, p, mode='first-first', overwrite=True)
    # M.rank_transpose(overwrite=True)
    M = M.cores[0][:, 0, 0, :]

    print('calculating eigenvalues and eigentensors...')
    # calculate eigenvalues of M
    eigvals, eigvecs = np.linalg.eig(M)
    if not (eigvals < 0).all():
        print('WARNING: there were eigenvalues >= 0, which have been removed')
        eigvals = eigvals[eigvals < 0]

    sorted_indices = np.argsort(-eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
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


def _tt_decomposition_one_chunk(x, basis_list, b, sigma, start_chunk, m_total):
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
    Calculates dPsi(X).

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


def amuset_hocur(data_matrix, basis_list, b, sigma, max_rank=1000, multiplier=2):
    """

    Parameters
    ----------
    data_matrix
    basis_list
    b
    sigma
    max_rank
    multiplier

    Returns
    -------

    """

    pass


def hocur(x, basis_list, b, sigma, ranks, repeats=1, multiplier=10):
    """
    Higher-order CUR decomposition of dPsi(X).

    Given a snapshot matrix x and a list of basis functions in each mode, construct a TT decomposition dPsi(X)
    using a higher-order CUR decomposition and maximum-volume subtensors.

    Parameters
    ----------
    x : np.ndarray
        data matrix
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)
    ranks : list[int] or int
        maximum TT ranks of the resulting TT representation; if type is int, then the ranks are set to
        [1, ranks, ..., ranks, 1]; note that - depending on the number of linearly independent rows/columns that have
        been found - the TT ranks may be reduced during the decomposition
    repeats : int, optional
        number of repeats, default is 1
    multiplier : int, optional
        multiply the number of initially chosen column indices (given by ranks) in order to increase the probability of
        finding a 'full' set of linearly independent columns; default is 10

    Returns
    -------
    TT
        TT representation of dPsi(X)
    """

    m = x.shape[1]

    # number of modes
    p = len(basis_list)

    # mode dimensions
    n = [len(basis_list[k]) for k in range(p)] + [m]

    # ranks
    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(n) - 1)] + [1]

    # initial definitions
    # -------------------

    # define initial lists of column indices
    col_inds = __hocur_first_col_inds(n, ranks, multiplier)

    # define list of cores
    cores = [None] * (p + 1)

    # start decomposition
    # -------------------

    for k in range(repeats):

        row_inds = [None]

        # first half sweep
        for i in range(p):

            # extract submatrix
            y = __hocur_extract_matrix(x, basis_list, b, sigma, row_inds[i], col_inds[i])

            if k == 0:
                # find linearly independent columns
                cols = __hocur_find_li_cols(y)
                cols = cols[:ranks[i + 1]]
                y = y[:, cols]

            # find optimal rows
            rows = __hocur_maxvolume(y)

            # adapt ranks if necessary
            ranks[i + 1] = len(rows)

            if i == 0:

                # store row indices for first dimensions
                row_inds.append([[rows[j]] for j in range(ranks[i + 1])])

            else:

                # convert rows to multi indices
                multi_indices = np.array(np.unravel_index(rows, (ranks[i], n[i])))

                # store row indices for dimensions m_1, n_1, ..., m_i, n_i
                row_inds.append([row_inds[i][multi_indices[0, j]] + [multi_indices[1, j]] for j in
                                 range(ranks[i + 1])])

            # define core
            if len(rows) < y.shape[1]:
                y = y[:, :len(rows)]
            u_inv = np.linalg.inv(y[rows, :].copy())
            cores[i] = y.dot(u_inv).reshape([ranks[i], n[i], 1, ranks[i + 1]])

        # second half sweep
        for i in range(p, 0, -1):

            # extract submatrix
            y = __hocur_extract_matrix(x, basis_list, b, sigma, row_inds[i], col_inds[i]).reshape([ranks[i], n[i] * ranks[i + 1]])

            # find optimal rows
            cols = __hocur_maxvolume(y.T)

            # adapt ranks if necessary
            ranks[i] = len(cols)

            if i == p:

                # store row indices for first dimensions
                col_inds[p - 1] = [[cols[j]] for j in range(ranks[i])]

            else:

                # convert cols to multi indices
                multi_indices = np.array(np.unravel_index(cols, (n[i], ranks[i + 1])))

                # store col indices for dimensions m_i, n_i, ... , m_d, n_d
                col_inds[i - 1] = [[multi_indices[0, j]] + col_inds[i][multi_indices[1, j]] for j in range(ranks[i])]

            # define TT core
            if len(cols) < y.shape[0]:
                y = y[:len(cols), :]
            u_inv = np.linalg.inv(y[:, cols].copy())
            cores[i] = u_inv.dot(y).reshape([ranks[i], n[i], 1, ranks[i + 1]])

        # define first core
        y = __hocur_extract_matrix(x, basis_list, b, sigma, None, col_inds[0])
        cores[0] = y.reshape([1, n[0], 1, ranks[1]])

    # construct tensor train
    # ----------------------

    psi = TT(cores)

    return psi


def __hocur_first_col_inds(dimensions, ranks, multiplier):
    # todo: max rank needs to be m(d+2) instead of m
    """
    Create random column indices.

    Parameters
    ----------
    dimensions : list[int]
        dimensions of a given tensor
    ranks : list[int]
        ranks for decomposition, has to be smaller than the last dimension
    multiplier : int
        multiply the number of initially chosen column indices (given by ranks) in order to increase the probability of
        finding a 'full' set of linearly independent columns

    Returns
    -------
    col_inds : list[list[int]]
        array containing single indices
    """

    if max(ranks) > dimensions[-1]:
        raise ValueError('ranks need to be smaller than last dimension')

    # define list of column indices
    col_inds = [None]

    # insert column indices for last dimension
    col_inds.insert(0, [[j] for j in range(np.minimum(multiplier * ranks[-2], dimensions[-1]))])

    for i in range(len(dimensions) - 3, -1, -1):
        # define array of flat indices
        flat_inds = np.arange(np.minimum(multiplier * ranks[i + 1], dimensions[i + 1] * ranks[i + 2]))

        # convert flat indices to tuples
        multi_inds = np.array(np.unravel_index(flat_inds, (dimensions[i + 1], ranks[i + 2])))

        # insert column indices
        col_inds.insert(0, [[multi_inds[0, j]] + col_inds[0][multi_inds[1, j]] for j in range(multi_inds.shape[1])])

    return col_inds


def __hocur_extract_matrix(data, basis_list, b, sigma, row_coordinates_list, col_coordinates_list):
    """
    Extraction of a submatrix of dPsi(X).

    Given a set of row and column coordinates, extracts a submatrix from dPsi(X) corresponding to
    the data matrix x and the set of basis functions stored in basis_list.

    Parameters
    ----------
    data : np.ndarray
        data matrix
    basis_list : list[list[Function]]
        list of basis functions in every mode
    b : function
        drift, b:R^d -> R^d
    sigma : function
        diffusion, sigma: R^d -> R^(d,d)
    row_coordinates_list : list[list[int]]
        list of row indices
    col_coordinates_list : list[list[int]]
        list of column indices

    Returns
    -------
    np.ndarray
        extracted matrix
    """

    if row_coordinates_list is None:
        current_mode = len(basis_list[0])
        n_rows = 1
        n_cols = len(col_coordinates_list)

        matrix = np.zeros([current_mode, n_cols])
        for j in range(n_cols):
            col_coordinates = col_coordinates_list[j]
            snapshot = data[:, col_coordinates[-1]]
            for l in range(current_mode):
                s = [l] + col_coordinates[:-1]
                matrix[l, j] = generator_on_product(basis_list, s, snapshot, b, sigma)

    elif col_coordinates_list is None:
        current_mode = data.shape[1]
        n_rows = len(row_coordinates_list)
        n_cols = 1

        matrix = np.zeros([n_rows * current_mode, 1])
        for i in range(n_rows):
            row_coordinates = row_coordinates_list[i]
            for l in range(current_mode):
                s = row_coordinates
                snapshot = data[:, l]
                matrix[i * current_mode + l, 0] = generator_on_product(basis_list, s, snapshot, b, sigma)

    else:
        n_rows = len(row_coordinates_list)
        n_cols = len(col_coordinates_list)
        current_index = len(row_coordinates_list[0])
        current_mode = len(basis_list[current_index])

        matrix = np.zeros([n_rows * current_mode, n_cols])
        for j in range(n_cols):
            for i in range(n_rows):
                col_coordinates = col_coordinates_list[j]
                row_coordinates = row_coordinates_list[i]
                snapshot = data[:, col_coordinates[-1]]
                for l in range(current_mode):
                    s = row_coordinates + [l] + col_coordinates[:-1]
                    matrix[i * current_mode + l, j] = generator_on_product(basis_list, s, snapshot, b, sigma)

    return matrix


# def amuset_reversible_exact(data_matrix, basis_list, sigma, threshold=1e-2):
#     """
#     AMUSE algorithm for calculation of eigenvalues of the Koopman generator. Assuming reversible dynamics.
#     The tensor-trains are created using the exact TT decomposition.
#
#     Parameters
#     ----------
#     data_matrix : np.ndarray
#         snapshot matrix, shape (d, m)
#     basis_list : list of (list of Function)
#         list of basis functions in every mode
#     sigma : function
#         diffusion, sigma: R^d -> R^(d,d)
#     threshold : float
#         threshold for svd of psi
#
#     Returns
#     -------
#     eigvals : np.ndarray
#         eigenvalues of Koopman generator
#     eigtensors : list of TT
#         eigentensors of Koopman generator
#     """
#
#     # calculate psi and dpsi
#     print('calculating psi...')
#     psi = basis_decomposition(data_matrix, basis_list)
#     print('calculating dpsi...')
#     dpsi = tt_decomposition_reversible(data_matrix, basis_list, sigma)
#     p = psi.order - 1
#     dpsi.ortho(threshold)
#     print('dpsi cores:')
#     for core in dpsi.cores:
#         print(core.shape)
#     print('')
#
#     # SVD of psi
#     print('calculating svd of psi...')
#     u, s, v = psi.svd(psi.order - 1, threshold=threshold)
#     s_inv = 1.0 / s
#     s = np.diag(s)
#     s_inv = np.diag(s_inv)
#
#     # calculate M
#     print('calculating matrix M for AMUSE...')
#     u.rank_tensordot(s_inv, overwrite=True)
#     M = -0.5 * u.tensordot(dpsi, p, mode='first-first')
#     M.rank_transpose(overwrite=True)
#     for core in M.cores:
#         print(core.shape)
#
#     M.tensordot(dpsi, 2, mode='last-last', overwrite=True)  # MemoryError: cannot resize list (m=5000)
#     M.rank_transpose(overwrite=True)
#     M.tensordot(u, p, mode='first-first', overwrite=True)
#     M = M.cores[0]
#     M = np.squeeze(M)
#
#     print('calculating eigenvalues and eigentensors...')
#     # calculate eigenvalues of M
#     eigvals, eigvecs = np.linalg.eig(M)
#
#     # calculate eigentensors
#     eigvecs = eigvecs[:, :, np.newaxis]
#     eigtensors = []
#     for i in range(eigvals.shape[0]):
#         eigtensor = u.copy()
#         eigtensor.cores[-1] = np.tensordot(eigtensor.cores[-1], eigvecs[:, i, :], axes=([3], [0]))
#         eigtensor.ranks = [eigtensor.cores[i].shape[0] for i in range(eigtensor.order)] + [eigtensor.cores[-1].shape[3]]
#         eigtensors.append(eigtensor)
#
#     return eigvals, eigtensors
#
#
# def tt_decomposition_reversible(x, basis_list, sigma):
#     """
#     Calculates dPsi(X).
#
#     Parameters
#     ----------
#     x : np.ndarray
#         snapshot matrix of size d x m
#     basis_list: list of (list of Function)
#         list of basis functions in every mode, classes need to have methods "__call" and "partial"
#     sigma : function
#         diffusion, sigma(x[:,k]) should have shape (d, d)
#
#     Returns
#     -------
#     dPsiX: TT
#         tensor train of basis function evaluations
#     """
#
#     # number of snapshots
#     m = x.shape[1]
#     # dimension
#     d = x.shape[0]
#     # number of modes
#     p = len(basis_list)
#     # mode dimensions
#     n = [len(basis_list[i]) for i in range(p)]
#
#     # define cores 1,...,(p+1) as a list of empty arrays
#     cores = [np.zeros([1, n[0], 1, m * (d + 1)])] + \
#             [np.zeros([m * (d + 1), n[i], 1, m * (d + 1)]) for i in range(1, p - 1)] + \
#             [np.zeros([m * (d + 1), n[p - 1], 1, m * d])] + \
#             [np.zeros([m * d, d, 1, m])]
#
#     # insert elements of core 1
#     cores[0] = np.concatenate([dPsix_reversible(basis_list[0], x[:, k], position='first') for k in range(m)],
#                               axis=3)
#
#     # insert elements of cores 2,...,p-1
#     for i in range(1, p - 1):
#         for k in range(m):
#             cores[i][k * (d + 1): (k + 1) * (d + 1), :, :, k * (d + 1): (k + 1) * (d + 1)] = dPsix_reversible(
#                 basis_list[i], x[:, k], position='middle')
#
#     # insert elements of core p
#     for k in range(m):
#         cores[p - 1][k * (d + 1): (k + 1) * (d + 1), :, :, k * d: (k + 1) * d] = dPsix_reversible(
#             basis_list[p - 1], x[:, k], position='last')
#
#     # insert elements of core p + 1
#     for k in range(m):
#         cores[p][k * d: (k + 1) * d, :, 0, k] = sigma(x[:, k])
#
#     # append core containing unit vectors
#     cores.append(np.eye(m).reshape(m, m, 1, 1))
#
#     dPsiX = TT(cores)
#     return dPsiX
#
#
# def dPsix_reversible(psi_k, x, position='middle'):
#     """
#     Computes the k-th core of dPsi(x).
#
#     Parameters
#     ----------
#     psi_k : list of Function
#         [psi_{k,1}, ... , psi_{k, n_k}]
#     x : np.ndarray
#         shape (d,)
#     position: {'first', 'middle', 'last'}, optional
#         first core: k = 1
#         middle core: 2 <= k <= p-1
#         last core: k = p
#
#     Returns
#     -------
#     np.ndarray
#         k-th core of dPsi(x)
#     """
#
#     d = x.shape[0]
#     nk = len(psi_k)
#     psi_kx = [fun(x) for fun in psi_k]
#
#     if position == 'middle':
#         core = np.zeros((d + 1, nk, 1, d + 1))
#
#         # diagonal
#         for i in range(d + 1):
#             core[i, :, 0, i] = psi_kx
#
#         # partials
#         for i in range(1, d + 1):
#             core[0, :, 0, i] = [fun.partial(x, i - 1) for fun in psi_k]
#
#     elif position == 'first':
#         core = np.zeros((1, nk, 1, d + 1))
#         core[0, :, 0, 0] = psi_kx
#         for i in range(1, d + 1):
#             core[0, :, 0, i] = [fun.partial(x, i - 1) for fun in psi_k]
#
#     else:
#         core = np.zeros((d + 1, nk, 1, d))
#         for i in range(d):
#             core[0, :, 0, i] = [fun.partial(x, i) for fun in psi_k]
#             core[i + 1, :, 0, i] = psi_kx
#
#     return core


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
