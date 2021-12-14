from numba import cuda
import numba
from .tensor_data import (
    to_index,
    index_to_position,
    TensorData,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)
THREADS_PER_BLOCK = 32


def tensor_map(fn):
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out_storage (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        out_size (array): size for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out_storage, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.uint16)
            in_index = cuda.local.array(MAX_DIMS, numba.uint16)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out_position = index_to_position(out_index, out_strides)
            in_position = index_to_position(in_index, in_strides)
            out_storage[out_position] = fn(in_storage[in_position])

    return cuda.jit()(_map)


def map(fn):
    # CUDA compile your kernel
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)

        # Instantiate and run the cuda kernel.
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.
        out_storage (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out_storage,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.uint16)
            a_index = cuda.local.array(MAX_DIMS, numba.uint16)
            b_index = cuda.local.array(MAX_DIMS, numba.uint16)
            to_index(i, out_shape, out_index)
            out_position = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out_storage[out_position] = fn(a_storage[index_to_position(a_index, a_strides)],
                                           b_storage[index_to_position(b_index, b_strides)])

    return cuda.jit()(_zip)


def zip(fn):
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def _sum_practice(out_storage, a_storage, size):
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length :math:`n` and out of size :math:`n // blockDIM`
    it should sum up each blockDim values into an out cell.

    [a_1, a_2, ..., a_100]

    |

    [a_1 +...+ a_32, a_32 + ... + a_64, ... ,]

    Note: Each block must do the sum using shared memory!

    Args:
        out_storage (array): storage for `out` tensor.
        a_storage (array): storage for `a` tensor.
        size (int):  length of a.
    """
    BLOCK_DIM = 32
    local_index = cuda.threadIdx.x
    x = cuda.blockIdx.x * cuda.blockDim.x + local_index
    shared = cuda.shared.array(BLOCK_DIM, numba.float64)
    shared[local_index] = a_storage[x] if (x + 1) <= size else 0.
    cuda.syncthreads()
    reduction = 2
    while x < size and x % reduction == 0 and reduction <= BLOCK_DIM:
        shared[local_index] = shared[local_index] + shared[local_index + reduction // 2]
        cuda.syncthreads()
        reduction = min(BLOCK_DIM + 1, reduction * 2)
    if local_index == 0:
        out_storage[cuda.blockIdx.x] = shared[local_index]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a):
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.
        out_storage (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out_storage,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_dim,
        reduce_value,
    ):
        BLOCK_DIM = 1024

        shared = cuda.shared.array(BLOCK_DIM, numba.float64)
        local_index = cuda.threadIdx.x
        block_id = cuda.blockIdx.x
        out_index = cuda.local.array(MAX_DIMS, numba.uint16)
        to_index(block_id, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)
        out_index[reduce_dim] = local_index
        a_pos = index_to_position(out_index, a_strides)
        shared[local_index] = a_storage[a_pos] if local_index < a_shape[reduce_dim] else reduce_value
        cuda.syncthreads()
        reduction = 2
        while local_index % reduction == 0 and reduction <= BLOCK_DIM:
            shared[local_index] = fn(shared[local_index], shared[local_index + reduction // 2])
            cuda.syncthreads()
            reduction *= 2
        if local_index == 0:
            out_storage[out_pos] = shared[local_index]

    return cuda.jit()(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
        out_a = a.zeros(tuple(out_shape))

        threadsperblock = 1024
        blockspergrid = out_a.size
        f[blockspergrid, threadsperblock](
            *out_a.tuple(), out_a.size, *a.tuple(), dim, start
        )

        return out_a

    return ret


def _mm_practice(out, a, b, size):
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Compute ::

    for i:
        for j:
             for k:
                 out[i, j] += a[i, k] * b[k, j]

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        b (array): storage for `a` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    i = cuda.blockIdx.x * cuda.blockDim.x + local_i
    j = cuda.blockIdx.y * cuda.blockDim.y + local_j
    if i < size and j < size:
        sharedA = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
        sharedB = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
        t = 0
        for s in range(0, size, BLOCK_DIM):
            sharedA[local_i, local_j] = a[i * size + s + j]
            sharedB[local_i, local_j] = b[(s + i) * size + j]
            cuda.syncthreads()

            for k in range(BLOCK_DIM):
                t += sharedA[local_i, k] * sharedB[k, local_j]
        out[i * size + j] = t


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a, b):

    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


@cuda.jit()
def tensor_matrix_multiply(
    out_storage,
    out_shape,
    out_strides,
    out_size,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    CUDA tensor matrix multiply function.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out_storage (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    BLOCK_DIM = 32
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    i = cuda.blockIdx.x * cuda.blockDim.x + local_i
    j = cuda.blockIdx.y * cuda.blockDim.y + local_j
    block_z = cuda.blockIdx.z
    sharedA = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    sharedB = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    t = 0
    for s in range(0, a_shape[-1], BLOCK_DIM):  # 120
        a_position = block_z * a_batch_stride + j * a_strides[1] + (local_i + s) * a_strides[2]
        b_position = block_z * b_batch_stride + (local_j + s) * b_strides[1] + i * b_strides[2]
        sharedA[local_j, local_i] = a_storage[a_position] if j < a_shape[1] and (local_i + s) < a_shape[2] else 0
        sharedB[local_j, local_i] = b_storage[b_position] if (local_j + s) < b_shape[1] and i < b_shape[2] else 0
        cuda.syncthreads()

        for b in range(BLOCK_DIM):
            t += sharedA[local_j, b] * sharedB[b, local_i]

    if j < out_shape[1] and i < out_shape[2]:
        out_storage[block_z * out_strides[0] + j * out_strides[1] + i * out_strides[2]] = t


def matrix_multiply(a, b):
    """
    Tensor matrix multiply
    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::
        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    # One block per batch, extra rows, extra col
    blockspergrid = (
        (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        out.shape[0],
    )
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

    tensor_matrix_multiply[blockspergrid, threadsperblock](
        *out.tuple(), out.size, *a.tuple(), *b.tuple()
    )

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class CudaOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
