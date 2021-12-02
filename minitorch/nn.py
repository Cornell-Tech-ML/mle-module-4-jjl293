from numpy.lib.arraysetops import isin
from minitorch import Tensor
from minitorch.tensor_data import TensorData
from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators


def tile(input, kernel) -> 'tuple[Tensor, int, int]':
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    out = input.contiguous().view(batch, channel, height, new_width, kw)
    out = out.permute(0, 1, 3, 2, 4)
    out = out.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return out, new_height, new_width


def avgpool2d(input: 'Tensor', kernel) -> 'Tensor':
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, _, _ = input.shape
    input, new_height, new_width = tile(input, kernel)
    out = input.mean(dim=4)

    return out.view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: 'Tensor', dim: int) -> 'Tensor':
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax

    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    return input == max_reduce(input, dim)


class Max(Function):
    @staticmethod
    def forward(ctx, input: 'Tensor', dim: int) -> 'Tensor':
        "Forward of max should be max reduction"
        if dim is None:
            assert False, "dims was None"
            # out = max_reduce(input, input.dims()).view(1)
            # dim = input.dims()[0]
        else:
            out = max_reduce(input, dim)
        ctx.save_for_backward(input, dim)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_values
        # return argmax(input, dim) * grad_output
        t = rand(input.shape)
        return argmax(t + input, dim) * grad_output


max = Max.apply


def softmax(input: 'Tensor', dim: int) -> 'Tensor':
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    a = input.exp()
    return a / a.sum(dim=dim)


def logsoftmax(input: 'Tensor', dim: int) -> 'Tensor':
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    max_i = max(input, dim)
    return input - (input - max_i).exp().sum(dim=dim).log() - max_i


def maxpool2d(input, kernel) -> 'Tensor':
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape

    input, new_h, new_w = tile(input, kernel)
    out = max(input, 4)

    return out.view(batch, channel, new_h, new_w)


def dropout(input, rate, ignore=False) -> 'Tensor':
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with randoom positions dropped out
    """
    if ignore:
        return input
    m = rand(input.shape)
    return input * (m < (1 - rate))
