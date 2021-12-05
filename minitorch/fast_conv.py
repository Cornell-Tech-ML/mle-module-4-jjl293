import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


@njit(parallel=True)
def tensor_conv1d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    for i in prange(out_size):
        out_index = np.empty(len(out_shape), np.uint16)
        idx = (out_size - i - 1) if reverse else i
        to_index(idx, out_shape, out_index)
        current_batch, current_out_channel, current_width = out_index
        out_pos = (
            current_batch * out_strides[0]
            + current_out_channel * out_strides[1]
            + current_width * out_strides[2]
        )
        for current_in_channel in range(in_channels):
            for current_kw in range(kw):
                if reverse:
                    current_kw = kw - current_kw -1
                weight_pos = (
                    current_out_channel * weight_strides[0]  +
                    current_in_channel * weight_strides[1] +
                    current_kw * weight_strides[2]
                )
                input_val = 0
                if reverse and 0 <= (current_width - current_kw):
                    input_val = input[
                        current_batch * input_strides[0] +
                        current_in_channel * input_strides[1] +
                        (current_width - current_kw) * input_strides[2]
                    ]
                elif not reverse and (current_width + current_kw) < width:
                    input_val = input[
                        current_batch * input_strides[0] +
                        current_in_channel * input_strides[1] +
                        (current_width + current_kw) * input_strides[2]
                    ]
                out[out_pos] += input_val * weight[weight_pos]
    

class Conv1dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


@njit(parallel=True, fastmath=True)
def tensor_conv2d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    for i in prange(out_size):
        out_index = np.empty(len(out_shape), np.uint16)
        idx = (out_size - i - 1) if reverse else i
        to_index(idx, out_shape, out_index)
        current_batch, current_out_channel, current_height, current_width  = out_index
        out_pos = (
            current_batch * out_strides[0] + current_out_channel * out_strides[1]
            + current_height * out_strides[2] + current_width * out_strides[3]
        )
        for current_in_channel in range(in_channels):
            for current_kh in range(kh):
                for current_kw in range(kw):
                    if reverse:
                        current_kh = kh - current_kh -1
                        current_kw = kw - current_kw -1
                    weight_pos = (
                        current_out_channel * weight_strides[0] + current_in_channel * weight_strides[1] +
                        current_kh * weight_strides[2] + current_kw * weight_strides[3]
                    )
                    input_val = 0
                    if reverse and 0 <= (current_height - current_kh) and 0 <= (current_width - current_kw):
                        input_val = input[
                            current_batch * input_strides[0] +
                            current_in_channel * input_strides[1] +
                            (current_height - current_kh) * input_strides[2] +
                            (current_width - current_kw) * input_strides[3]
                        ]
                    elif not reverse and (current_height + current_kh) < height and (current_width + current_kw) < width:
                        input_val = input[
                            current_batch * input_strides[0] +
                            current_in_channel * input_strides[1] +
                            (current_height + current_kh) * input_strides[2] +
                            (current_width + current_kw) * input_strides[3]
                        ]
                    out[out_pos] += input_val * weight[weight_pos]


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
