# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest
import numpy as np
from tvm import te, topi
from tvm.tir.stmt_functor import post_order_visit

import tvm.testing
from tvm.topi import testing
from tvm.contrib.hexagon.build import HexagonLauncher
from .conftest import requires_hexagon_toolchain
import tvm.topi.hexagon.slice_ops as sl

input_layout = tvm.testing.parameter(
    "nhwc-8h2w32c2w-1d",
)

output_layout = tvm.testing.parameter(
    "nhwc-8h2w32c2w-1d",
)

def transform_numpy(arr_np, layout):
    if layout == "nhwc":
        return arr_np
    elif layout in ["nhwc-8h2w32c2w-1d", "nhwc-8h2w32c2w-2d"]:
        N, H, W, C = arr_np.shape
        return arr_np.reshape([N, H // 8, 8, W // 4, 2, 2, C // 32, 32]).transpose(0, 1, 3, 6, 2, 4, 7, 5)
    else:
        raise RuntimeError(f"Unexpected layout '{layout}'")

def fp16_layout_transform_1d(n, h, w, c):
    return [
        n,
        c // 32,
        h // 8,
        w // 4,
        h % 8,
        (w % 4) // 2,
        c % 32,
        w %2
    ]

def fp16_layout_transform_2d(n, h, w, c):
    return [
        n,
        c // 32,
        h // 8,
        w // 4,
        te.AXIS_SEPARATOR,
        h % 8,
        (w % 4) // 2,
        c % 32,
        w %2
    ]

@tvm.testing.fixture
def input_np(input_shape, dtype):
    return (100 * np.random.uniform(size=input_shape)).astype(dtype)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    return transform_numpy(expected_output_np, output_layout)


@tvm.testing.fixture
def transformed_input_np_padded(input_np_padded, input_layout):
    return transform_numpy(input_np_padded, input_layout)


class BaseAvgPool2d:
    # TODO: Add more testing params..
    batch = tvm.testing.parameter(1)
    channel = tvm.testing.parameter(32)
    in_size = tvm.testing.parameter(10)
    out_size = tvm.testing.parameter(8)
    stride = tvm.testing.parameter(1)
    dilation = tvm.testing.parameter(1)
    padding = tvm.testing.parameter(0)
    kernel_size = tvm.testing.parameter(3)
    dtype = tvm.testing.parameter("float16")


class TestAvgPoolSlice2d(BaseAvgPool2d):
    @tvm.testing.fixture
    def expected_output_np(self,
        batch,
        in_size,
        channel,
        out_size,
        stride, #ignored at the moment
        kernel_size,
        dtype,
        dilation,
        padding,
        input_np):
        ref_np = tvm.topi.testing.poolnd_python(
            input_np,
            [kernel_size, kernel_size],
            [stride, stride],   # Only stride = 1 supported at the moment.
            [dilation, dilation], # I will be adding support for dilation but it's not there at the moment.
            [padding, padding], #padding_before; This should be set to 0 since the input is expected to be padded
            [padding, padding], #padding_after; This should be set to 0 as well.
            "avg", #pool_type
            True, #count_include_pad; The current implementation of sliced avgpool doesn't have an option to exclude pad from the count.
            False, #ceil_mode,
            layout="NHWC", #nhwc
        )
        return ref_np

    @tvm.testing.fixture
    def input_shape(self, batch, channel, out_size, kernel_size, padding, stride, dilation):
      dialated_kernel_size = dilation * (kernel_size - 1) + 1
      # out_size = (in_size - (dilation * (kernel_size - 1) + 1) + 2 * padding + 1))// stride + 1
      # Input size without crouton padding; 'ceil' is being ignored from calculation:
      input_size = (out_size - 1) * stride + dilation * (kernel_size - 1) + 1 - 2 * padding
      return [batch, input_size, input_size, channel]

    @tvm.testing.fixture
    def input_shape_padded(self, input_shape):
      # pad input width and height to a multiple of croutons
      padded_input_height = ((input_shape[1] + 7) // 8 ) * 8
      padded_input_width = ((input_shape[2] + 3) // 4) * 4
      return [input_shape[0], padded_input_height, padded_input_width, input_shape[3]]

    @tvm.testing.fixture
    def output_shape(self, batch, channel, out_size):
      return [batch, out_size, out_size, channel]

    @tvm.testing.fixture
    def input_np_padded(self, input_np, input_shape, input_shape_padded):
      pad_height = input_shape_padded[1] - input_shape[1]
      pad_width = input_shape_padded[2] - input_shape[2]
      input_padded = np.pad(input_np, ((0,0),(0, 6), (0, 2),(0,0)), 'constant')
      return input_padded


    @requires_hexagon_toolchain
    def test_avg_pool2d_slice(
        self,
        batch,
        in_size,
        channel,
        out_size,
        stride, #ignored at the moment
        kernel_size,
        dtype,
        dilation,
        padding,
        input_layout,
        output_layout,
        input_shape,
        output_shape,
        input_shape_padded,
        input_np,
        input_np_padded,
        transformed_input_np_padded,
        transformed_expected_output_np,
        hexagon_session
    ):

        target_hexagon = tvm.target.hexagon("v68")
        A = te.placeholder(input_shape_padded, name = "A", dtype = dtype)
        M = sl.avg_pool2d_compute(A, out_size, kernel_size, dtype)
        te_s = sl.avg_pool2d_schedule(M, A, fp16_layout_transform_2d)
        with tvm.transform.PassContext(opt_level=3, config={'tir.disable_assert':True}):
          func = tvm.build(te_s, [A, M], tvm.target.Target(target_hexagon, host=target_hexagon), name = 'avg_pool2d')
        #func.save('a.ll')
        #func.save('a.s')
        '''
        A_data = tvm.nd.empty(A.shape, A.dtype, hexagon_session.device) # It doens't work at the moment
        A_data.copyfrom(transformed_input_np_padded)
        M_data = tvm.nd.empty(M.shape, M.dtype, hexagon_session.device) # It doens't work at the moment
        mod = hexagon_session.load_module(func)
        mod(A_data, M_data)
        output_np = M_data.numpy()
        np.testing.assert_array_equal(output_np, transformed_expected_output_np)
        '''

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
