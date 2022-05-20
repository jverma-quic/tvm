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
np.set_printoptions(threshold=np.inf)
from tvm import te, topi
from tvm.tir.stmt_functor import post_order_visit

import tvm.testing
from tvm.topi import testing
from tvm.contrib.hexagon.build import HexagonLauncher
import tvm.topi.hexagon.slice_ops as sl
from .infrastructure import allocate_hexagon_array

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)


input_layout = tvm.testing.parameter(
    "nhwc-8h2w32c2w-2d",
)

def transform_numpy(arr_np, layout):
    if layout == "nhwc":
        return arr_np
    elif layout == "nhwc-8h2w32c2w-2d":
        N, H, W, C = arr_np.shape
        return arr_np.reshape([N, H // 8, 8, W // 4, 2, 2, C // 32, 32]).transpose(0, 1, 3, 6, 2, 4, 7, 5)
    elif layout == "n11c-1024c-2d":
        N, H, W, C = arr_np.shape
        assert (H == 1 and W == 1), "The size of H and W must be 1!"
        return arr_np.reshape([N, C//1024, 1024]).transpose(0, 1, 2)
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

def fp16_n11c_1024c_2d(n, h, w, c):
    return [
        n,
        h,
        w,
        c // 1024,
        te.AXIS_SEPARATOR,
        c % 1024
    ]

def fp16_nhwc_8h2w32c2w_2d(n, h, w, c):
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
    return np.random.random(input_shape).astype(dtype)

@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, out_layout):
    return transform_numpy(expected_output_np, out_layout)

@tvm.testing.fixture
def transformed_input_np_padded(input_np_padded, input_layout):
    return transform_numpy(input_np_padded, input_layout)


class TestAvgPoolSlice2d:
    # NOTE: input_layout is always assumed to be "nhwc-8h2w32c2w-2d"
    output_shape, kernel, stride, dilation, padding, ceil_mode, count_include_pad, out_layout, dtype = tvm.testing.parameters(
        ([1, 8, 8, 32], [3, 3], [1, 1], [1, 1], [0, 0, 0, 0], False, True, "nhwc-8h2w32c2w-2d", "float16"),
    )
    '''
        ([1, 16, 16, 32], [3, 3], [1, 1], [1, 1], [0, 0, 0, 0], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        ([1, 8, 8, 32], [8, 8], [1, 1], [1, 1], [0, 0, 0, 0], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        ([1, 8, 8, 32], [1, 1], [1, 1], [1, 1], [0, 0, 0, 0], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        # Test non-one stride and dilation
        ([1, 8, 8, 32], [3, 3], [2, 3], [1, 1], [0, 0, 0, 0], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        ([1, 8, 8, 32], [3, 3], [2, 2], [2, 2], [0, 0, 0, 0], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        ([1, 8, 8, 32], [3, 3], [2, 2], [2, 3], [0, 0, 0, 0], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        # Test non-zero padding
        ([1, 8, 8, 32], [3, 3], [1, 1], [1, 1], [1, 1, 1, 1], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        ([1, 8, 8, 32], [3, 3], [1, 1], [1, 1], [1, 2, 3, 4], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        ([1, 8, 8, 32], [3, 3], [1, 1], [1, 1], [1, 2, 3, 4], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        ([1, 8, 8, 32], [3, 3], [3, 2], [2, 3], [1, 2, 3, 4], False, True, "nhwc-8h2w32c2w-2d", "float16"),
        # Test n11c-1024c-2d layout which will require input and output to have different layout
        # Doesn't work right now - Produces incorrect output
        #([1, 1, 1, 2048], [8, 8], [1, 1], [1, 1], [0, 0, 0, 0], False, True, "n11c-1024c-2d", "float16"),
        #([1, 1, 1, 2048], [6, 6], [1, 1], [1, 1], [0, 0, 0, 0], False, True, "n11c-1024c-2d", "float16"),
    )
    '''
    @tvm.testing.fixture
    def expected_output_np(self,
        input_np,
        kernel,
        stride,
        dilation,
        padding,
        ceil_mode,
        count_include_pad,
        ):
        pad_before = padding[:2]
        pad_after = padding[2:]
        ref_np = tvm.topi.testing.poolnd_python(
            input_np,
            kernel,
            stride,
            dilation,
            pad_before, #padding_before; This should be set to 0 since the input is expected to be padded
            pad_after, #padding_after; This should be set to 0 as well.
            "avg", #pool_type
            count_include_pad, #count_include_pad; The current implementation of sliced avgpool doesn't have an option to exclude pad from the count.
            False, #ceil_mode,
            layout="NHWC",
        )
        return ref_np

    @tvm.testing.fixture
    def input_shape(self, output_shape, kernel, padding, stride, dilation, out_layout):
      # Input shape without crouton padding; 'ceil' is being ignored from calculation:
      o_b, o_h, o_w, o_c = output_shape
      d_h, d_w = dilation
      s_h, s_w = stride
      k_h, k_w = kernel
      pad_before_h, pad_before_w = padding[:2]
      pad_after_h, pad_after_w = padding[2:]

      if out_layout == "n11c-1024c-2d":
        assert pad_before_w == 0 and pad_after_w == 0 and pad_before_h == 0 and pad_after_h == 0, ("Padding must be zero for n11c-1024c-2d layout!!")
        assert o_h == 1 and o_w == 1, ("Output height and width must be 1!")

      in_h = (o_h - 1) * s_h + d_h * (k_h - 1) + 1 - pad_before_h - pad_after_h
      in_w = (o_w - 1) * s_w + d_w * (k_w - 1) + 1 - pad_before_w - pad_after_w

      return [o_b, in_h, in_w, o_c]

    @tvm.testing.fixture
    def input_shape_padded(self, input_shape, padding, out_layout):
      # Input shape with regular and crouton padding.
      # Input width and height are padded to a multiple of croutons.
      # NOTE: Input layout is always assumed to be nhwc-8h2w32c2w-2d. Only the output layout can be
      # different.
      pad_before_h, pad_before_w = padding[:2]
      pad_after_h, pad_after_w = padding[2:]
      padded_input_height = ((input_shape[1] + pad_before_h + pad_after_h + 7) // 8 ) * 8
      padded_input_width = ((input_shape[2] + pad_before_w + pad_after_w + 3) // 4) * 4
      return [input_shape[0], padded_input_height, padded_input_width, input_shape[3]]

    @tvm.testing.fixture
    def input_np_padded(self, input_np, input_shape, input_shape_padded, padding):
      pad_before_h, pad_before_w = padding[:2]
      pad_after_h = input_shape_padded[1] - input_shape[1] - pad_before_h # pad_after for height with crouton padding
      pad_after_w = input_shape_padded[2] - input_shape[2] - pad_before_w # pad_after for width with crouton padding
      input_padded = np.pad(input_np, ((0,0),(pad_before_h, pad_after_h), (pad_before_w, pad_after_w),(0,0)), 'constant')
      return input_padded

    @tvm.testing.requires_hexagon
    def test_avg_pool2d_slice(
        self,
        stride,
        kernel,
        dtype,
        dilation,
        padding,
        input_layout,
        out_layout,
        output_shape,
        input_shape,
        input_shape_padded,
        input_np,
        input_np_padded,
        transformed_input_np_padded,
        transformed_expected_output_np,
        expected_output_np,
        hexagon_session
    ):

        target_hexagon = tvm.target.hexagon("v68")
        A = te.placeholder(input_shape_padded, name = "A", dtype = dtype)

        M = sl.avg_pool2d_compute(A, output_shape, kernel, stride, dilation, dtype)

        # TIR Schedule
        #tir_schedule = sl.avg_pool2d_STIR_schedule(M, A, "nhwc-8h2w32c2w-1d", "nhwc-8h2w32c2w-1d")
        #sch = tir_schedule.mod

        if out_layout == "nhwc-8h2w32c2w-2d":
          te_s = sl.avg_pool2d_schedule1(M, A, fp16_nhwc_8h2w32c2w_2d, fp16_nhwc_8h2w32c2w_2d)
          sch = te_s
          input_axis_separator = [4]
          output_axis_separator = [4]
        elif out_layout == "n11c-1024c-2d":
          te_s = sl.avg_pool2d_schedule2(M, A, fp16_nhwc_8h2w32c2w_2d, fp16_n11c_1024c_2d)
          sch = te_s
          input_axis_separator = [4]
          output_axis_separator = [2]
        else:
          assert(False), "Unsupported output layout"

        with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()], config={'tir.disable_assert':True}):
          func = tvm.build(sch, [A, M], tvm.target.Target(target_hexagon, host=target_hexagon), name = 'avg_pool2d')

        def flatten_2d(data, axis_separators):
          tensor_shape = data.shape
          boundaries = [0, *axis_separators, len(tensor_shape)]
          physical_shape = [
              np.prod(tensor_shape[dim_i:dim_f])
              for dim_i, dim_f in zip(boundaries[:-1], boundaries[1:])
          ]
          return physical_shape

        flattened_output_shape = flatten_2d(transformed_expected_output_np, output_axis_separator)
        flattened_input_shape = flatten_2d(transformed_input_np_padded, input_axis_separator)
        reshaped_transformed_input_np_padded = np.reshape(transformed_input_np_padded, flattened_input_shape)
        input_arr = tvm.nd.empty(flattened_input_shape, A.dtype, hexagon_session.device, 'global.vtcm')
        input_arr.copyfrom(reshaped_transformed_input_np_padded)
        output_arr = tvm.nd.empty(flattened_output_shape, A.dtype, hexagon_session.device, 'global.vtcm')
        mod = hexagon_session.load_module(func)
        mod(input_arr, output_arr)
        b, h, w, c = output_shape
        if out_layout == "nhwc-8h2w32c2w-2d":
          output_np = output_arr.numpy().reshape([b, h // 8, w // 4, c // 32, 8, 2, 32, 2])
        elif out_layout == "n11c-1024c-2d":
          output_np = output_arr.numpy().reshape([b, c//1024, 1024])
        else:
          assert(False), "Unsupported output layout"

        np.testing.assert_allclose(output_np, transformed_expected_output_np, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
