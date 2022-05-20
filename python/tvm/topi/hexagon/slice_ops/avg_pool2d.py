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

from tvm.ir.module import IRModule
from tvm import te
from tvm import tir
from tvm.script import tir as T
from ..utils import apply_transform


def avg_pool2d_compute(A, out_shape, kernel, stride, dilation, dtype):
    kh, kw = kernel
    rh = te.reduce_axis((0, kh), name='rh')
    rw = te.reduce_axis((0, kw), name='rw')
    Area = float(1)/(kh * kw)

    ob, oh, ow, oc = out_shape
    sh, sw = stride
    dh, dw = dilation
    Sum = te.compute((ob, oh, ow, oc),
        lambda b, h, w, c: te.sum(A[b, h * sh + dh * rh, w * sw + dw * rw, c].astype("float32"), axis=[rh, rw]), name='sum')
    Avg = te.compute((ob, oh, ow, oc),
        lambda b, h, w, c: (Sum[b, h, w, c] * Area).astype(dtype), name='avg')
    return Avg


# TIR based schedule
def avg_pool2d_STIR_schedule(outs, ins, output_layout:str, input_layout:str):
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)
    Sum = s.get_block("sum")
    Avg = s.get_block("avg")

    apply_transform(s, Sum, 0, "read", input_layout)
    apply_transform(s, Avg, 0, "write", output_layout)

    bn, bh, bw, bc, rx, ry = s.get_loops(Sum)
    ho, hi = s.split(bh, [None, 8])
    wo, wi = s.split(bw, [None, 4])
    wio, wii = s.split(wi, [None, 2]) # Doesn't seem to be doing anything
    co, ci = s.split(bc, [None, 32])
    s.reorder(bn, ho, wo, co, hi, wio, rx, ry, ci, wii) # --- DOESN'T do anything
    ci_wii = s.fuse(ci, wii) # --- DOESN'T do anything
    #s.vectorize(ci_wii) # --- DOESN'T WORK -- errors out

    n, h, w, c = s.get_loops(Avg)
    ho, hi = s.split(h, [None, 8])
    wo, wi = s.split(w, [None, 4])
    wio, wii = s.split(wi, [None, 2])
    co, ci = s.split(c, [None, 32])
    s.reorder(n, ho, wo, co, hi, wio, ci, wii)
    ci_wii = s.fuse(ci, wii)
    s.vectorize(ci_wii)

    s.compute_at(Sum, hi)
    return s


# Schedule for "nhwc-8h2w32c2w-2d" input and output layout
def avg_pool2d_schedule1(outs, ins, input_layout, output_layout):
    A = ins
    M = outs
    s = te.create_schedule([M.op])
    B = s[M].op.input_tensors[0]

    s[A].transform_layout(input_layout)
    #B_axis = s[B].transform_layout(input_layout) # Results into runtime failure
    M_axis = s[M].transform_layout(output_layout)

    # B
    bn, bh, bw, bc = s[B].op.axis
    rx, ry = s[B].op.reduce_axis
    bwo, bwi = s[B].split(bw, factor=4)
    bwio, bwii = s[B].split(bwi, factor=2)
    bco, bci = s[B].split(bc, factor = 32)
    s[B].reorder(bn, bco, bh, bwo, bwio, ry, rx, bci, bwii)
    b_inner = s[B].fuse(bci, bwii) # We want to treat b_inner as the fastest changing axis and vectorize it.

    # NOTE: My understanding is that 'vectorize' would require crouton specific handling during codegen.
    #s[B].vectorize(b_inner) # This doesn't work with or without applying transform layout on 'B'.

    s[B].compute_at(s[M], M_axis[5])

    m_inner = s[M].fuse(M_axis[7], M_axis[6])
    s[M].vectorize(m_inner)
    return s


#Schedule for "n11c-1024c-2d" output layout
def avg_pool2d_schedule2(outs, ins, input_layout, output_layout):
    A = ins
    M = outs
    s = te.create_schedule([M.op])
    B = s[M].op.input_tensors[0]

    s[A].transform_layout(input_layout)
    M_axis = s[M].transform_layout(output_layout)

    # Schedule Sum
    #B_axis = s[B].transform_layout(output_layout)
    B_axis = s[B].op.axis
    rx, ry = s[B].op.reduce_axis
    bco, bci = s[B].split(B_axis[3], factor = 64)
    s[B].reorder(bco, rx, ry, bci)
    s[B].unroll(s[B].fuse(rx, ry))
    mco, mci = s[M].split(M_axis[4], factor = 64)

    #s[B].vectorize(bci)

    s[B].compute_at(s[M], mco)

    s[M].vectorize(mci)
    return s

