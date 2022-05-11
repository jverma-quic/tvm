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

def avg_pool2d_compute(A, out_size, kernel_size, dtype):
    rh = te.reduce_axis((0, kernel_size), name='rh')
    rw = te.reduce_axis((0, kernel_size), name='rw')
    Area = kernel_size * kernel_size
    batch = A.shape[0]
    channel = A.shape[3]
    B = te.compute((batch, out_size, out_size, channel),
        lambda b, h, w, c: te.sum(A[b, h + rh, w + rw, c].astype(dtype), axis=[rh, rw]), name='B')
    M = te.compute((batch, out_size, out_size, channel),
        lambda b, h, w, c: (B[b, h, w, c] / Area).astype(dtype), name='M')
    return M
    #return te.create_prim_func([A, M])


def avg_pool2d_schedule(outs, ins, layout):
    A = ins
    M = outs
    s = te.create_schedule([M.op])
    B = s[M].op.input_tensors[0]

    s[A].transform_layout(layout)
    M_axis = s[M].transform_layout(layout)

    # Sum
    bn, bh, bw, bc = s[B].op.axis
    rx, ry = s[B].op.reduce_axis
    bwo, bwi = s[B].split(bw, factor=4)
    bwio, bwii = s[B].split(bwi, factor=2)
    bco, bci = s[B].split(bc, factor = 32)
    s[B].reorder(bn, bco, bh, bwo, bwio, ry, rx, bci, bwii)
    b_inner = s[B].fuse(bci, bwii)

    #s[B].vectorize(b_inner) # This doesn't work!

    s[B].compute_at(s[M], M_axis[5])
    m_inner = s[M].fuse(M_axis[7], M_axis[6])
    s[M].vectorize(m_inner)
    return s

@T.prim_func
def avg_pool2d_slice(
    aT: T.handle,
    bT: T.handle,
    N: T.int32,
    AH: T.int32,
    AW: T.int32,
    C: T.int32,
    BH: T.int32,
    BW: T.int32,
    K: T.int32
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    AT = T.match_buffer(aT, (N, AH, AW, C), dtype="float16")
    BT = T.match_buffer(bT, (N, BH, BW, C), dtype="float16")
    for n, bh, bw, c in T.grid(N, BH, BW, C):
        with T.block("init"):
            vn, vbh, vbw, vc = T.axis.remap("SSSS", [n, bh, bw, c])
            BT[vn, vbh, vbw, vc] = T.float16(0)
        for kh, kw in T.grid(K, K):
            with T.block("update"):
                vn, vbh, vbw, vc, vkh, vkw = T.axis.remap("SSSSRR", [n, bh, bw, c, kh, kw])
                BT[vn, vbh, vbw, vc] = BT[vn, vbh, vbw, vc] + AT[vn, vbh + vkh, vbw + vkw, vc]



# TODO:Add Schedulable Tensor IR
def avg_pool2d_STIR(outs, ins):
    return
