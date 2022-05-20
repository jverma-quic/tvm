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

def fp16_n11c_1024c_1d(n, h, w, c):
    return [
        n,
        h,
        w,
        c // 1024,
        c % 1024
    ]

def fp16_nhwc_8h2w32c2w_1d(n, h, w, c):
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

def get_transform_layout_fn(layout):
    if layout == "nhwc-8h2w32c2w-2d":
        return fp16_nhwc_8h2w32c2w_2d
    if layout == "nhwc-8h2w32c2w-1d":
        return fp16_nhwc_8h2w32c2w_1d
    elif layout == "n11c-1024c-2d":
        return fp16_n11c_1024c_2d
    elif layout == "n11c-1024c-1d":
        return fp16_n11c_1024c_1d

def get_axis_separators(layout):
    if layout == "nhwc-8h2w32c2w-2d":
        return [4]
    elif layout == "nhwc-8h2w32c2w-1d":
        return [4]
    elif layout == "n11c-1024c-2d":
        return [2]
    elif layout == "n11c-1024c-1d":
        return [2]


def apply_transform(s, block, block_index, buffer_type, layout):
    """Apply transform layout on a buffer

    Parameters
    ----------
    s: Schedule
    """
    transform_fn = get_transform_layout_fn(layout)
    axis_separators = get_axis_separators(layout)
    s.transform_layout(block, block_index, buffer_type, transform_fn)
    if axis_separators:
        s.set_axis_separator(block, block_index, buffer_type, axis_separators)
