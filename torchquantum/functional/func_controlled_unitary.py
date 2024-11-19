"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch

from torchquantum.functional.gate_wrapper import gate_wrapper
from torchquantum.macro import C_DTYPE


def controlled_unitary(
    qdev,
    c_wires,
    t_wires,
    params,
):
    """Create a controlled operation from a given operation.

    Args:
        op: the operation
        c_wires: controlled wires, will only be a set such as 1, [2,3]
        t_wires: can be a list of list of wires, multiple sets
        [[1,2], [3,4]]
        params: the parameters of the unitary

    Returns:
        None.

    """
    if isinstance(params, np.ndarray):
        params = torch.from_numpy(params)

    c_wires = np.array(c_wires)
    t_wires = np.array(t_wires)
    # self.n_t_wires = op.n_wires
    # assert len(t_wires) == op.n_wires

    orig_u = params

    orig_u_n_wires = int(np.log2(orig_u.shape[-1]))

    wires = []

    if c_wires.ndim == 0:
        # only one control qubit
        # 1
        n_c_wires = 1
        wires.append(c_wires.item())
    elif c_wires.ndim == 1:
        # multiple control qubits
        # [1, 2]
        n_c_wires = c_wires.shape[0]
        wires.extend(list(c_wires))

    if t_wires.ndim == 0:
        # single qubit U on one set
        # 2
        n_t_wires = 1
        n_set_t_wires = 1
        wires.append(t_wires.item())
    elif t_wires.ndim == 1:
        # single qubit U on multiple sets
        # [1, 2, 3]
        # or multi qubit U on one set
        # [2, 3]
        n_t_wires = t_wires.shape[0]
        n_set_t_wires = n_t_wires // orig_u_n_wires
        wires.extend(list(t_wires.flatten()))

    elif t_wires.ndim == 2:
        # multi qubit unitary on multiple sets
        # [[2, 3], [4, 5]]
        n_t_wires = t_wires.flatten().shape[0]
        n_set_t_wires = n_t_wires // orig_u_n_wires
        wires.extend(list(t_wires.flatten()))

    n_wires = n_c_wires + n_t_wires

    # compute the new unitary, then permute
    unitary = torch.zeros(2**n_wires, 2**n_wires, dtype=C_DTYPE)
    for k in range(2**n_wires - 2**n_t_wires):
        unitary[k, k] = 1.0 + 0.0j

    # compute kronecker product of all the controlled target

    controlled_u = None
    for k in range(n_set_t_wires):
        if controlled_u is None:
            controlled_u = orig_u
        else:
            controlled_u = torch.kron(controlled_u, orig_u)

    d_controlled_u = controlled_u.shape[-1]
    unitary[-d_controlled_u:, -d_controlled_u:] = controlled_u

    # return cls(
    # has_params=True,
    # trainable=trainable,
    # init_params=unitary,
    # n_wires=n_wires,
    # wires=wires,
    # )

    name = "qubitunitaryfast"
    unitary = unitary.to(qdev.device)
    gate_wrapper(
        name=name,
        mat=unitary,
        method="bmm",
        q_device=qdev,
        wires=wires,
        n_wires=n_wires,
    )
