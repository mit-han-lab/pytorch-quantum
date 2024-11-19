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

# test the controlled unitary function


import torchquantum as tq
import torch
from test.utils import check_all_close


def test_GeneralEncoder():

    parameterised_funclist = [
        {"input_idx": [0], "func": "crx", "wires": [1, 0]},
        {"input_idx": [1, 2, 3], "func": "u3", "wires": [1]},
        {"input_idx": [4], "func": "ry", "wires": [0]},
        {"input_idx": [5], "func": "ry", "wires": [1]},
    ]

    semiparam_funclist = [
        {"params": [0.2], "func": "crx", "wires": [1, 0]},
        {"params": [0.3, 0.4, 0.5], "func": "u3", "wires": [1]},
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
    ]

    expected_states = torch.complex(
        torch.Tensor(
            [[0.8423, 0.4474, 0.2605, 0.1384], [0.7649, 0.5103, 0.3234, 0.2157]]
        ),
        torch.Tensor(
            [[-0.0191, 0.0522, -0.0059, 0.0162], [-0.0233, 0.0483, -0.0099, 0.0204]]
        ),
    )

    parameterised_enc = tq.GeneralEncoder(parameterised_funclist)
    semiparam_enc = tq.GeneralEncoder(semiparam_funclist)

    param_vec = torch.Tensor(
        [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.2, 0.3, 0.4, 0.5, 0.8, 0.9]]
    )
    semiparam_vec = torch.Tensor([[0.6, 0.7], [0.8, 0.9]])

    qd = tq.QuantumDevice(n_wires=2)

    qd.reset_states(bsz=2)
    parameterised_enc(qd, param_vec)
    state1 = qd.get_states_1d()

    qd.reset_states(bsz=2)
    semiparam_enc(qd, semiparam_vec)
    state2 = qd.get_states_1d()

    check_all_close(state1, state2)
    check_all_close(state1, expected_states)


if __name__ == "__main__":
    test_GeneralEncoder()
