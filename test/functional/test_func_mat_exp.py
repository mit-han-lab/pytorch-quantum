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

import torchquantum as tq


def test_func_mat_exp():
    qdev = tq.QuantumDevice(n_wires=3)
    qdev.reset_states(bsz=1)

    qdev.matrix_exp(wires=[0], params=torch.tensor([[1.0, 2.0], [3.0, 4.0 + 1.0j]]))

    assert np.allclose(
        qdev.get_states_1d().cpu().detach().numpy(),
        np.array(
            [
                [
                    44.2796 + 23.9129j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    85.5304 + 68.1896j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                ]
            ]
        ),
    )

    qdev = tq.QuantumDevice(n_wires=3)
    qdev.reset_states(bsz=2)

    qdev.matrix_exp(
        wires=[0, 2],
        params=torch.tensor(
            [
                [1.0, 2.0, 2, 1],
                [3.0, 4.0 + 1.0j, 2, 1],
                [1.0, 2.0, 2, 1],
                [3.0, 4.0 + 1.0j, 2, 1],
            ]
        ),
    )  # type: ignore
    # print(qdev.get_states_1d().cpu().detach().numpy())

    assert np.allclose(
        qdev.get_states_1d().cpu().detach().numpy(),
        np.array(
            [
                [
                    483.20386 + 254.27155j,
                    747.27014 + 521.95013j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    482.2038 + 254.27151j,
                    747.27014 + 521.95013j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    483.20386 + 254.27155j,
                    747.27014 + 521.95013j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    482.2038 + 254.27151j,
                    747.27014 + 521.95013j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
            ]
        ),
    )


if __name__ == "__main__":
    import pdb

    pdb.set_trace()

    test_func_mat_exp()
