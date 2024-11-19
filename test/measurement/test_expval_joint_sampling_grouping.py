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

import random

import numpy as np

import torchquantum as tq
from torchquantum.measurement import (
    expval_joint_analytical,
    expval_joint_sampling_grouping,
)


def test_expval_joint_sampling_grouping():
    n_obs = 20
    n_wires = 4
    obs_all = []
    for _ in range(n_obs):
        obs = random.choices(["X", "Y", "Z", "I"], k=n_wires)
        obs_all.append("".join(obs))
    obs_all = list(set(obs_all))

    random_layer = tq.RandomLayer(n_ops=100, wires=list(range(n_wires)))
    qdev = tq.QuantumDevice(n_wires=n_wires, bsz=1, record_op=True)
    random_layer(qdev)

    expval_ana = {}
    for obs in obs_all:
        expval_ana[obs] = expval_joint_analytical(qdev, observable=obs)[0].item()

    expval_sam = expval_joint_sampling_grouping(
        qdev, observables=obs_all, n_shots_per_group=1000000
    )
    for obs in obs_all:
        # assert
        assert np.isclose(expval_ana[obs], expval_sam[obs][0].item(), atol=0.015)
        print(obs, expval_ana[obs], expval_sam[obs][0].item())


if __name__ == "__main__":
    test_expval_joint_sampling_grouping()
