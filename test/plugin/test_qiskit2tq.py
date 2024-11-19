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
import pytest
import torch
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchquantum as tq
from torchquantum.plugin import qiskit2tq

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class TQModel(tq.QuantumModule):
    def __init__(self, init_params=None):
        super().__init__()
        self.n_wires = 2
        self.rx = tq.RX(has_params=True, trainable=True, init_params=[init_params[0]])
        self.u3_0 = tq.U3(has_params=True, trainable=True, init_params=init_params[1:4])
        self.u3_1 = tq.U3(
            has_params=True,
            trainable=True,
            init_params=torch.tensor(
                [
                    init_params[4] + init_params[2],
                    init_params[5] * init_params[3],
                    init_params[6] * init_params[1],
                ]
            ),
        )
        self.cu3_0 = tq.CU3(
            has_params=True,
            trainable=True,
            init_params=torch.tensor(
                [
                    torch.sin(init_params[7]),
                    torch.abs(torch.sin(init_params[8])),
                    torch.abs(torch.sin(init_params[9]))
                    * torch.exp(init_params[2] + init_params[3]),
                ]
            ),
        )

    def forward(self, q_device: tq.QuantumDevice):
        q_device.reset_states(1)
        self.rx(q_device, wires=0)
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.cu3_0(q_device, wires=[0, 1])


def get_qiskit_ansatz():
    ansatz = QuantumCircuit(2)
    ansatz_param = Parameter("Θ")  # parameter
    ansatz.rx(ansatz_param, 0)
    ansatz_param_vector = ParameterVector("φ", 9)  # parameter vector
    ansatz.u(ansatz_param_vector[0], ansatz_param_vector[1], ansatz_param_vector[2], 0)
    ansatz.u(
        ansatz_param_vector[3] + ansatz_param_vector[1],  # parameter expression
        ansatz_param_vector[4] * ansatz_param_vector[2],
        ansatz_param_vector[5] / ansatz_param_vector[0],
        1,
    )
    ansatz.cu(
        np.sin(ansatz_param_vector[6]),  # numpy functions
        np.abs(np.sin(ansatz_param_vector[7])),  # nested numpy functions
        # complex expression
        np.abs(np.sin(ansatz_param_vector[8]))
        * np.exp(ansatz_param_vector[1] + ansatz_param_vector[2]),
        0.0,
        0,
        1,
    )
    return ansatz


def train_step(target_state, device, model, optimizer):
    model(device)
    result_state = device.get_states_1d()[0]

    # compute the state infidelity
    loss = 1 - torch.dot(result_state, target_state).abs() ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    infidelity = loss.item()
    target_state_vector = target_state.detach().cpu().numpy()
    result_state_vector = result_state.detach().cpu().numpy()
    print(
        f"infidelity (loss): {infidelity}, \n target state : "
        f"{target_state_vector}, \n "
        f"result state : {result_state_vector}\n"
    )
    return infidelity, target_state_vector, result_state_vector


def train(init_params, backend):
    device = torch.device("cpu")

    if backend == "qiskit":
        ansatz = get_qiskit_ansatz()
        model = qiskit2tq(ansatz, initial_parameters=init_params).to(device)
    elif backend == "torchquantum":
        model = TQModel(init_params).to(device)

    print(f"{backend} model:", model)

    n_epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    q_device = tq.QuantumDevice(n_wires=2)
    target_state = torch.tensor([0, 1, 0, 0], dtype=torch.complex64)

    result_list = []
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        result_list.append(train_step(target_state, q_device, model, optimizer))
        scheduler.step()

    return result_list


@pytest.mark.parametrize(
    "init_params",
    [
        torch.nn.init.uniform_(torch.ones(10), -np.pi, np.pi),
        torch.nn.init.uniform_(torch.ones(10), -np.pi, np.pi),
        torch.nn.init.uniform_(torch.ones(10), -np.pi, np.pi),
    ],
)
def test_qiskit2tq(init_params):
    qiskit_result = train(init_params, "qiskit")
    tq_result = train(init_params, "torchquantum")
    for qi_tensor, tq_tensor in zip(qiskit_result, tq_result):
        torch.testing.assert_close(qi_tensor[0], tq_tensor[0])
        torch.testing.assert_close(qi_tensor[1], tq_tensor[1])
        torch.testing.assert_close(qi_tensor[2], tq_tensor[2])


if __name__ == "__main__":
    test_qiskit2tq(torch.nn.init.uniform_(torch.ones(10), -np.pi, np.pi))
