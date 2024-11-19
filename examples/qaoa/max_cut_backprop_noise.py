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

import torch
import torchquantum as tq

import random
import numpy as np

from torchquantum.functional import mat_dict

from torchquantum.measurement import expval_joint_analytical_density

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class MAXCUT(tq.QuantumModule):
    """computes the optimal cut for a given graph.
    outputs: the most probable bitstring decides the set {0 or 1} each
    node belongs to.
    """

    def __init__(self, n_wires, input_graph, n_layers):
        super().__init__()

        self.n_wires = n_wires

        self.input_graph = input_graph  # list of edges
        self.n_layers = n_layers

        self.betas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))
        self.gammas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))

    def mixer(self, qdev, beta):
        """
        Apply the single rotation and entangling layer of the QAOA ansatz.
        mixer = exp(-i * beta * sigma_x)
        """
        for wire in range(self.n_wires):
            qdev.rx(
                wires=wire,
                params=beta.unsqueeze(0),
            )  # type: ignore

    def entangler(self, qdev, gamma):
        """
        Apply the single rotation and entangling layer of the QAOA ansatz.
        entangler = exp(-i * gamma * (1 - sigma_z * sigma_z)/2)
        """
        for edge in self.input_graph:
            qdev.cx(
                [edge[0], edge[1]],
            )  # type: ignore
            qdev.rz(
                wires=edge[1],
                params=gamma.unsqueeze(0),
            )  # type: ignore
            qdev.cx(
                [edge[0], edge[1]],
            )  # type: ignore

    def edge_to_PauliString(self, edge):
        # construct pauli string
        pauli_string = ""
        for wire in range(self.n_wires):
            if wire in edge:
                pauli_string += "Z"
            else:
                pauli_string += "I"
        return pauli_string

    def circuit(self, qdev):
        """
        execute the quantum circuit
        """
        # print(self.betas, self.gammas)
        for wire in range(self.n_wires):
            qdev.h(
                wires=wire,
            )  # type: ignore

        for i in range(self.n_layers):
            self.mixer(qdev, self.betas[i])
            self.entangler(qdev, self.gammas[i])

    def forward(self, measure_all=False):
        """
        Apply the QAOA ansatz and only measure the edge qubit on z-basis.
        Args:
            if edge is None
        """
        qdev = tq.NoiseDevice(
            n_wires=self.n_wires, device=self.betas.device, record_op=False,
            noise_model=tq.NoiseModel(kraus_dict={"Bitflip": 0.12, "Phaseflip": 0.12})
        )

        self.circuit(qdev)

        # turn on the record_op above to print the circuit
        # print(op_history2qiskit(self.n_wires, qdev.op_history))

        # print(tq.measure(qdev, n_shots=1024))
        # compute the expectation value
        # print(qdev.get_states_1d())
        if measure_all is False:
            expVal = 0
            for edge in self.input_graph:
                pauli_string = self.edge_to_PauliString(edge)
                expv = expval_joint_analytical_density(qdev, observable=pauli_string)
                expVal += 0.5 * expv
                # print(pauli_string, expv)
            # print(expVal)
            return expVal
        else:
            return tq.measure_density(qdev, n_shots=1024, draw_id=0)


def backprop_optimize(model, n_steps=100, lr=0.1):
    """
    Optimize the QAOA ansatz over the parameters gamma and beta
    Args:
        betas (np.array): A list of beta parameters.
        gammas (np.array): A list of gamma parameters.
        n_steps (int): The number of steps to optimize, defaults to 10.
        lr (float): The learning rate, defaults to 0.1.
    """
    # measure all edges in the input_graph
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(
        "The initial parameters are betas = {} and gammas = {}".format(
            *model.parameters()
        )
    )
    # optimize the parameters and return the optimal values
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        if step % 2 == 0:
            print("Step: {}, Cost Objective: {}".format(step, loss.item()))

    print(
        "The optimal parameters are betas = {} and gammas = {}".format(
            *model.parameters()
        )
    )
    return model(measure_all=True)


def main():
    # create a input_graph
    input_graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
    n_wires = 4
    n_layers = 3
    model = MAXCUT(n_wires=n_wires, input_graph=input_graph, n_layers=n_layers)
    # model.to("cuda")
    # model.to(torch.device("cuda"))
    # circ = tq2qiskit(tq.QuantumDevice(n_wires=4), model)
    # print(circ)
    # print("The circuit is", circ.draw(output="mpl"))
    # circ.draw(output="mpl")
    # use backprop
    backprop_optimize(model, n_steps=300, lr=0.01)
    # use parameter shift rule
    # param_shift_optimize(model, n_steps=500, step_size=100000)


"""
Notes:
1. input_graph = [(0, 1), (3, 0), (1, 2), (2, 3)], mixer 1st & entangler 2nd, n_layers >= 2, answer is correct.

"""

if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()

    main()
