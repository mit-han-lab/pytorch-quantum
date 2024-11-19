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

from torchquantum.measurement import expval_joint_analytical_density

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from torchquantum.plugin import QiskitProcessor, op_history2qiskit


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
        self.n_edges = len(input_graph)

        self.betas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))
        self.gammas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))

        self.reset_shift_param()

    def mixer(self, qdev, beta, layer_id):
        """
        Apply the single rotation and entangling layer of the QAOA ansatz.
        mixer = exp(-i * beta * sigma_x)
        """

        for wire in range(self.n_wires):
            if (
                    self.shift_param_name == "beta"
                    and self.shift_wire == wire
                    and layer_id == self.shift_layer
            ):
                degree = self.shift_degree
            else:
                degree = 0
            qdev.rx(
                wires=wire,
                params=(beta.unsqueeze(0) + degree),
            )  # type: ignore

    def entangler(self, qdev, gamma, layer_id):
        """
        Apply the single rotation and entangling layer of the QAOA ansatz.
        entangler = exp(-i * gamma * (1 - sigma_z * sigma_z)/2)
        """
        for edge_id, edge in enumerate(self.input_graph):
            if (
                    self.shift_param_name == "gamma"
                    and edge_id == self.shift_edge_id
                    and layer_id == self.shift_layer
            ):
                degree = self.shift_degree
            else:
                degree = 0
            qdev.cx(
                [edge[0], edge[1]],
            )  # type: ignore
            qdev.rz(
                wires=edge[1],
                params=(gamma.unsqueeze(0) + degree),
            )  # type: ignore
            qdev.cx(
                [edge[0], edge[1]],
            )  # type: ignore

    def set_shift_param(self, layer, wire, param_name, degree, edge_id):
        """
        set the shift parameter for the parameter shift rule
        """
        self.shift_layer = layer
        self.shift_wire = wire
        self.shift_param_name = param_name
        self.shift_degree = degree
        self.shift_edge_id = edge_id

    def reset_shift_param(self):
        """
        reset the shift parameter
        """
        self.shift_layer = None
        self.shift_wire = None
        self.shift_param_name = None
        self.shift_degree = None
        self.shift_edge_id = None

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
            self.mixer(qdev, self.betas[i], i)
            self.entangler(qdev, self.gammas[i], i)

    def forward(self, use_qiskit, measure_all=False):
        """
        Apply the QAOA ansatz and only measure the edge qubit on z-basis.
        Args:
            if edge is None
        """
        qdev = tq.NoiseDevice(n_wires=self.n_wires, device=self.betas.device,
                              noise_model=tq.NoiseModel(kraus_dict={"Bitflip": 0.12, "Phaseflip": 0.12}))


        # print(tq.measure(qdev, n_shots=1024))
        # compute the expectation value
        # print(qdev.get_states_1d())

        if not use_qiskit:
            self.circuit(qdev)
            expVal = 0
            for edge in self.input_graph:
                pauli_string = self.edge_to_PauliString(edge)
                expv = expval_joint_analytical_density(qdev, observable=pauli_string)
                expVal += 0.5 * expv
        else:
            # use qiskit to compute the expectation value
            expVal = 0
            for edge in self.input_graph:
                pauli_string = self.edge_to_PauliString(edge)

                with torch.no_grad():
                    self.circuit(qdev)
                circ = op_history2qiskit(qdev.n_wires, qdev.op_history)

                expv = self.qiskit_processor.process_circs_get_joint_expval([circ], pauli_string)[0]
                expVal += 0.5 * expv
            expVal = torch.Tensor([expVal])
        return expVal


def shift_and_run(model, use_qiskit):
    # flatten the parameters into 1D array

    grad_betas = []
    grad_gammas = []
    n_layers = model.n_layers
    n_wires = model.n_wires
    n_edges = model.n_edges

    for i in range(n_layers):
        grad_gamma = 0
        for k in range(n_edges):
            model.set_shift_param(i, None, "gamma", np.pi * 0.5, k)
            out1 = model(use_qiskit)
            model.reset_shift_param()

            model.set_shift_param(i, None, "gamma", -np.pi * 0.5, k)
            out2 = model(use_qiskit)
            model.reset_shift_param()

            grad_gamma += 0.5 * (out1 - out2).squeeze().item()
        grad_gammas.append(grad_gamma)

        grad_beta = 0
        for j in range(n_wires):
            model.set_shift_param(i, j, "beta", np.pi * 0.5, None)
            out1 = model(use_qiskit)
            model.reset_shift_param()

            model.set_shift_param(i, j, "beta", -np.pi * 0.5, None)
            out2 = model(use_qiskit)
            model.reset_shift_param()

            grad_beta += 0.5 * (out1 - out2).squeeze().item()
        grad_betas.append(grad_beta)

    return model(use_qiskit), [grad_betas, grad_gammas]


def param_shift_optimize(model, n_steps=10, step_size=0.1, use_qiskit=False):
    """finds the optimal cut where parameter shift rule is used to compute the gradient"""
    # optimize the parameters and return the optimal values
    # print(
    # "The initial parameters are betas = {} and gammas = {}".format(
    # *model.parameters()
    # )
    # )
    n_layers = model.n_layers
    for step in range(n_steps):
        with torch.no_grad():
            loss, grad_list = shift_and_run(model, use_qiskit=use_qiskit)
        # param_list = list(model.parameters())
        # print(
        # "The initial parameters are betas = {} and gammas = {}".format(
        # *model.parameters()
        # )
        # )
        # param_list = torch.cat([param.flatten() for param in param_list])

        # print("The shape of the params", len(param_list), param_list[0].shape, param_list)
        # print("")
        # print("The shape of the grad_list = {}, 0th elem shape = {}, grad_list = {}".format(len(grad_list), grad_list[0].shape, grad_list))
        # print(grad_list, loss, model.betas, model.gammas)
        print(loss)
        with torch.no_grad():
            for i in range(n_layers):
                model.betas[i].copy_(model.betas[i] - step_size * grad_list[0][i])
                model.gammas[i].copy_(model.gammas[i] - step_size * grad_list[1][i])

            # for param, grad in zip(param_list, grad_list):
            # modify the parameters and ensure that there are no multiple views
            # param.copy_(param - step_size * grad)
        # if step % 5 == 0:
        # print("Step: {}, Cost Objective: {}".format(step, loss.item()))

        # print(
        #     "The updated parameters are betas = {} and gammas = {}".format(
        #         *model.parameters()
        #     )
        # )
    return model(use_qiskit=False, measure_all=True)


"""
Notes:
1. input_graph = [(0, 1), (3, 0), (1, 2), (2, 3)], mixer 1st & entangler 2nd, n_layers >= 2, answer is correct.

"""


def main(use_qiskit):
    # create a input_graph
    input_graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
    n_wires = 4
    n_layers = 1
    model = MAXCUT(n_wires=n_wires, input_graph=input_graph, n_layers=n_layers)

    # set the qiskit processor
    # processor_simulation = QiskitProcessor(use_real_qc=False, n_shots=10000)
    # model.set_qiskit_processor(processor_simulation)

    # firstly perform simulate
    # model.to("cuda")
    # model.to(torch.device("cuda"))
    # circ = tq2qiskit(tq.QuantumDevice(n_wires=4), model)
    # print(circ)
    # print("The circuit is", circ.draw(output="mpl"))
    # circ.draw(output="mpl")
    # use backprop
    # backprop_optimize(model, n_steps=300, lr=0.01)
    # use parameter shift rule
    param_shift_optimize(model, n_steps=500, step_size=0.01, use_qiskit=use_qiskit)


if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()
    use_qiskit = False
    main(use_qiskit)
