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

from torchquantum.functional import func_name_dict
from typing import Iterable
from torchquantum.macro import C_DTYPE
from abc import ABCMeta
from qiskit import QuantumCircuit


class Encoder(tq.QuantumModule):
    """Encodes the given classic state to quantum state.

    Methods:
        - forward(qdev: tq.QuantumDevice, x): Performs the encoding using a quantum device.

    """

    def __init__(self):
        super().__init__()
        pass

    def forward(self, qdev: tq.QuantumDevice, x):
        raise NotImplementedError


class GeneralEncoder(Encoder, metaclass=ABCMeta):
    """func_list list of dict
    Example 1:
    [
    {'input_idx': [0], 'func': 'rx', 'wires': [0]},
    {'input_idx': [1], 'func': 'rx', 'wires': [1]},
    {'input_idx': [2], 'func': 'rx', 'wires': [2]},
    {'input_idx': [3], 'func': 'rx', 'wires': [3]},
    {'input_idx': [4], 'func': 'ry', 'wires': [0]},
    {'input_idx': [5], 'func': 'ry', 'wires': [1]},
    {'input_idx': [6], 'func': 'ry', 'wires': [2]},
    {'input_idx': [7], 'func': 'ry', 'wires': [3]},
    {'input_idx': [8], 'func': 'rz', 'wires': [0]},
    {'input_idx': [9], 'func': 'rz', 'wires': [1]},
    {'input_idx': [10], 'func': 'rz', 'wires': [2]},
    {'input_idx': [11], 'func': 'rz', 'wires': [3]},
    {'input_idx': [12], 'func': 'rx', 'wires': [0]},
    {'input_idx': [13], 'func': 'rx', 'wires': [1]},
    {'input_idx': [14], 'func': 'rx', 'wires': [2]},
    {'input_idx': [15], 'func': 'rx', 'wires': [3]},
    ]

    Example 2:
    [
    {'input_idx': [0, 1, 2], 'func': 'u3', 'wires': [0]},
    {'input_idx': [3], 'func': 'u1', 'wires': [0]},
    {'input_idx': [4, 5, 6], 'func': 'u3', 'wires': [1]},
    {'input_idx': [7], 'func': 'u1', 'wires': [1]},
    {'input_idx': [8, 9, 10], 'func': 'u3', 'wires': [2]},
    {'input_idx': [11], 'func': 'u1', 'wires': [2]},
    {'input_idx': [12, 13, 14], 'func': 'u3', 'wires': [3]},
    {'input_idx': [15], 'func': 'u1', 'wires': [3]},
    ]

    Example 3:
    [
    {'params': [0.25], 'func': 'rx', 'wires': [0]},
    {'params': [0.25], 'func': 'rx', 'wires': [1]},
    {'params': [0.25], 'func': 'rx', 'wires': [2]},
    {'params': [0.25], 'func': 'rx', 'wires': [3]},
    {'input_idx': [0], 'func': 'ry', 'wires': [0]},
    {'input_idx': [1], 'func': 'ry', 'wires': [1]},
    {'input_idx': [2], 'func': 'ry', 'wires': [2]},
    {'input_idx': [3], 'func': 'ry', 'wires': [3]}
    ]
    """

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x):
        for info in self.func_list:
            if tq.op_name_dict[info["func"]].num_params > 0:
                # If params are provided in encoder, use those,
                #  else use params from x
                params = (torch.Tensor(info["params"]).repeat(x.shape[0], 1)
                          if info.get("params")
                          else x[:, info["input_idx"]])
            else:
                params = None
            func_name_dict[info["func"]](
                qdev,
                wires=info["wires"],
                params=params,
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def to_qiskit(self, n_wires, x):
        # assuming the x is in batch mode
        bsz = x.shape[0]

        circs = []
        for k in range(bsz):
            circ = QuantumCircuit(n_wires)
            for info in self.func_list:
                if info["func"] == "rx":
                    circ.rx(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "ry":
                    circ.ry(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rz":
                    circ.rz(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rxx":
                    circ.rxx(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "ryy":
                    circ.ryy(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rzz":
                    circ.rzz(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rzx":
                    circ.rzx(x[k][info["input_idx"][0]].item(), *info["wires"])
                else:
                    raise NotImplementedError(info["func"])
            circs.append(circ)

        return circs


class PhaseEncoder(Encoder, metaclass=ABCMeta):
    """PhaseEncoder is a subclass of Encoder and represents a phase encoder.
    It applies a specified quantum function to encode input data using a quantum device."""

    def __init__(self, func):
        super().__init__()
        self.func = func

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x):
        """
                Performs the encoding using a quantum device.

                Args:
                    qdev (tq.QuantumDevice): The quantum device to be used for encoding.
                    x (torch.Tensor): The input data to be encoded.

                Returns:
                    torch.Tensor: The encoded data.

                """
        for k in range(qdev.n_wires):
            self.func(
                qdev,
                wires=k,
                params=x[:, k],
                static=self.static_mode,
                parent_graph=self.graph,
            )


class MultiPhaseEncoder(Encoder, metaclass=ABCMeta):
    """PhaseEncoder is a subclass of Encoder and represents a phase encoder.
    It applies a specified quantum function to encode input data using a quantum device."""

    def __init__(self, funcs, wires=None):
        super().__init__()
        self.funcs = funcs if isinstance(funcs, Iterable) else [funcs]
        self.wires = wires

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x):
        """
                Performs the encoding using a quantum device.

                Args:
                    qdev (tq.QuantumDevice): The quantum device to be used for encoding.
                    x (torch.Tensor): The input data to be encoded.

                Returns:
                    torch.Tensor: The encoded data.

                """
        if self.wires is None:
            self.wires = list(range(qdev.n_wires)) * (len(self.funcs) // qdev.n_wires)

        x_id = 0
        for k, func in enumerate(self.funcs):
            if func in ["rx", "ry", "rz", "u1", "phaseshift"]:
                stride = 1
            elif func == "u2":
                stride = 2
            elif func == "u3":
                stride = 3
            else:
                raise ValueError(func)

            func_name_dict[func](
                qdev,
                wires=self.wires[k],
                params=x[:, x_id: (x_id + stride)],
                static=self.static_mode,
                parent_graph=self.graph,
            )
            x_id += stride


class StateEncoder(Encoder, metaclass=ABCMeta):
    """StateEncoder is a subclass of Encoder and represents a state encoder.
    It encodes the input data into the state vector of a quantum device."""

    def __init__(self):
        super().__init__()

    def forward(self, qdev: tq.QuantumDevice, x):
        """
            Performs the encoding by preparing the state vector of the quantum device.

                Args:
                    qdev (tq.QuantumDevice): The quantum device to be used for encoding.
                    x (torch.Tensor): The input data to be encoded.
                Returns:
                    torch.Tensor: The encoded data.

                """
        # encoder the x to the statevector of the quantum device

        # normalize the input
        x = x / (torch.sqrt((x.abs() ** 2).sum(dim=-1))).unsqueeze(-1)
        state = torch.cat(
            (
                x,
                torch.zeros(
                    x.shape[0], 2 ** qdev.n_wires - x.shape[1], device=x.device
                ),
            ),
            dim=-1,
        )
        state = state.view([x.shape[0]] + [2] * qdev.n_wires)

        #TODO: Change to united format
        if qdev.device_name == "noisedevice":
            qdev.clone_from_states(state.type(C_DTYPE))
        else:
            qdev.states = state.type(C_DTYPE)


class MagnitudeEncoder(Encoder, metaclass=ABCMeta):
    """MagnitudeEncoder is a subclass of Encoder and represents a magnitude encoder.
    It encodes the input data by considering the magnitudes of the elements."""

    def __init__(self):
        super().__init__()


AmplitudeEncoder = StateEncoder

encoder_op_list_name_dict = {
    "1x1_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
    ],
    "2x1_ryry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
    ],
    "2x8_rxryrzrxryrzrxry": [
        {"input_idx": [0], "func": "rx", "wires": [0]},
        {"input_idx": [1], "func": "rx", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [0]},
        {"input_idx": [3], "func": "ry", "wires": [1]},
        {"input_idx": [4], "func": "rz", "wires": [0]},
        {"input_idx": [5], "func": "rz", "wires": [1]},
        {"input_idx": [6], "func": "rx", "wires": [0]},
        {"input_idx": [7], "func": "rx", "wires": [1]},
        {"input_idx": [8], "func": "ry", "wires": [0]},
        {"input_idx": [9], "func": "ry", "wires": [1]},
        {"input_idx": [10], "func": "rz", "wires": [0]},
        {"input_idx": [11], "func": "rz", "wires": [1]},
        {"input_idx": [12], "func": "rx", "wires": [0]},
        {"input_idx": [13], "func": "rx", "wires": [1]},
        {"input_idx": [14], "func": "ry", "wires": [0]},
        {"input_idx": [15], "func": "ry", "wires": [1]},
    ],
    "3x1_ryryry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
    ],
    "3x1_rxrxrx": [
        {"input_idx": [0], "func": "rx", "wires": [0]},
        {"input_idx": [1], "func": "rx", "wires": [1]},
        {"input_idx": [2], "func": "rx", "wires": [2]},
    ],
    "4_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
    ],
    "4x4_u3rx": [
        {"input_idx": [0, 1, 2], "func": "u3", "wires": [0]},
        {"input_idx": [3], "func": "rx", "wires": [0]},
        {"input_idx": [4, 5, 6], "func": "u3", "wires": [1]},
        {"input_idx": [7], "func": "rx", "wires": [1]},
        {"input_idx": [8, 9, 10], "func": "u3", "wires": [2]},
        {"input_idx": [11], "func": "rx", "wires": [2]},
        {"input_idx": [12, 13, 14], "func": "u3", "wires": [3]},
        {"input_idx": [15], "func": "rx", "wires": [3]},
    ],
    "4x4_u3_h_rx": [
        {"input_idx": [0, 1, 2], "func": "u3", "wires": [0]},
        {"input_idx": [3], "func": "rx", "wires": [0]},
        {"func": "h", "wires": [0]},
        {"func": "h", "wires": [1]},
        {"func": "h", "wires": [2]},
        {"func": "h", "wires": [3]},
        {"input_idx": [4, 5, 6], "func": "u3", "wires": [1]},
        {"input_idx": [7], "func": "rx", "wires": [1]},
        {"input_idx": [8, 9, 10], "func": "u3", "wires": [2]},
        {"input_idx": [11], "func": "rx", "wires": [2]},
        {"input_idx": [12, 13, 14], "func": "u3", "wires": [3]},
        {"input_idx": [15], "func": "rx", "wires": [3]},
    ],
    "4x4_ryzxy": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "rz", "wires": [0]},
        {"input_idx": [5], "func": "rz", "wires": [1]},
        {"input_idx": [6], "func": "rz", "wires": [2]},
        {"input_idx": [7], "func": "rz", "wires": [3]},
        {"input_idx": [8], "func": "rx", "wires": [0]},
        {"input_idx": [9], "func": "rx", "wires": [1]},
        {"input_idx": [10], "func": "rx", "wires": [2]},
        {"input_idx": [11], "func": "rx", "wires": [3]},
        {"input_idx": [12], "func": "ry", "wires": [0]},
        {"input_idx": [13], "func": "ry", "wires": [1]},
        {"input_idx": [14], "func": "ry", "wires": [2]},
        {"input_idx": [15], "func": "ry", "wires": [3]},
    ],
    "8x2_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [0]},
        {"input_idx": [9], "func": "ry", "wires": [1]},
        {"input_idx": [10], "func": "ry", "wires": [2]},
        {"input_idx": [11], "func": "ry", "wires": [3]},
        {"input_idx": [12], "func": "ry", "wires": [4]},
        {"input_idx": [13], "func": "ry", "wires": [5]},
        {"input_idx": [14], "func": "ry", "wires": [6]},
        {"input_idx": [15], "func": "ry", "wires": [7]},
    ],
    "16_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
    ],
    "4x4_rzsx": [
        {"input_idx": [0], "func": "rz", "wires": [0]},
        {"input_idx": None, "func": "sx", "wires": [0]},
        {"input_idx": [1], "func": "rz", "wires": [1]},
        {"input_idx": None, "func": "sx", "wires": [1]},
        {"input_idx": [2], "func": "rz", "wires": [2]},
        {"input_idx": None, "func": "sx", "wires": [2]},
        {"input_idx": [3], "func": "rz", "wires": [3]},
        {"input_idx": None, "func": "sx", "wires": [3]},
        {"input_idx": [4], "func": "rz", "wires": [0]},
        {"input_idx": None, "func": "sx", "wires": [0]},
        {"input_idx": [5], "func": "rz", "wires": [1]},
        {"input_idx": None, "func": "sx", "wires": [1]},
        {"input_idx": [6], "func": "rz", "wires": [2]},
        {"input_idx": None, "func": "sx", "wires": [2]},
        {"input_idx": [7], "func": "rz", "wires": [3]},
        {"input_idx": None, "func": "sx", "wires": [3]},
        {"input_idx": [8], "func": "rz", "wires": [0]},
        {"input_idx": None, "func": "sx", "wires": [0]},
        {"input_idx": [9], "func": "rz", "wires": [1]},
        {"input_idx": None, "func": "sx", "wires": [1]},
        {"input_idx": [10], "func": "rz", "wires": [2]},
        {"input_idx": None, "func": "sx", "wires": [2]},
        {"input_idx": [11], "func": "rz", "wires": [3]},
        {"input_idx": None, "func": "sx", "wires": [3]},
        {"input_idx": [12], "func": "rz", "wires": [0]},
        {"input_idx": None, "func": "sx", "wires": [0]},
        {"input_idx": [13], "func": "rz", "wires": [1]},
        {"input_idx": None, "func": "sx", "wires": [1]},
        {"input_idx": [14], "func": "rz", "wires": [2]},
        {"input_idx": None, "func": "sx", "wires": [2]},
        {"input_idx": [15], "func": "rz", "wires": [3]},
        {"input_idx": None, "func": "sx", "wires": [3]},
    ],
    "15_ryrz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "rz", "wires": [0]},
    ],
    "2x8_ryzxyzxyz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "rz", "wires": [0]},
        {"input_idx": [3], "func": "rz", "wires": [1]},
        {"input_idx": [4], "func": "rx", "wires": [0]},
        {"input_idx": [5], "func": "rx", "wires": [1]},
        {"input_idx": [6], "func": "ry", "wires": [0]},
        {"input_idx": [7], "func": "ry", "wires": [1]},
        {"input_idx": [8], "func": "rz", "wires": [0]},
        {"input_idx": [9], "func": "rz", "wires": [1]},
        {"input_idx": [10], "func": "rx", "wires": [0]},
        {"input_idx": [11], "func": "rx", "wires": [1]},
        {"input_idx": [12], "func": "ry", "wires": [0]},
        {"input_idx": [13], "func": "ry", "wires": [1]},
        {"input_idx": [14], "func": "rz", "wires": [0]},
        {"input_idx": [15], "func": "rz", "wires": [1]},
    ],
    "10_ryzx": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "rz", "wires": [0]},
        {"input_idx": [5], "func": "rz", "wires": [1]},
        {"input_idx": [6], "func": "rz", "wires": [2]},
        {"input_idx": [7], "func": "rz", "wires": [3]},
        {"input_idx": [8], "func": "rx", "wires": [0]},
        {"input_idx": [9], "func": "rx", "wires": [1]},
    ],
    "10_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
    ],
    "25_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
        {"input_idx": [16], "func": "ry", "wires": [16]},
        {"input_idx": [17], "func": "ry", "wires": [17]},
        {"input_idx": [18], "func": "ry", "wires": [18]},
        {"input_idx": [19], "func": "ry", "wires": [19]},
        {"input_idx": [20], "func": "ry", "wires": [20]},
        {"input_idx": [21], "func": "ry", "wires": [21]},
        {"input_idx": [22], "func": "ry", "wires": [22]},
        {"input_idx": [23], "func": "ry", "wires": [23]},
        {"input_idx": [24], "func": "ry", "wires": [24]},
    ],
    "25_ryrz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
        {"input_idx": [16], "func": "ry", "wires": [16]},
        {"input_idx": [17], "func": "ry", "wires": [17]},
        {"input_idx": [18], "func": "ry", "wires": [18]},
        {"input_idx": [19], "func": "ry", "wires": [19]},
        {"input_idx": [20], "func": "ry", "wires": [20]},
        {"input_idx": [21], "func": "rz", "wires": [0]},
        {"input_idx": [22], "func": "rz", "wires": [1]},
        {"input_idx": [23], "func": "rz", "wires": [2]},
        {"input_idx": [24], "func": "rz", "wires": [3]},
    ],
    "6x6_ryzxy": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "rz", "wires": [0]},
        {"input_idx": [11], "func": "rz", "wires": [1]},
        {"input_idx": [12], "func": "rz", "wires": [2]},
        {"input_idx": [13], "func": "rz", "wires": [3]},
        {"input_idx": [14], "func": "rz", "wires": [4]},
        {"input_idx": [15], "func": "rz", "wires": [5]},
        {"input_idx": [16], "func": "rz", "wires": [6]},
        {"input_idx": [17], "func": "rz", "wires": [7]},
        {"input_idx": [18], "func": "rz", "wires": [8]},
        {"input_idx": [19], "func": "rz", "wires": [9]},
        {"input_idx": [20], "func": "rx", "wires": [0]},
        {"input_idx": [21], "func": "rx", "wires": [1]},
        {"input_idx": [22], "func": "rx", "wires": [2]},
        {"input_idx": [23], "func": "rx", "wires": [3]},
        {"input_idx": [24], "func": "rx", "wires": [4]},
        {"input_idx": [25], "func": "rx", "wires": [5]},
        {"input_idx": [26], "func": "rx", "wires": [6]},
        {"input_idx": [27], "func": "rx", "wires": [7]},
        {"input_idx": [28], "func": "rx", "wires": [8]},
        {"input_idx": [29], "func": "rx", "wires": [9]},
        {"input_idx": [30], "func": "ry", "wires": [0]},
        {"input_idx": [31], "func": "ry", "wires": [1]},
        {"input_idx": [32], "func": "ry", "wires": [2]},
        {"input_idx": [33], "func": "ry", "wires": [3]},
        {"input_idx": [34], "func": "ry", "wires": [4]},
        {"input_idx": [35], "func": "ry", "wires": [5]},
    ],
    "6x6_ryrz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
        {"input_idx": [16], "func": "ry", "wires": [16]},
        {"input_idx": [17], "func": "ry", "wires": [17]},
        {"input_idx": [18], "func": "ry", "wires": [18]},
        {"input_idx": [19], "func": "ry", "wires": [19]},
        {"input_idx": [20], "func": "ry", "wires": [20]},
        {"input_idx": [21], "func": "rz", "wires": [0]},
        {"input_idx": [22], "func": "rz", "wires": [1]},
        {"input_idx": [23], "func": "rz", "wires": [2]},
        {"input_idx": [24], "func": "rz", "wires": [3]},
        {"input_idx": [25], "func": "rz", "wires": [4]},
        {"input_idx": [26], "func": "rz", "wires": [5]},
        {"input_idx": [27], "func": "rz", "wires": [6]},
        {"input_idx": [28], "func": "rz", "wires": [7]},
        {"input_idx": [29], "func": "rz", "wires": [8]},
        {"input_idx": [30], "func": "rz", "wires": [9]},
        {"input_idx": [31], "func": "rz", "wires": [10]},
        {"input_idx": [32], "func": "rz", "wires": [11]},
        {"input_idx": [33], "func": "rz", "wires": [12]},
        {"input_idx": [34], "func": "rz", "wires": [13]},
        {"input_idx": [35], "func": "rz", "wires": [14]},
    ],
    "6x6_ryrzrx": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
        {"input_idx": [16], "func": "rz", "wires": [0]},
        {"input_idx": [17], "func": "rz", "wires": [1]},
        {"input_idx": [18], "func": "rz", "wires": [2]},
        {"input_idx": [19], "func": "rz", "wires": [3]},
        {"input_idx": [20], "func": "rz", "wires": [4]},
        {"input_idx": [21], "func": "rz", "wires": [5]},
        {"input_idx": [22], "func": "rz", "wires": [6]},
        {"input_idx": [23], "func": "rz", "wires": [7]},
        {"input_idx": [24], "func": "rz", "wires": [8]},
        {"input_idx": [25], "func": "rz", "wires": [9]},
        {"input_idx": [26], "func": "rz", "wires": [10]},
        {"input_idx": [27], "func": "rz", "wires": [11]},
        {"input_idx": [28], "func": "rz", "wires": [12]},
        {"input_idx": [29], "func": "rz", "wires": [13]},
        {"input_idx": [30], "func": "rz", "wires": [14]},
        {"input_idx": [31], "func": "rz", "wires": [15]},
        {"input_idx": [32], "func": "rx", "wires": [0]},
        {"input_idx": [33], "func": "rx", "wires": [1]},
        {"input_idx": [34], "func": "rx", "wires": [2]},
        {"input_idx": [35], "func": "rx", "wires": [3]},
    ],
    "10x10_ryzxyzxyzxy": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "rz", "wires": [0]},
        {"input_idx": [11], "func": "rz", "wires": [1]},
        {"input_idx": [12], "func": "rz", "wires": [2]},
        {"input_idx": [13], "func": "rz", "wires": [3]},
        {"input_idx": [14], "func": "rz", "wires": [4]},
        {"input_idx": [15], "func": "rz", "wires": [5]},
        {"input_idx": [16], "func": "rz", "wires": [6]},
        {"input_idx": [17], "func": "rz", "wires": [7]},
        {"input_idx": [18], "func": "rz", "wires": [8]},
        {"input_idx": [19], "func": "rz", "wires": [9]},
        {"input_idx": [20], "func": "rx", "wires": [0]},
        {"input_idx": [21], "func": "rx", "wires": [1]},
        {"input_idx": [22], "func": "rx", "wires": [2]},
        {"input_idx": [23], "func": "rx", "wires": [3]},
        {"input_idx": [24], "func": "rx", "wires": [4]},
        {"input_idx": [25], "func": "rx", "wires": [5]},
        {"input_idx": [26], "func": "rx", "wires": [6]},
        {"input_idx": [27], "func": "rx", "wires": [7]},
        {"input_idx": [28], "func": "rx", "wires": [8]},
        {"input_idx": [29], "func": "rx", "wires": [9]},
        {"input_idx": [30], "func": "ry", "wires": [0]},
        {"input_idx": [31], "func": "ry", "wires": [1]},
        {"input_idx": [32], "func": "ry", "wires": [2]},
        {"input_idx": [33], "func": "ry", "wires": [3]},
        {"input_idx": [34], "func": "ry", "wires": [4]},
        {"input_idx": [35], "func": "ry", "wires": [5]},
        {"input_idx": [36], "func": "ry", "wires": [6]},
        {"input_idx": [37], "func": "ry", "wires": [7]},
        {"input_idx": [38], "func": "ry", "wires": [8]},
        {"input_idx": [39], "func": "ry", "wires": [9]},
        {"input_idx": [40], "func": "rz", "wires": [0]},
        {"input_idx": [41], "func": "rz", "wires": [1]},
        {"input_idx": [42], "func": "rz", "wires": [2]},
        {"input_idx": [43], "func": "rz", "wires": [3]},
        {"input_idx": [44], "func": "rz", "wires": [4]},
        {"input_idx": [45], "func": "rz", "wires": [5]},
        {"input_idx": [46], "func": "rz", "wires": [6]},
        {"input_idx": [47], "func": "rz", "wires": [7]},
        {"input_idx": [48], "func": "rz", "wires": [8]},
        {"input_idx": [49], "func": "rz", "wires": [9]},
        {"input_idx": [50], "func": "rx", "wires": [0]},
        {"input_idx": [51], "func": "rx", "wires": [1]},
        {"input_idx": [52], "func": "rx", "wires": [2]},
        {"input_idx": [53], "func": "rx", "wires": [3]},
        {"input_idx": [54], "func": "rx", "wires": [4]},
        {"input_idx": [55], "func": "rx", "wires": [5]},
        {"input_idx": [56], "func": "rx", "wires": [6]},
        {"input_idx": [57], "func": "rx", "wires": [7]},
        {"input_idx": [58], "func": "rx", "wires": [8]},
        {"input_idx": [59], "func": "rx", "wires": [9]},
        {"input_idx": [60], "func": "ry", "wires": [0]},
        {"input_idx": [61], "func": "ry", "wires": [1]},
        {"input_idx": [62], "func": "ry", "wires": [2]},
        {"input_idx": [63], "func": "ry", "wires": [3]},
        {"input_idx": [64], "func": "ry", "wires": [4]},
        {"input_idx": [65], "func": "ry", "wires": [5]},
        {"input_idx": [66], "func": "ry", "wires": [6]},
        {"input_idx": [67], "func": "ry", "wires": [7]},
        {"input_idx": [68], "func": "ry", "wires": [8]},
        {"input_idx": [69], "func": "ry", "wires": [9]},
        {"input_idx": [70], "func": "rz", "wires": [0]},
        {"input_idx": [71], "func": "rz", "wires": [1]},
        {"input_idx": [72], "func": "rz", "wires": [2]},
        {"input_idx": [73], "func": "rz", "wires": [3]},
        {"input_idx": [74], "func": "rz", "wires": [4]},
        {"input_idx": [75], "func": "rz", "wires": [5]},
        {"input_idx": [76], "func": "rz", "wires": [6]},
        {"input_idx": [77], "func": "rz", "wires": [7]},
        {"input_idx": [78], "func": "rz", "wires": [8]},
        {"input_idx": [79], "func": "rz", "wires": [9]},
        {"input_idx": [80], "func": "rx", "wires": [0]},
        {"input_idx": [81], "func": "rx", "wires": [1]},
        {"input_idx": [82], "func": "rx", "wires": [2]},
        {"input_idx": [83], "func": "rx", "wires": [3]},
        {"input_idx": [84], "func": "rx", "wires": [4]},
        {"input_idx": [85], "func": "rx", "wires": [5]},
        {"input_idx": [86], "func": "rx", "wires": [6]},
        {"input_idx": [87], "func": "rx", "wires": [7]},
        {"input_idx": [88], "func": "rx", "wires": [8]},
        {"input_idx": [89], "func": "rx", "wires": [9]},
        {"input_idx": [90], "func": "ry", "wires": [0]},
        {"input_idx": [91], "func": "ry", "wires": [1]},
        {"input_idx": [92], "func": "ry", "wires": [2]},
        {"input_idx": [93], "func": "ry", "wires": [3]},
        {"input_idx": [94], "func": "ry", "wires": [4]},
        {"input_idx": [95], "func": "ry", "wires": [5]},
        {"input_idx": [96], "func": "ry", "wires": [6]},
        {"input_idx": [97], "func": "ry", "wires": [7]},
        {"input_idx": [98], "func": "ry", "wires": [8]},
        {"input_idx": [99], "func": "ry", "wires": [9]},
    ],
    "8x8_ryzxyzxy": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "rz", "wires": [0]},
        {"input_idx": [11], "func": "rz", "wires": [1]},
        {"input_idx": [12], "func": "rz", "wires": [2]},
        {"input_idx": [13], "func": "rz", "wires": [3]},
        {"input_idx": [14], "func": "rz", "wires": [4]},
        {"input_idx": [15], "func": "rz", "wires": [5]},
        {"input_idx": [16], "func": "rz", "wires": [6]},
        {"input_idx": [17], "func": "rz", "wires": [7]},
        {"input_idx": [18], "func": "rz", "wires": [8]},
        {"input_idx": [19], "func": "rz", "wires": [9]},
        {"input_idx": [20], "func": "rx", "wires": [0]},
        {"input_idx": [21], "func": "rx", "wires": [1]},
        {"input_idx": [22], "func": "rx", "wires": [2]},
        {"input_idx": [23], "func": "rx", "wires": [3]},
        {"input_idx": [24], "func": "rx", "wires": [4]},
        {"input_idx": [25], "func": "rx", "wires": [5]},
        {"input_idx": [26], "func": "rx", "wires": [6]},
        {"input_idx": [27], "func": "rx", "wires": [7]},
        {"input_idx": [28], "func": "rx", "wires": [8]},
        {"input_idx": [29], "func": "rx", "wires": [9]},
        {"input_idx": [30], "func": "ry", "wires": [0]},
        {"input_idx": [31], "func": "ry", "wires": [1]},
        {"input_idx": [32], "func": "ry", "wires": [2]},
        {"input_idx": [33], "func": "ry", "wires": [3]},
        {"input_idx": [34], "func": "ry", "wires": [4]},
        {"input_idx": [35], "func": "ry", "wires": [5]},
        {"input_idx": [36], "func": "ry", "wires": [6]},
        {"input_idx": [37], "func": "ry", "wires": [7]},
        {"input_idx": [38], "func": "ry", "wires": [8]},
        {"input_idx": [39], "func": "ry", "wires": [9]},
        {"input_idx": [40], "func": "rz", "wires": [0]},
        {"input_idx": [41], "func": "rz", "wires": [1]},
        {"input_idx": [42], "func": "rz", "wires": [2]},
        {"input_idx": [43], "func": "rz", "wires": [3]},
        {"input_idx": [44], "func": "rz", "wires": [4]},
        {"input_idx": [45], "func": "rz", "wires": [5]},
        {"input_idx": [46], "func": "rz", "wires": [6]},
        {"input_idx": [47], "func": "rz", "wires": [7]},
        {"input_idx": [48], "func": "rz", "wires": [8]},
        {"input_idx": [49], "func": "rz", "wires": [9]},
        {"input_idx": [50], "func": "rx", "wires": [0]},
        {"input_idx": [51], "func": "rx", "wires": [1]},
        {"input_idx": [52], "func": "rx", "wires": [2]},
        {"input_idx": [53], "func": "rx", "wires": [3]},
        {"input_idx": [54], "func": "rx", "wires": [4]},
        {"input_idx": [55], "func": "rx", "wires": [5]},
        {"input_idx": [56], "func": "rx", "wires": [6]},
        {"input_idx": [57], "func": "rx", "wires": [7]},
        {"input_idx": [58], "func": "rx", "wires": [8]},
        {"input_idx": [59], "func": "rx", "wires": [9]},
        {"input_idx": [60], "func": "ry", "wires": [0]},
        {"input_idx": [61], "func": "ry", "wires": [1]},
        {"input_idx": [62], "func": "ry", "wires": [2]},
        {"input_idx": [63], "func": "ry", "wires": [3]},
    ],
    "8x2_ryz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "rz", "wires": [0]},
        {"input_idx": [9], "func": "rz", "wires": [1]},
        {"input_idx": [10], "func": "rz", "wires": [2]},
        {"input_idx": [11], "func": "rz", "wires": [3]},
        {"input_idx": [12], "func": "rz", "wires": [4]},
        {"input_idx": [13], "func": "rz", "wires": [5]},
        {"input_idx": [14], "func": "rz", "wires": [6]},
        {"input_idx": [15], "func": "rz", "wires": [7]},
    ],
    "16x1_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
    ],
}
