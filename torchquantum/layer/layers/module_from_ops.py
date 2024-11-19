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

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torchpack.utils.logging import logger

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.plugin.qiskit import QISKIT_INCOMPATIBLE_FUNC_NAMES

__all__ = [
    "QuantumModuleFromOps",
]


class QuantumModuleFromOps(tq.QuantumModule):
    """Initializes a QuantumModuleFromOps instance.

    Args:
        ops (List[tq.Operation]): List of quantum operations.

    """

    def __init__(self, ops):
        super().__init__()
        self.ops = tq.QuantumModuleList(ops)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        """Performs the forward pass of the quantum module.

        Args:
            q_device (tq.QuantumDevice): Quantum device to apply the operations on.

        Returns:
            None

        """
        q_device.reset_states(1)
        for op in self.ops:
            op(q_device, wires=op.wires)
