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

from qiskit import QuantumCircuit

import torchquantum as tq
from torchquantum.plugin import qiskit2tq_op_history


def test_qiskit2tp_op_history():
    circ = QuantumCircuit(3, 3)
    circ.h(0)
    circ.rx(0.1, 1)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.u(0.1, 0.2, 0.3, 0)
    print(circ)
    ops = qiskit2tq_op_history(circ)
    qmodule = tq.QuantumModule.from_op_history(ops)
    print(qmodule.Operator_list)


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    test_qiskit2tp_op_history()
