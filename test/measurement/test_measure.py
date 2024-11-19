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
from qiskit import transpile
from qiskit_aer import AerSimulator

import torchquantum as tq
from torchquantum.plugin import op_history2qiskit


def test_measure():
    n_shots = 10000
    qdev = tq.QuantumDevice(n_wires=3, bsz=1, record_op=True)
    qdev.x(wires=2)  # type: ignore
    qdev.x(wires=1)  # type: ignore
    qdev.ry(wires=0, params=0.98)  # type: ignore
    qdev.rx(wires=1, params=1.2)  # type: ignore
    qdev.cnot(wires=[0, 2])  # type: ignore

    tq_counts = tq.measure(qdev, n_shots=n_shots)

    circ = op_history2qiskit(qdev.n_wires, qdev.op_history)
    circ.measure_all()
    simulator = AerSimulator()
    circ = transpile(circ, simulator)
    qiskit_res = simulator.run(circ, shots=n_shots).result()
    qiskit_counts = qiskit_res.get_counts()

    for k, v in tq_counts[0].items():
        # need to reverse the bitstring because qiskit is in little endian
        qiskit_ratio = qiskit_counts.get(k[::-1], 0) / n_shots
        tq_ratio = v / n_shots
        print(k, qiskit_ratio, tq_ratio)
        assert np.isclose(qiskit_ratio, tq_ratio, atol=0.1)

    print("tq.measure test passed")


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    test_measure()
