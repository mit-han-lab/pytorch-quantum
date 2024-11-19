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
from qiskit.quantum_info import Pauli, Statevector

import torchquantum as tq
from torchquantum.plugin import QiskitProcessor, op_history2qiskit
from torchquantum.util import switch_little_big_endian_state

pauli_str_op_dict = {
    "X": Pauli("X"),
    "Y": Pauli("Y"),
    "Z": Pauli("Z"),
    "I": Pauli("I"),
}


@pytest.mark.skip
def test_expval_observable():
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    processor = QiskitProcessor(use_real_qc=False, n_shots=100000)

    for k in range(10):
        # print(k)
        n_wires = random.randint(1, 4)
        obs = random.choices(["X", "Y", "Z", "I"], k=n_wires)
        random_layer = tq.RandomLayer(n_ops=100, wires=list(range(n_wires)))
        qdev = tq.QuantumDevice(n_wires=n_wires, bsz=1, record_op=True)
        random_layer(qdev)
        qiskit_circ = op_history2qiskit(qdev.n_wires, qdev.op_history)

        expval_qiskit_processor = processor.process_circs_get_joint_expval(
            [qiskit_circ], "".join(obs), parallel=False
        )

        operator = pauli_str_op_dict[obs[0]]
        for ob in obs[1:]:
            # note here the order is reversed because qiskit is in little endian
            operator = pauli_str_op_dict[ob] ^ operator
        psi = Statevector(qiskit_circ)
        state_tq = switch_little_big_endian_state(
            qdev.get_states_1d().detach().numpy()
        )[0]
        assert np.allclose(psi.data, state_tq, atol=1e-5)

        expval_qiskit = psi.expectation_value(operator).real
        # print(expval_qiskit_processor, expval_qiskit)
        if (
            n_wires <= 3
        ):  # if too many wires, the stochastic method is not accurate due to limited shots
            assert np.isclose(expval_qiskit_processor, expval_qiskit, atol=0.015)

    print("expval observable test passed")


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    test_expval_observable()
