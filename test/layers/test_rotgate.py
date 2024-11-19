import numpy as np
from qiskit import transpile
from qiskit.circuit.library import GR, GRX, GRY, GRZ
from qiskit_aer import AerSimulator

import torchquantum as tq
from torchquantum.util import find_global_phase, switch_little_big_endian_matrix

all_pairs = [
    {"qiskit": GR, "tq": tq.layer.GlobalR, "params": 2},
    {"qiskit": GRX, "tq": tq.layer.GlobalRX, "params": 1},
    {"qiskit": GRY, "tq": tq.layer.GlobalRY, "params": 1},
    {"qiskit": GRZ, "tq": tq.layer.GlobalRZ, "params": 1},
]

ITERATIONS = 2


def test_rotgates():
    # test each pair
    for pair in all_pairs:
        # test 2-5 wires
        for num_wires in range(2, 5):
            # try multiple random parameters
            for _ in range(ITERATIONS):
                # generate random parameters
                params = [
                    np.random.uniform(-2 * np.pi, 2 * np.pi)
                    for _ in range(pair["params"])
                ]

                # create the qiskit circuit
                qiskit_circuit = pair["qiskit"](num_wires, *params)

                # get the unitary from qiskit
                backend = AerSimulator(method="unitary")
                qiskit_circuit = transpile(qiskit_circuit, backend)
                qiskit_circuit.save_unitary()
                result = backend.run(qiskit_circuit).result()
                unitary_qiskit = result.get_unitary(qiskit_circuit)

                # create tq circuit
                qdev = tq.QuantumDevice(num_wires)
                tq_circuit = pair["tq"](num_wires, *params)
                tq_circuit(qdev)

                # get the unitary from tq
                unitary_tq = tq_circuit.get_unitary(qdev)
                unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

                # phase?
                phase = find_global_phase(unitary_tq, unitary_qiskit, 1e-4)

                assert np.allclose(
                    unitary_tq * phase, unitary_qiskit, atol=1e-6
                ), f"{pair} not equal with {params=}!"
