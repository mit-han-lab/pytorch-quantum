from qiskit.circuit.library import (
    EfficientSU2,
    ExcitationPreserving,
    PauliTwoDesign,
    RealAmplitudes,
    TwoLocal,
)

import torchquantum as tq


def compare_tq_to_qiskit(tq_circuit, qiskit_circuit, instance_info=""):
    """
    helper function to compare if tq and qiskit have the same gates configuration
    """

    qiskit_ops = []
    for bit in qiskit_circuit.decompose():
        wires = []
        for qb in bit.qubits:
            wires.append(qiskit_circuit.find_bit(qb).index)
        qiskit_ops.append(
            {
                "name": bit.operation.name,
                "wires": tuple(wires),
            }
        )

    # create operations list
    tq_ops = [
        {
            "name": op["name"],
            "wires": (
                (op["wires"],) if isinstance(op["wires"], int) else tuple(op["wires"])
            ),
        }
        for op in tq_circuit.op_history
    ]

    # create tuples, preserving order
    tq_ops_tuple = [tuple(op) for op in tq_ops]
    qiskit_ops_tuple = [tuple(op) for op in qiskit_ops]

    # assert if they are the same
    assert len(tq_ops) == len(
        qiskit_ops
    ), f"operations are varying lengths for {instance_info}"
    assert (
        tq_ops_tuple == qiskit_ops_tuple
    ), f"operations do not match for {instance_info}"


## TEST TWOLOCAL


def test_twolocal():
    # iterate through different parameters to test
    for entanglement_type in ("linear", "circular", "full"):
        for n_wires in (3, 5, 10):
            for reps in range(1, 5):
                # create the TQ circuit
                tq_two = tq.layer.TwoLocal(
                    n_wires,
                    ["ry", "rz"],
                    "cz",
                    entanglement_layer=entanglement_type,
                    reps=reps,
                )
                qdev = tq.QuantumDevice(n_wires, record_op=True)
                tq_two(qdev)

                # create the qiskit circuit
                qiskit_two = TwoLocal(
                    n_wires,
                    ["ry", "rz"],
                    "cz",
                    entanglement_type,
                    reps=reps,
                    insert_barriers=False,
                )

                # compare the circuits
                test_info = f"{entanglement_type} with {n_wires} wires and {reps} reps"
                compare_tq_to_qiskit(qdev, qiskit_two)


## TEST OTHER CIRCUITS


def test_twolocal_variants():
    tq_to_qiskit = {
        "EfficientSU2": (tq.layer.EfficientSU2, EfficientSU2),
        "ExcitationPreserving": (tq.layer.ExcitationPreserving, ExcitationPreserving),
        "RealAmplitudes": (tq.layer.RealAmplitudes, RealAmplitudes),
        "PauliTwo": (tq.layer.PauliTwoDesign, PauliTwoDesign),
    }

    # run all the tests
    for circuit_name in tq_to_qiskit:
        tq_instance, qiskit_instance = tq_to_qiskit[circuit_name]
        for n_wires in range(2, 5):
            tq_circuit = tq_instance(n_wires)
            circuit = qiskit_instance(n_wires)
            qdev = tq.QuantumDevice(n_wires, record_op=True)
            tq_circuit(qdev)
            compare_tq_to_qiskit(qdev, circuit, f"{circuit_name} with {n_wires} wires")
