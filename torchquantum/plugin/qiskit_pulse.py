from qiskit import pulse

from .qiskit.qiskit_macros import IBMQ_PNAMES


def circ2pulse(circuits, name):
    """
    Convert a circuit to a pulse schedule using the specified backend.

    Args:
        circuits (QuantumCircuit): The input quantum circuit.
        name (str): The name of the backend.

    Returns:
        None.

    Example:
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> circ2pulse(qc, 'ibmq_oslo')
    """

    if name in IBMQ_PNAMES:
        backend = name()
        with pulse.build(backend) as pulse_tq:
            qc = circuits
            qc.measure_all()
            pulse.call(qc)
        pulse_tq.draw()
