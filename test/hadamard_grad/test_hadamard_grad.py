import numpy as np
import pytest

from examples.hadamard_grad.circ import Circ1, Circ2, Circ3
from examples.hadamard_grad.hadamard_grad import hadamard_grad


@pytest.mark.skip
def test_hadamard_grad():
    """
    We assume the circuits have unique and ordered parameters for now.
    This simplifies the hadamard_grad function so that it only needs to return a list ordered as op_history
    """

    example_circuits = [Circ1, Circ2, Circ3]

    for Circ in example_circuits:
        circ = Circ()
        expval, qdev = circ()

        # hadamard grad
        op_history = qdev.op_history
        n_wires = qdev.n_wires
        observable = "ZZZZ"
        hadamard_grad_result = hadamard_grad(op_history, n_wires, observable)
        hadamard_grad_result = [
            gradient for gradient in hadamard_grad_result if gradient != None
        ]

        # backpropagation
        expval.backward()

        # comparison
        for i, (name, param) in enumerate(circ.named_parameters()):
            assert np.isclose(
                hadamard_grad_result[i], param.grad, atol=0.001
            ), "The gradient for {} is incorrect.".format(name)

    print("tq.hadamard_grad test passed")


if __name__ == "__main__":
    test_hadamard_grad()
