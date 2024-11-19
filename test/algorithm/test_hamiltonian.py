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

from torchquantum.algorithm import Hamiltonian
import numpy as np


def test_hamiltonian():
    coeffs = [1.0, 1.0]
    paulis = ["ZZ", "ZX"]
    hamil = Hamiltonian(coeffs, paulis)
    assert np.allclose(
        hamil.get_matrix().cpu().detach().numpy(),
        np.array(
            [
                [1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, -1.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j],
            ]
        ),
    )

    coeffs = [0.6]
    paulis = ["XXZ"]
    hamil = Hamiltonian(coeffs, paulis)
    assert np.allclose(
        hamil.get_matrix().cpu().detach().numpy(),
        np.array(
            [
                [
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.6000 + 0.0j,
                    0.0000 + 0.0j,
                ],
                [
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.6000 + 0.0j,
                ],
                [
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.6000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                ],
                [
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.6000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                ],
                [
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.6000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                ],
                [
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.6000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                ],
                [
                    0.6000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    0.0000 + 0.0j,
                ],
                [
                    0.0000 + 0.0j,
                    -0.6000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                    0.0000 + 0.0j,
                    -0.0000 + 0.0j,
                ],
            ]
        ),
    )
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'algorithm', 'h2.txt')
    hamil = Hamiltonian.from_file(file_path)

    #hamil = Hamiltonian.from_file("./h2.txt")

    assert np.allclose(
        hamil.matrix.cpu().detach().numpy(),
        np.array(
            [
                [
                    -1.0636533 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.1809312 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    -1.0636533 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.1809312 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -1.8369681 + 0.0j,
                    0.0 + 0.0j,
                    0.1809312 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -1.8369681 + 0.0j,
                    0.0 + 0.0j,
                    0.1809312 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.1809312 + 0.0j,
                    0.0 + 0.0j,
                    -0.24521835 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.1809312 + 0.0j,
                    0.0 + 0.0j,
                    -0.24521835 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.1809312 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -1.0636533 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.1809312 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -1.0636533 + 0.0j,
                ],
            ]
        ),
    )
    print("hamiltonian test passed!")


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    test_hamiltonian()
