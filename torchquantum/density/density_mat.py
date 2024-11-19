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

import imp
import torch
import torch.nn as nn
import numpy as np
import functools
import torchquantum.functional as tqf
import torchquantum.density.density_func as dfunc
import torchquantum as tqf
from torchquantum.macro import C_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from typing import Union, List, Iterable

__all__ = ["DensityMatrix"]


class DensityMatrix(nn.Module):
    def __init__(self, n_wires: int, bsz: int = 1):
        """Init function for DensityMatrix class(Density Operator)
        Args:
            n_wires (int): how many qubits for the densityMatrix.
        """
        super().__init__()

        self.n_wires = n_wires
        """
        For example, when n_wires=3
        matrix[001110] denotes the index of |001><110|=|index1><index2|
        Set Initial value the density matrix of the pure state |00...00>
        """
        _matrix = torch.zeros(2 ** (2 * self.n_wires), dtype=C_DTYPE)
        _matrix[0] = 1 + 0j
        _matrix = torch.reshape(_matrix, [2] * (2 * self.n_wires))
        self._dims = 2 * self.n_wires
        self.register_buffer("matrix", _matrix)

        repeat_times = [bsz] + [1] * len(self.matrix.shape)
        self._matrix = self.matrix.repeat(*repeat_times)
        self.register_buffer("matrix", self._matrix)

        """
        Remember whether or not a standard matrix on a given wire is contructed
        """
        self.construct = {}
        for key in tqf.func_name_dict.keys():
            self.construct[key] = [False] * n_wires

        """
        Store the constructed operator matrix
        """
        self.operator_matrix = {}
        for key in tqf.func_name_dict.keys():
            self.operator_matrix[key] = {}

    def update_matrix(self, matrix):
        bsz = self._matrix.shape[0]
        self._matrix = matrix
        self.register_buffer("matrix", matrix)
        repeat_times = [bsz] + [1] * len(self.matrix.shape)
        self._matrix = self.matrix.repeat(*repeat_times)
        self.register_buffer("matrix", self._matrix)
        return

    @property
    def dim(self):
        return self._dims

    def vector(self):
        """Return the density matrix as a vector.

        This method reshapes the density matrix `_matrix` into a vector representation.

        Returns:
            torch.Tensor: The density matrix as a vector.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.matrix = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            >>> vector = device.vector()
            >>> print(vector)
            tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        """

        return torch.reshape(self._matrix, [2 ** (2 * self.n_wires)])

    def print_2d(self, index):
        """Print the matrix value at the given index.

        This method prints the matrix value of `matrix[index]`. It reshapes the value into a 2D matrix
        using the `torch.reshape` function and then prints it.

        Args:
            index (int): The index of the matrix value to print.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.matrix = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            >>> device.print_2d(1)
            tensor([[0, 0],
                    [0, 1]])

        """

        _matrix = torch.reshape(self._matrix[index], [2 ** self.n_wires] * 2)
        print(_matrix)

        
    def get_2d_matrix(self, index):
        _matrix = torch.reshape(self._matrix[index], [2 ** self.n_wires] * 2)
        return _matrix


    def trace(self, index):
        """Calculate and return the trace of the density matrix at the given index.

        This method calculates the trace of the density matrix stored in `matrix[index]`
        using the `torch.trace` function and returns the result.

        Args:
            index (int): The index of the matrix value.

        Returns:
            float: The trace of the density matrix.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.matrix = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            >>> device.trace(0)
            tensor(2)
        """
        _matrix = torch.reshape(self._matrix[index], [2 ** self.n_wires] * 2)
        return torch.trace(_matrix)

    def positive_semidefinite(self, index):
        """Check whether the matrix is positive semidefinite by Sylvester's_criterion"""
        return np.all(np.linalg.eigvals(self._matrix[index]) > 0)

    def check_valid(self):
        """Check whether the matrix has trace 1 and is positive semidefinite"""
        for i in range(0, self._matrix.shape[0]):
            if self.trace(i) != 1 or not self.positive_semidefinite(i):
                return False
        return True

    def expand(self, other):
        """Return other tensor self(Notice the order)
        
        Args:
            other (DensityMatrix: Another density matrix
        """
        self._matrix = torch.kron(other._matrix, self._matrix)

    def clone_matrix(self, existing_matrix: torch.Tensor):
        self._matrix = existing_matrix.clone()

    def set_matrix(self, matrix: Union[torch.tensor, List]):
        """
        """

        matrix = torch.tensor(matrix, dtype=C_DTYPE).to(self.matrix.device)
        bsz = matrix.shape[0]
        self.matrix = torch.reshape(
            matrix, [bsz] + [2 ** (2 * self.n_wires), 2 ** (2 * self.n_wires)]
        )

    def evolve(self, operator):
        """Evolve the density matrix in batchmode
        operator has size [2**(2*self.n_wires),2**(2*self.n_wires)]
        """
        """Convert the matrix to vector of shape [bsz,2**(2*self.n_wires)]
           Return U rho U^\dagger
        """
        bsz = self.matrix.shape[0]
        expand_shape = [bsz] + list(operator.shape)

        new_matrix = operator.expand(expand_shape).bmm(
            torch.reshape(self.matrix, [bsz, 2 ** (2 * self.n_wires)])
        )
        self.matrix = torch.reshape(
            new_matrix, [bsz, 2 ** self.n_wires, 2 ** self.n_wires]
        )

    def expectation(self):
        """Expectation of a measurement"""
        return

    def purity(self):
        """Calculate the purity of the DensityMatrix defined as \gamma=tr(\rho^2)"""
        return torch.trace(torch.matmul(self._matrix, self._matrix))

    def partial_trace(self, dims: List[int]):
        """Calculate the partial trace of given sub-dimension, return a new density_matrix.
        
        Args:
            dims:The list of sub-dimension
            For example, If we have 3 qubit, the matrix shape is (8,8),
            We want to do partial trace to qubit 0,2, dims=[0,2].
            First, the matrix should be reshped to (2,2,2,2,2,2)
            then we call  np.einsum('ijiiqi->jq', reshaped_dm)
        """
        return False

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"Density Matrix"

    def hadamard(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a Hadamard gate on the specified wires.

        This method applies a Hadamard gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the Hadamard gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.hadamard(wires=[0, 1], inverse=False)
        """
        dfunc.hadamard(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def shadamard(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a SHadamard gate (square root of Hadamard gate) on the specified wires.

        This method applies a SHadamard gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the SHadamard gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.shadamard(wires=[0, 1], inverse=False)
        """
        dfunc.shadamard(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def paulix(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a Pauli-X gate (also known as the NOT gate) on the specified wires.

        This method applies a Pauli-X gate (NOT gate) on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the Pauli-X gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.paulix(wires=[0, 1], inverse=False)
        """

        dfunc.paulix(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def pauliy(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a Pauli-Y gate on the specified wires.

        This method applies a Pauli-Y gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the Pauli-Y gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.pauliy(wires=[0, 1], inverse=False)
        """

        dfunc.pauliy(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def pauliz(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a Pauli-Z gate on the specified wires.

        This method applies a Pauli-Z gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the Pauli-Z gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.pauliz(wires=[0, 1], inverse=False)
        """
        dfunc.pauliz(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def i(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply an identity gate on the specified wires.

        This method applies an identity gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the identity gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.i(wires=[0, 1], inverse=False)
        """

        dfunc.i(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def s(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply an S gate on the specified wires.

        This method applies an S gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the S gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.s(wires=[0, 1], inverse=False)
        """

        dfunc.s(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def t(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a T gate on the specified wires.

        This method applies a T gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the T gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires. Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.t(wires=[0, 1], inverse=False)
        """

        dfunc.t(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def sx(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a SX gate (square root of Pauli-X gate) on the specified wires.

        This method applies a SX gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the SX gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.sx(wires=[0, 1], inverse=False)
        """

        dfunc.sx(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def cnot(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a controlled-NOT (CNOT) gate on the specified wires.

        This method applies a controlled-NOT gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the CNOT gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.cnot(wires=[0, 1], inverse=False)
        """

        dfunc.cnot(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def cz(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a controlled-Z (CZ) gate on the specified wires.

        This method applies a controlled-Z gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the CZ gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.cz(wires=[0, 1], inverse=False)
        """

        dfunc.cz(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def cy(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a controlled-Y (CY) gate on the specified wires.

        This method applies a controlled-Y gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the CY gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.cy(wires=[0, 1], inverse=False)
        """

        dfunc.cy(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def swap(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a swap gate on the specified wires.

        This method applies a swap gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the swap gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.swap(wires=[0, 1, 2], inverse=False)
        """

        dfunc.swap(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def sswap(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a symmetric swap gate on the specified wires.

        This method applies a symmetric swap gate on the specified wires of the quantum device.
        The gate is applied to all the wires if the inverse flag is set to False.
        If the inverse flag is set to True, the gate is applied to all the wires except the first one.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the symmetric swap gate.
            inverse (bool, optional): If True, apply the gate to all the wires except the first one.
                If False, apply the gate to all the wires.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.sswap(wires=[0, 1, 2], inverse=False)
        """

        dfunc.sswap(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def cswap(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a controlled swap (Fredkin) gate on the specified wires.

        This method applies a controlled swap (Fredkin) gate on the specified wires of the quantum device.
        The gate is applied to all the wires except the control wire if the inverse flag is set to True.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the controlled swap gate.
            inverse (bool, optional): If True, apply the gate as a swap gate
                (i.e., swap the target wires if the control wire is ON).
                If False, apply the gate as a controlled swap gate (Fredkin gate).
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.cswap(wires=[0, 1, 2], inverse=False)
        """

        dfunc.cswap(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def toffoli(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):
        """Apply a Toffoli (CCNOT) gate on the specified wires.

        This method applies a Toffoli (CCNOT) gate on the specified wires of the quantum device.
        The gate is applied to all the wires except the control wires if the inverse flag is set to True.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the Toffoli gate.
            inverse (bool, optional): If True, apply the gate as an X gate
                (i.e., flip the target wire if both control wires are ON).
                If False, apply the gate as a Toffoli gate.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.toffoli(wires=[0, 1, 2], inverse=False)
        """

        dfunc.toffoli(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def multicnot(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a multi-qubit controlled-NOT (CNOT) gate on the specified wires.

        This method applies a multi-qubit controlled-NOT (CNOT) gate on the specified wires of the quantum device.
        The gate is applied to all the wires except the control wire(s) if the inverse flag is set to True.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the CNOT gate.
            inverse (bool, optional): If True, apply the gate as a multi-qubit X gate
                (i.e., flip all target wires except the control wire(s)).
                If False, apply the gate as a multi-qubit CNOT gate.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.multicnot(wires=[0, 1, 2], inverse=False)
        """

        dfunc.multicnot(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def multixcnot(
            self,
            wires: Union[List[int], int],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a multi-qubit X gate or CNOT gate on the specified wires.

        This method applies a multi-qubit X gate or CNOT gate on the specified wires of the quantum device.
        The gate is applied to all the wires except the control wire(s) if the inverse flag is set to True.
        The computation method for applying the gate can be controlled using the comp_method parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the X or CNOT gate.
            inverse (bool, optional): If True, apply the gate as a multi-qubit X gate
                (i.e., flip all target wires except the control wire(s)).
                If False, apply the gate as a multi-qubit CNOT gate.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.multixcnot(wires=[0, 1, 2], inverse=False)
        """

        dfunc.multixcnot(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def rx(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a single-qubit Rx gate on the specified wires.

        This method applies a single-qubit Rx gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` tensor. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the Rx gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the Rx gate. It can be a tensor, numpy array, or a list of floats or ints.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=1)
            >>> params = torch.tensor(0.5)
            >>> device.rx(wires=0, params=params)
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rx(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def ry(
            self,
            wires: Union[List[int], int],
            params: torch.Tensor,
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a single-qubit Ry gate on the specified wires.

        This method applies a single-qubit Ry gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` tensor. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the Ry gate.
            params (torch.Tensor): The parameters of the Ry gate. It should be a tensor.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=1)
            >>> params = torch.tensor(0.5)
            >>> device.ry(wires=0, params=params)
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.ry(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rz(
            self,
            wires: Union[List[int], int],
            params: torch.Tensor,
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a single-qubit Rz gate on the specified wires.

        This method applies a single-qubit Rz gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` tensor. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the Rz gate.
            params (torch.Tensor): The parameters of the Rz gate. It should be a tensor.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=1)
            >>> params = torch.tensor(0.5)
            >>> device.rz(wires=0, params=params)
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rz(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rxx(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a rotation XX gate on the specified wires.

        This method applies a rotation XX gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the rotation XX gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the rotation XX gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.rxx(wires=[0, 1], params=0.1)
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rxx(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def ryy(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a rotation YY gate on the specified wires.

        This method applies a rotation YY gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the rotation YY gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the rotation YY gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.rzz(wires=[0, 1], params=0.1)
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.ryy(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rzz(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a rotation ZZ gate on the specified wires.

        This method applies a controlled ZZ gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the rotation ZZ gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the rotation ZZ gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.rzz(wires=[0, 1], params=0.3)
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rzz(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rzx(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a controlled Rz gate on the specified wires.

        This method applies a controlled Rz gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the controlled Rz gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the controlled Rz gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.rzx(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rzx(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def phaseshift(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a phase shift gate on the specified wires.

        This method applies a phase shift gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the phase shift gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the phase shift gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.phaseshift(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.phaseshift(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rot(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a rotation gate on the specified wires.

        This method applies a rotation gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The wires on which to apply the rotation gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the rotation gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.rot(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.rot(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def multirz(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a multi-controlled Z-rotation gate on the specified control wires.

        This method applies a multi-controlled Z-rotation gate on the specified control wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The control wires on which to apply the multi-controlled Z-rotation gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the multi-controlled Z-rotation gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.multirz(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.multirz(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def crx(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a controlled X-rotation gate on the specified control and target wires.

        This method applies a controlled X-rotation gate on the specified control and target wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the controlled X-rotation gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the controlled X-rotation gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.crx(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.crx(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def cry(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a controlled Y-rotation gate on the specified control and target wires.

        This method applies a controlled Y-rotation gate on the specified control and target wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the controlled Y-rotation gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the controlled Y-rotation gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.cry(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.cry(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def crz(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a controlled phase rotation gate on the specified control and target wires.

        This method applies a controlled phase rotation gate on the specified control and target wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the controlled phase rotation gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the controlled phase rotation gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.crz(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.crz(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def crot(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a controlled-rotation gate on the specified control and target wires.

        This method applies a controlled-rotation gate on the specified control and target wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The control and target wires on which to apply the controlled-rotation gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the controlled-rotation gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.crot(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.crot(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def u1(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a u1 gate on the specified wires.

        This method applies a u1 gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the u1 gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the u1 gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.u1(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.u1(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def u2(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a u2 gate on the specified wires.

        This method applies a u2 gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the u2 gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the u2 gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.u2(wires=[0, 1], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.u2(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def u3(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a u3 gate on the specified wires.

        This method applies a u3 gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the u3 gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the u3 gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.u3(wires=[0, 1], params=[0.1, 0.2, 0.3])

        """
        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.u3(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def cu1(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a controlled-u1 gate on the specified wires.

        This method applies a controlled-u1 gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the controlled-u1 gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the controlled-u1 gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.cu1(wires=[0, 1], params=[0.3, 0.5])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.cu1(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def cu2(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a controlled-u2 gate on the specified wires.

        This method applies a controlled-u2 gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the controlled-u2 gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the controlled-u2 gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.cu2(wires=[0, 1], params=[0.3, 0.5])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.cu2(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def cu3(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a controlled-u3 gate on the specified wires.

        This method applies a controlled-u3 gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the controlled-u3 gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the controlled-u3 gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.cu3(wires=[0, 1], params=[0.5, 0.5, 0.5])

        """
        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.cu3(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def qubitunitary(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a unitary gate on the specified wires.

        This method applies a unitary gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.qubitunitary(wires=[0, 1], params=[0.5, 0.5])
        """

        if isinstance(params, Iterable):
            params = torch.tensor(params, dtype=C_DTYPE)
        else:
            params = torch.tensor([params], dtype=C_DTYPE)

        dfunc.qubitunitary(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def qubitunitaryfast(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a unitary gate on the specified wires (fast method).

        This method applies a unitary gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` flag to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter. This method uses a fast implementation
        for applying the gate.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.qubitunitaryfast(wires=[0, 1], params=[0.5, 0.5])
        """

        if isinstance(params, Iterable):
            params = torch.tensor(params, dtype=C_DTYPE)
        else:
            params = torch.tensor([params], dtype=C_DTYPE)

        dfunc.qubitunitaryfast(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def qubitunitarystrict(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a unitary gate on the specified wires.

        This method applies a unitary gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` parameter to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".

        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.qubitunitarystrict(wires=[0, 1], params=[0.5, 0.5])

        """
        if isinstance(params, Iterable):
            params = torch.tensor(params, dtype=C_DTYPE)
        else:
            params = torch.tensor([params], dtype=C_DTYPE)

        dfunc.qubitunitarystrict(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def single_excitation(
            self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = "bmm",
    ):

        """Apply a single excitation gate on the specified wires.
        
        This method applies a single excitation gate on the specified wires of the quantum device.
        The gate is parametrized by the given `params` values. The gate can be applied in the inverse
        direction by setting the `inverse` parameter to True. The computation method for applying the gate
        can be controlled using the `comp_method` parameter.

        Args:
            wires (Union[List[int], int]): The target wires on which to apply the gate.
            params (Union[torch.Tensor, np.ndarray, List[float], List[int], int, float]):
                The parameters of the gate. It can be a tensor or array-like object.
                If a single value is provided, it will be broadcasted to all parameters.
            inverse (bool, optional): If True, apply the gate in the inverse direction.
                Defaults to False.
            comp_method (str, optional): The computation method for applying the gate.
                Supported options are "bmm" (batch matrix multiplication) and "einsum" (Einstein summation).
                Defaults to "bmm".
        
        Returns:
            None.

        Examples:
            >>> device = QuantumDevice(n_wires=3)
            >>> device.single_excitation(wires=[0, 2], params=[0.1, 0.2])
        """

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.single_excitation(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    h = hadamard
    sh = shadamard
    x = paulix
    y = pauliy
    z = pauliz
    xx = rxx
    yy = ryy
    zz = rzz
    zx = rzx
    cx = cnot
    ccnot = toffoli
    ccx = toffoli
    u = u3
    cu = cu3
    p = phaseshift
    cp = cu1
    cr = cu1
    cphase = cu
