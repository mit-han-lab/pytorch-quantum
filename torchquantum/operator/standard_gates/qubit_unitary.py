from abc import ABCMeta

import numpy as np
import torch

import torchquantum.functional as tqf
from torchquantum.macro import C_DTYPE

from ..op_types import *


class QubitUnitary(Operation, metaclass=ABCMeta):
    """Class for controlled Qubit Unitary gate."""

    num_params = AnyNParams
    num_wires = AnyWires
    op_name = "qubitunitary"
    func = staticmethod(tqf.qubitunitary)

    @classmethod
    def _matrix(cls, params):
        return tqf.qubitunitary_matrix(params)

    def build_params(self, trainable):
        return None

    def reset_params(self, init_params=None):
        self.params = torch.tensor(init_params, dtype=C_DTYPE)
        self.register_buffer(f"{self.name}_unitary", self.params)


class QubitUnitaryFast(Operation, metaclass=ABCMeta):
    """Class for fast implementation of
    controlled Qubit Unitary gate."""

    num_params = AnyNParams
    num_wires = AnyWires
    op_name = "qubitunitaryfast"
    func = staticmethod(tqf.qubitunitaryfast)

    def __init__(
        self,
        has_params: bool = False,
        trainable: bool = False,
        init_params=None,
        n_wires=None,
        wires=None,
    ):
        super().__init__(
            has_params=True,
            trainable=trainable,
            init_params=init_params,
            n_wires=n_wires,
            wires=wires,
        )

    @classmethod
    def from_controlled_operation(
        cls,
        op,
        c_wires,
        t_wires,
        trainable,
    ):
        """

        Args:
            op: the operation
            c_wires: controlled wires, will only be a set such as 1, [2,3]
            t_wires: can be a list of list of wires, multiple sets
            [[1,2], [3,4]]
            trainable:
        """
        op = op
        c_wires = np.array(c_wires)
        t_wires = np.array(t_wires)
        trainable = trainable
        # self.n_t_wires = op.n_wires
        # assert len(t_wires) == op.n_wires

        orig_u = op.matrix
        orig_u_n_wires = op.n_wires

        wires = []

        if c_wires.ndim == 0:
            # only one control qubit
            # 1
            n_c_wires = 1
            wires.append(c_wires.item())
        elif c_wires.ndim == 1:
            # multiple control qubits
            # [1, 2]
            n_c_wires = c_wires.shape[0]
            wires.extend(list(c_wires))

        if t_wires.ndim == 0:
            # single qubit U on one set
            # 2
            n_t_wires = 1
            n_set_t_wires = 1
            wires.append(t_wires.item())
        elif t_wires.ndim == 1:
            # single qubit U on multiple sets
            # [1, 2, 3]
            # or multi qubit U on one set
            # [2, 3]
            n_t_wires = t_wires.shape[0]
            n_set_t_wires = n_t_wires // orig_u_n_wires
            wires.extend(list(t_wires.flatten()))

        elif t_wires.ndim == 2:
            # multi qubit unitary on multiple sets
            # [[2, 3], [4, 5]]
            n_t_wires = t_wires.flatten().shape[0]
            n_set_t_wires = n_t_wires // orig_u_n_wires
            wires.extend(list(t_wires.flatten()))

        n_wires = n_c_wires + n_t_wires

        # compute the new unitary, then permute
        unitary = torch.zeros(2**n_wires, 2**n_wires, dtype=C_DTYPE)
        for k in range(2**n_wires - 2**n_t_wires):
            unitary[k, k] = 1.0 + 0.0j

        # compute kronecker product of all the controlled target

        controlled_u = None
        for k in range(n_set_t_wires):
            if controlled_u is None:
                controlled_u = orig_u
            else:
                controlled_u = torch.kron(controlled_u, orig_u)

        d_controlled_u = controlled_u.shape[-1]
        unitary[-d_controlled_u:, -d_controlled_u:] = controlled_u

        return cls(
            has_params=True,
            trainable=trainable,
            init_params=unitary,
            n_wires=n_wires,
            wires=wires,
        )

    @classmethod
    def _matrix(cls, params):
        return tqf.qubitunitaryfast_matrix(params)

    def build_params(self, trainable):
        return None

    def reset_params(self, init_params=None):
        self.params = torch.tensor(init_params, dtype=C_DTYPE)
        self.register_buffer(f"{self.name}_unitary", self.params)
