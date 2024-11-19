import functools
import torch
import numpy as np

from typing import Callable, Union, Optional, List, Dict, TYPE_CHECKING
from ..macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from ..util.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.util import normalize_statevector

from .gate_wrapper import gate_wrapper, apply_unitary_einsum, apply_unitary_bmm

if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
else:
    QuantumDevice = None


def qubitunitary_matrix(params):
    """Compute unitary matrix for Qubitunitary gate.

    Args:
        params (torch.Tensor): The unitary matrix.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    matrix = params.squeeze(0)
    try:
        assert matrix.shape[-1] == matrix.shape[-2]
    except AssertionError as err:
        logger.exception(f"Operator must be a square matrix.")
        raise err

    try:
        U = matrix.cpu().detach().numpy()
        if matrix.dim() > 2:
            # batched unitary
            bsz = matrix.shape[0]
            assert np.allclose(
                np.matmul(U, np.transpose(U.conj(), [0, 2, 1])),
                np.stack([np.identity(U.shape[-1])] * bsz),
                atol=1e-5,
            )
        else:
            assert np.allclose(
                np.matmul(U, np.transpose(U.conj(), [1, 0])),
                np.identity(U.shape[0]),
                atol=1e-5,
            )
    except AssertionError as err:
        logger.exception(f"Operator must be unitary.")
        raise err

    return matrix


def qubitunitaryfast_matrix(params):
    """Compute unitary matrix for Qubitunitary fast gate.

    Args:
        params (torch.Tensor): The unitary matrix.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    return params.squeeze(0)


def qubitunitarystrict_matrix(params):
    """Compute unitary matrix for Qubitunitary strict gate.
        Strictly be the unitary.

    Args:
        params (torch.Tensor): The unitary matrix.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    params.squeeze(0)
    mat = params
    U, Sigma, V = torch.svd(mat)
    return U.matmul(V)


_qubitunitary_mat_dict = {
    "qubitunitary": qubitunitary_matrix,
    "qubitunitaryfast": qubitunitaryfast_matrix,
    "qubitunitarystrict": qubitunitarystrict_matrix,
}


def qubitunitary(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the qubitunitary gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "qubitunitary"
    mat = _qubitunitary_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        paramnum=4,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )


def qubitunitaryfast(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the qubitunitaryfast gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "qubitunitaryfast"
    mat = _qubitunitary_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )


def qubitunitarystrict(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the qubitunitarystrict = gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "qubitunitarystrict"
    mat = _qubitunitary_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        paramnum=4,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )
