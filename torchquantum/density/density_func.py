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

import torch
import numpy as np
import torchquantum as tq
import functools
from typing import Callable, Union, Optional, List, Dict
from ..macro import C_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from ..util.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.util import normalize_statevector
from ..functional import  (hadamard,shadamard,paulix,pauliy,pauliz,i,s,t,sx,cnot,
                           cz,cy,swap,sswap,cswap,toffoli,multicnot,multixcnot,rx,ry,rz,rxx,ryy,rzz,rzx,
                           phaseshift,rot,multirz,crx,cry,crz,crot,u1,u2,u3, cu,cu1, cu2, cu3, qubitunitary,
                           qubitunitaryfast,qubitunitarystrict,singleexcitation,h,sh,x,y,z,xx,yy,zz,zx,cx,ccnot, ccx,
                           u,cu, p,cp,cr,cphase,ecr,echoedcrossresonance,qft,sdg,iswap,cs, csdg,csx,chadamard,ccz,
                           dcx,xxminyy,xxplusyy,c3x,tdg,sxdg,ch,r,c4x,rccx,rc3x,globalphase,c3sx)




__all__ = [
    "apply_unitary_density_einsum",
    "apply_unitary_density_bmm",
    "reset",
]


def apply_unitary_density_einsum(density, mat, wires):
    """Apply the unitary to the densitymatrix using torch.einsum method.

    Args:
        density (torch.Tensor): The densitymatrix.
        mat (torch.Tensor): The unitary matrix of the operation.
        wires (int or List[int]): Which qubit the operation is applied to.

    Returns:
        torch.Tensor: The new statevector.
    """

    device_wires = wires
    n_qubit = int((density.dim() - 1) / 2)

    # minus one because of batch
    total_wires = len(density.shape) - 1

    if len(mat.shape) > 2:
        is_batch_unitary = True
        bsz = mat.shape[0]
        shape_extension = [bsz]
    else:
        is_batch_unitary = False
        shape_extension = []

    """
    Compute U \rho
    """
    mat = mat.view(shape_extension + [2] * len(device_wires) * 2)
    mat = mat.type(C_DTYPE).to(density.device)
    if len(mat.shape) > 2:
        # both matrix and state are in batch mode
        # matdag is the dagger of mat
        matdag = torch.conj(mat.permute([0, 2, 1]))
    else:
        # matrix no batch, state in batch mode
        matdag = torch.conj(mat.permute([1, 0]))

    # Tensor indices of the quantum state
    density_indices = ABC[:total_wires]
    print("density_indices", density_indices)

    # Indices of the quantum state affected by this operation
    affected_indices = "".join(ABC_ARRAY[list(device_wires)].tolist())
    print("affected_indices", affected_indices)

    # All affected indices will be summed over, so we need the same number
    # of new indices
    new_indices = ABC[total_wires: total_wires + len(device_wires)]
    print("new_indices", new_indices)

    # The new indices of the state are given by the old ones with the
    # affected indices replaced by the new_indices
    new_density_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(affected_indices, new_indices),
        density_indices,
    )
    print("new_density_indices", new_density_indices)

    # Use the last literal as the indice of batch
    density_indices = ABC[-1] + density_indices
    new_density_indices = ABC[-1] + new_density_indices
    if is_batch_unitary:
        new_indices = ABC[-1] + new_indices

    # We now put together the indices in the notation numpy einsum
    # requires
    einsum_indices = (
        f"{new_indices}{affected_indices}," f"{density_indices}->{new_density_indices}"
    )
    print("einsum_indices", einsum_indices)

    new_density = torch.einsum(einsum_indices, mat, density)

    """
    Compute U \rho U^\dagger
    """
    print("dagger")

    # Tensor indices of the quantum state
    density_indices = ABC[:total_wires]
    print("density_indices", density_indices)

    # Indices of the quantum state affected by this operation
    affected_indices = "".join(
        ABC_ARRAY[[x + n_qubit for x in list(device_wires)]].tolist()
    )
    print("affected_indices", affected_indices)

    # All affected indices will be summed over, so we need the same number
    # of new indices
    new_indices = ABC[total_wires: total_wires + len(device_wires)]
    print("new_indices", new_indices)

    # The new indices of the state are given by the old ones with the
    # affected indices replaced by the new_indices
    new_density_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(affected_indices, new_indices),
        density_indices,
    )
    print("new_density_indices", new_density_indices)

    density_indices = ABC[-1] + density_indices
    new_density_indices = ABC[-1] + new_density_indices
    if is_batch_unitary:
        new_indices = ABC[-1] + new_indices

    # We now put together the indices in the notation numpy einsum
    # requires
    einsum_indices = (
        f"{density_indices}," f"{affected_indices}{new_indices}->{new_density_indices}"
    )
    print("einsum_indices", einsum_indices)

    new_density = torch.einsum(einsum_indices, density, matdag)

    return new_density


def apply_unitary_density_bmm(density, mat, wires):
    """Apply the unitary to the DensityMatrix using torch.bmm method.
    Args:
        state (torch.Tensor): The statevector.
        mat (torch.Tensor): The unitary matrix of the operation.
        wires (int or List[int]): Which qubit the operation is applied to.
    Returns:
        torch.Tensor: The new statevector.
    """
    device_wires = wires
    n_qubit = density.dim() // 2
    mat = mat.type(C_DTYPE).to(density.device)
    """
    Compute U \rho
    """
    devices_dims = [w + 1 for w in device_wires]
    permute_to = list(range(density.dim()))
    for d in sorted(devices_dims, reverse=True):
        del permute_to[d]
    permute_to = permute_to[:1] + devices_dims + permute_to[1:]
    permute_back = list(np.argsort(permute_to))
    original_shape = density.shape
    permuted = density.permute(permute_to).reshape([original_shape[0], mat.shape[-1], -1])

    if len(mat.shape) > 2:
        # both matrix and state are in batch mode
        new_density = mat.bmm(permuted)
    else:
        # matrix no batch, state in batch mode
        bsz = permuted.shape[0]
        expand_shape = [bsz] + list(mat.shape)
        new_density = mat.expand(expand_shape).bmm(permuted)
    new_density = new_density.view(original_shape).permute(permute_back)
    _matrix = torch.reshape(new_density[0], [2 ** n_qubit] * 2)
    """
    Compute \rho U^\dagger 
    """
    matdag = torch.conj(mat)
    matdag = matdag.type(C_DTYPE).to(density.device)

    devices_dims_dag = [n_qubit + w + 1 for w in device_wires]
    permute_to_dag = list(range(density.dim()))
    for d in sorted(devices_dims_dag, reverse=True):
        del permute_to_dag[d]
    permute_to_dag = permute_to_dag + devices_dims_dag
    permute_back_dag = list(np.argsort(permute_to_dag))
    original_shape = new_density.shape
    permuted_dag = new_density.permute(permute_to_dag).reshape([original_shape[0], -1, matdag.shape[-1]])

    if len(matdag.shape) > 2:
        # both matrix and state are in batch mode
        new_density = permuted_dag.bmm(matdag)
    else:
        # matrix no batch, state in batch mode
        bsz = permuted_dag.shape[0]
        expand_shape = [bsz] + list(matdag.shape)
        new_density = permuted_dag.bmm(matdag.expand(expand_shape))
    new_density = new_density.view(original_shape).permute(permute_back_dag)
    return new_density

def gate_wrapper(
        name,
        mat,
        method,
        q_device: tq.NoiseDevice,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
):
    """Perform the phaseshift gate.

    Args:
        name (str): The name of the operation.
        mat (torch.Tensor): The unitary matrix of the gate.
        method (str): 'bmm' or 'einsum' to compute matrix vector
            multiplication.
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

    if params is not None:
        if not isinstance(params, torch.Tensor):
            # this is for qubitunitary gate
            params = torch.tensor(params, dtype=C_DTYPE)

        if name in ["qubitunitary", "qubitunitaryfast", "qubitunitarystrict"]:
            params = params.unsqueeze(0) if params.dim() == 2 else params
        else:
            params = params.unsqueeze(-1) if params.dim() == 1 else params
    wires = [wires] if isinstance(wires, int) else wires

    if static:
        # in static mode, the function is not computed immediately, instead,
        # the unitary of a module will be computed and then applied
        parent_graph.add_func(
            name=name,
            wires=wires,
            parent_graph=parent_graph,
            params=params,
            n_wires=n_wires,
            inverse=inverse,
        )
    else:
        # in dynamic mode, the function is computed instantly
        if isinstance(mat, Callable):
            if n_wires is None or name in [
                "qubitunitary",
                "qubitunitaryfast",
                "qubitunitarystrict",
            ]:
                matrix = mat(params)
            elif name in ["multicnot", "multixcnot"]:
                # this is for gates that can be applied to arbitrary numbers of
                # qubits but no params, such as multicnot
                matrix = mat(n_wires)
            elif name in ["multirz"]:
                # this is for gates that can be applied to arbitrary numbers of
                # qubits such as multirz
                matrix = mat(params, n_wires)
            else:
                matrix = mat(params)

        else:
            matrix = mat

        if inverse:
            matrix = matrix.conj()
            if matrix.dim() == 3:
                matrix = matrix.permute(0, 2, 1)
            else:
                matrix = matrix.permute(1, 0)

        density = q_device.density

        if method == "einsum":
            q_device.states = apply_unitary_density_einsum(density, matrix, wires)
        elif method == "bmm":
            q_device.states = apply_unitary_density_bmm(density, matrix, wires)


def reset(q_device: tq.NoiseDevice, wires, inverse=False) -> None:
    """Reset the target qubits to the state 0. It is a non-unitary operation.

    Args:
        q_device (tq.QuantumDevice): The quantum device.
        wires (int or list): The target wire(s) to reset.
        inverse (bool, optional): If True, performs an inverse reset operation. 
            Defaults to False.

    Returns:
        None.

    Examples:
        >>> device = tq.QuantumDevice(n_wires=3)
        >>> reset(device, wires=1)
        >>> print(device.states)
        [1., 0., 1., 0., 0., 0., 0., 0.]

        >>> reset(device, wires=[0, 2])
        >>> print(device.states)
        [0., 0., 0., 0., 0., 0., 0., 0.]
    """
    state = q_device.states

    wires = [wires] if isinstance(wires, int) else wires

    for wire in wires:
        devices_dim = wire + 1
        permute_to = list(range(state.dim()))
        del permute_to[devices_dim]
        permute_to += [devices_dim]
        permute_back = list(np.argsort(permute_to))

        # permute the target wire to the last dim
        permuted = state.permute(permute_to)

        # reset the state
        permuted[..., 1] = 0

        # permute back
        state = state.permute(permute_back)

    # normalize the magnitude of states
    q_device.states = normalize_statevector(q_device.states)
