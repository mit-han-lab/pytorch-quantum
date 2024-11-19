import functools
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from ..macro import ABC, ABC_ARRAY, C_DTYPE, F_DTYPE

if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
else:
    QuantumDevice = None


def apply_unitary_einsum(state, mat, wires):
    """Apply the unitary to the statevector using torch.einsum method.

    Args:
        state (torch.Tensor): The statevector.
        mat (torch.Tensor): The unitary matrix of the operation.
        wires (int or List[int]): Which qubit the operation is applied to.

    Returns:
        torch.Tensor: The new statevector.

    """
    device_wires = wires

    # minus one because of batch
    total_wires = len(state.shape) - 1

    if len(mat.shape) > 2:
        is_batch_unitary = True
        bsz = mat.shape[0]
        shape_extension = [bsz]
        # try:
        #     assert state.shape[0] == bsz
        # except AssertionError as err:
        #     logger.exception(f"Batch size of Quantum Device must be the same"
        #                      f" with that of gate unitary matrix")
        #     raise err

    else:
        is_batch_unitary = False
        shape_extension = []

    mat = mat.view(shape_extension + [2] * len(device_wires) * 2)

    mat = mat.type(C_DTYPE).to(state.device)

    # Tensor indices of the quantum state
    state_indices = ABC[:total_wires]

    # Indices of the quantum state affected by this operation
    affected_indices = "".join(ABC_ARRAY[list(device_wires)].tolist())

    # All affected indices will be summed over, so we need the same number
    # of new indices
    new_indices = ABC[total_wires : total_wires + len(device_wires)]

    # The new indices of the state are given by the old ones with the
    # affected indices replaced by the new_indices
    new_state_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(affected_indices, new_indices),
        state_indices,
    )

    # try:
    #     cannot support too many qubits...
    #     assert ABC[-1] not in state_indices + new_state_indices  \
    #      + new_indices + affected_indices
    # except AssertionError as err:
    #     logger.exception(f"Cannot support too many qubit.")
    #     raise err

    state_indices = ABC[-1] + state_indices
    new_state_indices = ABC[-1] + new_state_indices
    if is_batch_unitary:
        new_indices = ABC[-1] + new_indices

    # We now put together the indices in the notation numpy einsum
    # requires
    einsum_indices = (
        f"{new_indices}{affected_indices}," f"{state_indices}->{new_state_indices}"
    )

    new_state = torch.einsum(einsum_indices, mat, state)

    return new_state


def apply_unitary_bmm(state, mat, wires):
    """Apply the unitary to the statevector using torch.bmm method.

    Args:
        state (torch.Tensor): The statevector.
        mat (torch.Tensor): The unitary matrix of the operation.
        wires (int or List[int]): Which qubit the operation is applied to.

    Returns:
        torch.Tensor: The new statevector.

    """
    device_wires = wires

    # if len(mat.shape) > 2:
    #         bsz = mat.shape[0]
    #     try:
    #         assert state.shape[0] == bsz
    #     except AssertionError as err:
    #         logger.exception(f"Batch size of Quantum Device must be the same"
    #                          f" with that of gate unitary matrix")
    #         raise err
    mat = mat.type(C_DTYPE).to(state.device)

    devices_dims = [w + 1 for w in device_wires]
    permute_to = list(range(state.dim()))
    for d in sorted(devices_dims, reverse=True):
        del permute_to[d]
    permute_to = permute_to[:1] + devices_dims + permute_to[1:]
    permute_back = list(np.argsort(permute_to))
    original_shape = state.shape
    permuted = state.permute(permute_to).reshape([original_shape[0], mat.shape[-1], -1])

    if len(mat.shape) > 2:
        # both matrix and state are in batch mode
        new_state = mat.bmm(permuted)
    else:
        # matrix no batch, state in batch mode
        bsz = permuted.shape[0]
        expand_shape = [bsz] + list(mat.shape)
        new_state = mat.expand(expand_shape).bmm(permuted)

    new_state = new_state.view(original_shape).permute(permute_back)

    return new_state


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

    # Indices of the quantum state affected by this operation
    affected_indices = "".join(ABC_ARRAY[list(device_wires)].tolist())

    # All affected indices will be summed over, so we need the same number
    # of new indices
    new_indices = ABC[total_wires : total_wires + len(device_wires)]

    # The new indices of the state are given by the old ones with the
    # affected indices replaced by the new_indices
    new_density_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(affected_indices, new_indices),
        density_indices,
    )

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

    new_density = torch.einsum(einsum_indices, mat, density)

    r"""
    Compute U \rho U^\dagger
    """

    # Tensor indices of the quantum state
    density_indices = ABC[:total_wires]

    # Indices of the quantum state affected by this operation
    affected_indices = "".join(
        ABC_ARRAY[[x + n_qubit for x in list(device_wires)]].tolist()
    )

    # All affected indices will be summed over, so we need the same number
    # of new indices
    new_indices = ABC[total_wires : total_wires + len(device_wires)]

    # The new indices of the state are given by the old ones with the
    # affected indices replaced by the new_indices
    new_density_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(affected_indices, new_indices),
        density_indices,
    )

    density_indices = ABC[-1] + density_indices
    new_density_indices = ABC[-1] + new_density_indices
    if is_batch_unitary:
        new_indices = ABC[-1] + new_indices

    # We now put together the indices in the notation numpy einsum
    # requires
    einsum_indices = (
        f"{density_indices}," f"{affected_indices}{new_indices}->{new_density_indices}"
    )

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
    permuted = density.permute(permute_to).reshape(
        [original_shape[0], mat.shape[-1], -1]
    )

    if len(mat.shape) > 2:
        # both matrix and state are in batch mode
        new_density = mat.bmm(permuted)
    else:
        # matrix no batch, state in batch mode
        bsz = permuted.shape[0]
        expand_shape = [bsz] + list(mat.shape)
        new_density = mat.expand(expand_shape).bmm(permuted)
    new_density = new_density.view(original_shape).permute(permute_back)
    r"""
    Compute \rho U^\dagger
    """

    matdag = mat.conj()
    if matdag.dim() == 3:
        matdag = matdag.permute(0, 2, 1)
    else:
        matdag = matdag.permute(1, 0)

    matdag = matdag.type(C_DTYPE).to(density.device)

    devices_dims_dag = [n_qubit + w + 1 for w in device_wires]
    permute_to_dag = list(range(density.dim()))
    for d in sorted(devices_dims_dag, reverse=True):
        del permute_to_dag[d]
    permute_to_dag = permute_to_dag + devices_dims_dag
    permute_back_dag = list(np.argsort(permute_to_dag))
    permuted_dag = new_density.permute(permute_to_dag).reshape(
        [original_shape[0], -1, matdag.shape[0]]
    )

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


_noise_mat_dict = {
    "Bitflip": torch.tensor([[0, 1], [1, 0]], dtype=C_DTYPE),
    "Phaseflip": torch.tensor([[1, 0], [0, -1]], dtype=C_DTYPE)
}


def gate_wrapper(
    name,
    mat,
    method,
    q_device: QuantumDevice,
    wires,
    paramnum=0,
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
            if name in ["qubitunitary", "qubitunitaryfast", "qubitunitarystrict"]:
                # this is for qubitunitary gate
                params = torch.tensor(params, dtype=C_DTYPE)
            else:
                # this is for directly inputting parameters as a number
                params = torch.tensor(params, dtype=F_DTYPE)
        '''
        Check whether user don't set parameters of multi parameters gate 
        in batch mode.
        '''
        if params.dim() == 1 and params.shape[0] == paramnum:
            params = params.unsqueeze(0)
        if name in ["qubitunitary", "qubitunitaryfast", "qubitunitarystrict"]:
            params = params.unsqueeze(0) if params.dim() == 2 else params
        else:
            if params.dim() == 1:
                params = params.unsqueeze(-1)
            elif params.dim() == 0:
                params = params.unsqueeze(-1).unsqueeze(-1)
            # params = params.unsqueeze(-1) if params.dim() == 1 else params
    wires = [wires] if isinstance(wires, int) else wires

    if q_device.record_op:
        q_device.op_history.append(
            {
                "name": name,  # type: ignore
                "wires": np.array(wires).squeeze().tolist(),
                "params": (
                    params.squeeze().detach().cpu().numpy().tolist()
                    if params is not None
                    else None
                ),
                "inverse": inverse,
                "trainable": params.requires_grad if params is not None else False,
            }
        )

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
            elif name in ["multicnot", "multixcnot", "qft"]:
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
        assert np.log2(matrix.shape[-1]) == len(wires)
        
        # TODO: There might be a better way to discriminate noisedevice and normal statevector device
        if q_device.device_name == "noisedevice":
            density = q_device.densities
            if method == "einsum":
                return
            elif method == "bmm":
                '''
                Apply kraus operator if there is noise
                '''
                kraus_dict = q_device.noise_model().kraus_dict()
                if (kraus_dict["Bitflip"] != 0 or kraus_dict["Phaseflip"] != 0):
                    p_identity = 1 - kraus_dict["Bitflip"] ** 2 - kraus_dict["Phaseflip"] ** 2
                    if kraus_dict["Bitflip"] != 0:
                        noise_mat = kraus_dict["Bitflip"] * _noise_mat_dict["Bitflip"]
                        density_noise = apply_unitary_density_bmm(density, noise_mat, wires)
                    if kraus_dict["Phaseflip"] != 0:
                        noise_mat = kraus_dict["Phaseflip"] * _noise_mat_dict["Bitflip"]
                        density_noise = density_noise + apply_unitary_density_bmm(density, noise_mat, wires)
                    density = p_identity * density + density_noise
                q_device.densities = apply_unitary_density_bmm(density, matrix, wires)
        else:
            state = q_device.states
            if method == "einsum":
                q_device.states = apply_unitary_einsum(state, matrix, wires)
            elif method == "bmm":
                q_device.states = apply_unitary_bmm(state, matrix, wires)
