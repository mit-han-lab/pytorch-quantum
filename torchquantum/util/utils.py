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

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from qiskit.exceptions import QiskitError
from torchpack.utils.config import Config
from torchpack.utils.logging import logger

import torchquantum as tq
from torchquantum.macro import C_DTYPE

if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
    from torchquantum.module import QuantumModule
else:
    QuantumModule = None
    QuantumDevice = None


__all__ = [
    "pauli_eigs",
    "diag",
    "Timer",
    "get_unitary_loss",
    "legalize_unitary",
    "switch_little_big_endian_matrix",
    "switch_little_big_endian_state",
    "get_expectations_from_counts",
    "find_global_phase",
    "build_module_op_list",
    "build_module_from_op_list",
    "build_module_description_test",
    "get_p_v_reg_mapping",
    "get_p_c_reg_mapping",
    "get_v_c_reg_mapping",
    "get_cared_configs",
    "get_success_rate",
    "get_provider",
    "get_provider_hub_group_project",
    "normalize_statevector",
    "get_circ_stats",
    "partial_trace",
    "tensor_form",
    "matrix_form",
    "dm_to_mixture_of_state",
    "pauli_string_to_matrix",
    "parameter_shift_gradient",
]


def pauli_eigs(n) -> np.ndarray:
    r"""Eigenvalues for :math:`A^{\o times n}`, where :math:`A` is
    Pauli operator, or shares its eigenvalues.

    As an example if n==2, then the eigenvalues of a tensor product consisting
    of two matrices sharing the eigenvalues with Pauli matrices is returned.

    Args:
        n (int): the number of qubits the matrix acts on
    Returns:
        list: the eigenvalues of the specified observable
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([pauli_eigs(n - 1), -pauli_eigs(n - 1)])


def diag(x):
    """
    Compute the diagonal matrix from a given input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Diagonal matrix with the diagonal elements from the input tensor.
    """
    # input tensor, output tensor with diagonal as the input
    # manual implementation because torch.diag does not support autograd of
    # complex number
    diag_len = x.shape[-1]
    x = x.unsqueeze(-1)
    dims = list(x.shape)
    x = torch.cat([x, torch.zeros(dims[:-1] + [diag_len]).to(x.device)], dim=-1)
    x = x.view(dims[:-2] + [diag_len * (diag_len + 1)])[..., :-diag_len]
    x = x.view(dims[:-2] + [diag_len, diag_len])
    return x


class Timer(object):
    """
    Timer class to measure the execution time of a code block.

    Args:
        device (str): Device to use for timing. Can be "gpu" or "cpu".
        name (str): Name of the task being measured.
        times (int): Number of times the task will be executed.

    Example:
        # Measure the execution time of a code block on the GPU
        with Timer(device="gpu", name="MyTask", times=100):
            # Code block to be measured
            ...

    """

    def __init__(self, device="gpu", name="", times=100):
        self.device = device
        self.name = name
        self.times = times
        if device == "gpu":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.device == "gpu":
            self.start.record()

    def __exit__(self, exc_type, exc_value, tb):
        if self.device == "gpu":
            self.end.record()
            torch.cuda.synchronize()
            print(
                f"Task: {self.name}: "
                f"{self.start.elapsed_time(self.end) / self.times} ms"
            )


def get_unitary_loss(model: nn.Module):
    """
    Calculate the unitary loss of a model.

    The unitary loss measures the deviation of the trainable unitary matrices
    in the model from the identity matrix.

    Args:
        model (nn.Module): The model containing trainable unitary matrices.

    Returns:
        torch.Tensor: The unitary loss.

    Example:
        loss = get_unitary_loss(model)
    """
    loss = 0
    for name, params in model.named_parameters():
        if "TrainableUnitary" in name:
            U = params
            like_identity = U.matmul(U.conj().permute(0, 2, 1))
            identity = torch.eye(U.shape[0], dtype=C_DTYPE, device=U.device)
            loss += F.mse_loss(
                torch.view_as_real(identity), torch.view_as_real(like_identity)
            )

    return loss


def legalize_unitary(model: nn.Module):
    """
    Legalize the unitary matrices in the model.

    The function modifies the trainable unitary matrices in the model by applying
    singular value decomposition (SVD) and reassembling the matrices using the
    reconstructed singular values.

    Args:
        model (nn.Module): The model containing trainable unitary matrices.

    Returns:
        None

    Example:
        legalize_unitary(model)
    """
    with torch.no_grad():
        for name, params in model.named_parameters():
            if "TrainableUnitary" in name:
                U = params
                U, Sigma, V = torch.svd(U)
                params.data.copy_(U.matmul(V.conj().permute(0, 2, 1)))


def switch_little_big_endian_matrix(mat):
    """
    Switches the little-endian and big-endian order of a multi-dimensional matrix.

    The function reshapes the input matrix to a 2D or multi-dimensional matrix with dimensions
    that are powers of 2. It then switches the order of the dimensions, effectively changing
    the little-endian order to big-endian, or vice versa. The function can handle both
    batched and non-batched matrices.

    Args:
        mat (numpy.ndarray): The input matrix.

    Returns:
        numpy.ndarray: The matrix with the switched endian order.

    Example:
        switched_mat = switch_little_big_endian_matrix(mat)
    """
    if len(mat.shape) % 2 == 1:
        is_batch_matrix = True
        bsz = mat.shape[0]
        reshape = [bsz] + [2] * int(np.log2(mat[0].size))
    else:
        is_batch_matrix = False
        reshape = [2] * int(np.log2(mat.size))

    original_shape = mat.shape
    mat = mat.reshape(reshape)
    axes = list(range(len(mat.shape) // 2))
    axes.reverse()
    axes += [axis + len(mat.shape) // 2 for axis in axes]

    if is_batch_matrix:
        axes = [0] + [axis + 1 for axis in axes]

    mat = np.transpose(mat, axes=axes).reshape(original_shape)
    return mat


def switch_little_big_endian_state(state):
    """
    Switches the little-endian and big-endian order of a quantum state vector.

    The function reshapes the input state vector to a 1D or multi-dimensional state vector with
    dimensions that are powers of 2. It then switches the order of the dimensions, effectively
    changing the little-endian order to big-endian, or vice versa. The function can handle both
    batched and non-batched state vectors.

    Args:
        state (numpy.ndarray): The input state vector.

    Returns:
        numpy.ndarray: The state vector with the switched endian order.

    Raises:
        ValueError: If the dimension of the state vector is not 1 or 2.

    Example:
        switched_state = switch_little_big_endian_state(state)
    """

    if len(state.shape) > 1:
        is_batch_state = True
        bsz = state.shape[0]
        reshape = [bsz] + [2] * int(np.log2(state[0].size))
    elif len(state.shape) == 1:
        is_batch_state = False
        reshape = [2] * int(np.log2(state.size))
    else:
        logger.exception("Dimension of statevector should be 1 or 2")
        raise ValueError

    original_shape = state.shape
    state = state.reshape(reshape)

    if is_batch_state:
        axes = list(range(1, len(state.shape)))
        axes.reverse()
        axes = [0] + axes
    else:
        axes = list(range(len(state.shape)))
        axes.reverse()

    mat = np.transpose(state, axes=axes).reshape(original_shape)

    return mat


def switch_little_big_endian_matrix_test():
    logger.info(switch_little_big_endian_matrix(np.ones((16, 16))))
    logger.info(switch_little_big_endian_matrix(np.ones((5, 16, 16))))


def switch_little_big_endian_state_test():
    logger.info(switch_little_big_endian_state(np.ones((5, 16))))
    logger.info(switch_little_big_endian_state(np.arange(8)))


def get_expectations_from_counts(counts, n_wires):
    """
    Calculate expectation values from counts.

    This function takes a counts dictionary or a list of counts dictionaries
    and calculates the expectation values based on the probability of measuring
    the state '1' on each wire. The expectation values are computed as the
    flipped difference between the probability of measuring '1' and the probability
    of measuring '0' on each wire.

    Args:
        counts (dict or list[dict]): The counts dictionary or a list of counts dictionaries.
        n_wires (int): The number of wires.

    Returns:
        numpy.ndarray: The expectation values.

    Example:
        counts = {'000': 10, '100': 5, '010': 15}
        expectations = get_expectations_from_counts(counts, 3)
    """
    exps = []
    if isinstance(counts, dict):
        counts = [counts]
    for count in counts:
        ctr_one = [0] * n_wires
        total_shots = 0
        for k, v in count.items():
            for wire in range(n_wires):
                if k[wire] == "1":
                    ctr_one[wire] += v
            total_shots += v
        prob_one = np.array(ctr_one) / total_shots
        exp = np.flip(-1 * prob_one + 1 * (1 - prob_one))
        exps.append(exp)
    res = np.stack(exps)
    return res


def find_global_phase(mat1, mat2, threshold):
    """
    Find a numerical stable global phase between two matrices.

    This function compares the elements of two matrices `mat1` and `mat2`
    and identifies a numerical stable global phase by finding the first
    non-zero element pair with absolute values greater than the specified
    threshold. The global phase is calculated as the ratio of the corresponding
    elements in `mat2` and `mat1`.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.
        threshold (float): The threshold for identifying non-zero elements.

    Returns:
        float or None: The global phase ratio if a numerical stable phase is found,
            None otherwise.

    Example:
        mat1 = np.array([[1+2j, 0+1j], [0-1j, 2+3j]])
        mat2 = np.array([[2+4j, 0+2j], [0-2j, 4+6j]])
        threshold = 0.5
        global_phase = find_global_phase(mat1, mat2, threshold)
    """
    if not isinstance(mat1, np.ndarray):
        mat1 = np.asarray(mat1)
    if not isinstance(mat2, np.ndarray):
        mat2 = np.asarray(mat2)
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            # find a numerical stable global phase
            if np.abs(mat1[i][j]) > threshold and np.abs(mat2[i][j]) > threshold:
                return mat2[i][j] / mat1[i][j]
    return None


def build_module_op_list(m: QuantumModule, x=None) -> list:
    """
    serialize all operations in the module and generate a list with
    [{'name': RX, 'has_params': True, 'trainable': True, 'wires': [0],
    n_wires: 1, 'params': [array([[0.01]])]}]
    so that an identity module can be reconstructed
    The module needs to have static support
    """

    m.static_off()
    m.static_on(wires_per_block=None)
    m.is_graph_top = False

    # forward to register all modules and parameters
    if x is None:
        m.forward(q_device=None)
    else:
        m.forward(q_device=None, x=x)

    m.is_graph_top = True
    m.graph.build_flat_module_list()

    module_list = m.graph.flat_module_list
    m.static_off()

    op_list = []

    for module in module_list:
        if module.params is not None:
            if module.params.shape[0] > 1:
                # more than one param, so it is from classical input with
                # batch mode
                assert not module.has_params
                params = None
            else:
                # has quantum params, batch has to be 1
                params = module.params[0].data.cpu().numpy()
        else:
            params = None

        op_list.append(
            {
                "name": module.name.lower(),
                "has_params": module.has_params,
                "trainable": module.trainable,
                "wires": module.wires,
                "n_wires": module.n_wires,
                "params": params,
            }
        )

    return op_list


def build_module_from_op_list(
    op_list: list[dict], remove_ops=False, thres=None
) -> QuantumModule:
    """
    Build a quantum module from an operation list.

    This function takes an operation list, which contains dictionaries representing
    quantum operations, and constructs a quantum module from those operations.
    The module can optionally remove operations based on certain criteria, such as
    low parameter values. The removed operations can be counted and logged.

    Args:
        op_list (List[Dict]): The operation list, where each dictionary represents
            an operation with keys: "name", "has_params", "trainable", "wires",
            "n_wires", and "params".
        remove_ops (bool): Whether to remove operations based on certain criteria.
            Defaults to False.
        thres (float): The threshold for removing operations. If a parameter value
            is smaller in absolute value than this threshold, the corresponding
            operation is removed. Defaults to None, in which case a threshold of
            1e-5 is used.

    Returns:
        QuantumModule: The constructed quantum module.

    Example:
        op_list = [
            {"name": "RX", "has_params": True, "trainable": True, "wires": [0], "n_wires": 2, "params": [0.5]},
            {"name": "CNOT", "has_params": False, "trainable": False, "wires": [0, 1], "n_wires": 2, "params": None},
            {"name": "RY", "has_params": True, "trainable": True, "wires": [1], "n_wires": 2, "params": [1.2]},
        ]
        module = build_module_from_op_list(op_list, remove_ops=True, thres=0.1)
    """
    logger.info("Building module from op_list...")
    thres = 1e-5 if thres is None else thres
    n_removed_ops = 0
    ops = []
    for info in op_list:
        params = info["params"]

        if remove_ops:
            if params is not None:
                params = (
                    np.array(params)
                    if isinstance(params, Iterable)
                    else np.array([params])
                )
                params = params % (2 * np.pi)
                params[params > np.pi] -= 2 * np.pi
                if all(abs(params) < thres):
                    n_removed_ops += 1
                    continue

        op = tq.op_name_dict[info["name"]](
            has_params=info["has_params"],
            trainable=info["trainable"],
            wires=info["wires"],
            n_wires=info["n_wires"],
            init_params=info["params"],
        )
        ops.append(op)

    if n_removed_ops > 0:
        logger.warning(f"Remove in total {n_removed_ops} pruned operations.")
    else:
        logger.info("Do not remove any operations.")

    return tq.QuantumModuleFromOps(ops)


def build_module_description_test():
    """
    Test function for building module descriptions.

    This function demonstrates the usage of `build_module_op_list` and `build_module_from_op_list`
    functions to build module descriptions and create quantum modules from those descriptions.

    Example:
        import pdb
        from torchquantum.plugins import tq2qiskit
        from examples.core.models.q_models import QFCModel12

        pdb.set_trace()
        q_model = QFCModel12({"n_blocks": 4})
        desc = build_module_op_list(q_model.q_layer)
        print(desc)
        q_dev = tq.QuantumDevice(n_wires=4)
        m = build_module_from_op_list(desc)
        tq2qiskit(q_dev, m, draw=True)

        desc = build_module_op_list(
            tq.RandomLayerAllTypes(n_ops=200, wires=[0, 1, 2, 3], qiskit_compatible=True)
        )
        print(desc)
        m1 = build_module_from_op_list(desc)
        tq2qiskit(q_dev, m1, draw=True)
    """
    import pdb

    from torchquantum.plugin import tq2qiskit

    pdb.set_trace()
    from examples.core.models.q_models import QFCModel12

    q_model = QFCModel12({"n_blocks": 4})
    desc = build_module_op_list(q_model.q_layer)
    print(desc)
    q_dev = tq.QuantumDevice(n_wires=4)
    m = build_module_from_op_list(desc)
    tq2qiskit(q_dev, m, draw=True)

    desc = build_module_op_list(
        tq.RandomLayerAllTypes(n_ops=200, wires=[0, 1, 2, 3], qiskit_compatible=True)
    )
    print(desc)
    m1 = build_module_from_op_list(desc)
    tq2qiskit(q_dev, m1, draw=True)


def get_p_v_reg_mapping(circ):
    """
    p are physical qubits
    v are logical qubits
    """
    try:
        p2v_orig = circ._layout.final_layout.get_physical_bits().copy()
    except AttributeError:
        p2v_orig = circ._layout.get_physical_bits().copy()
    mapping = {
        "p2v": {},
        "v2p": {},
    }

    for p, v in p2v_orig.items():
        if v.register.name == "q":
            mapping["p2v"][p] = v.index
            mapping["v2p"][v.index] = p

    return mapping


def get_p_c_reg_mapping(circ):
    """
    p are physical qubits
    c are classical registers
    """
    mapping = {
        "p2c": {},
        "c2p": {},
    }
    for gate in circ.data:
        if gate[0].name == "measure":
            mapping["p2c"][gate[1][0].index] = gate[2][0].index
            mapping["c2p"][gate[2][0].index] = gate[1][0].index

    return mapping


def get_v_c_reg_mapping(circ):
    """
    p are physical qubits, the real fabricated qubits
    v are logical qubits, also the 'wires' in torchquantum lib
    c are classical registers
    want to get v2c
    """
    try:
        p2v_orig = circ._layout.final_layout.get_physical_bits().copy()
    except AttributeError:
        p2v_orig = circ._layout.get_physical_bits().copy()
    p2v = {}
    for p, v in p2v_orig.items():
        if v.register.name == "q":
            p2v[p] = v.index

    mapping = {
        "p2c": {},
        "c2p": {},
    }
    for gate in circ.data:
        if gate[0].name == "measure":
            mapping["p2c"][gate[1][0].index] = gate[2][0].index
            mapping["c2p"][gate[2][0].index] = gate[1][0].index

    mapping2 = {"v2c": {}, "c2v": {}}

    for c, p in mapping["c2p"].items():
        mapping2["c2v"][c] = p2v[p]

    for c, v in mapping2["c2v"].items():
        mapping2["v2c"][v] = c

    return mapping2


def get_cared_configs(conf, mode) -> Config:
    """
    Get the relevant configurations based on the mode.

    Args:
        conf (Config): The configuration object.
        mode (str): The mode indicating the desired configuration.

    Returns:
        Config: The modified configuration object with only the relevant configurations preserved.
    """

    conf = copy.deepcopy(conf)
    ignores = [
        "callbacks",
        "criterion",
        "debug",
        "legalization",
        "regularization",
        "verbose",
        "get_n_params",
        "prune",
    ]

    if "super" not in conf.trainer.name:
        ignores.append("scheduler")
        ignores.append("optimizer")

    for ignore in ignores:
        if hasattr(conf, ignore):
            delattr(conf, ignore)

    if hasattr(conf, "dataset"):
        dataset_ignores = [
            "binarize",
            "binarize_threshold",
            "center_crop",
            "name",
            "resize",
            "resize_mode",
            "root",
            "train_valid_split_ratio",
        ]
        for dataset_ignore in dataset_ignores:
            if hasattr(conf.dataset, dataset_ignore):
                delattr(conf.dataset, dataset_ignore)

    if not mode == "es" and hasattr(conf, "es"):
        delattr(conf, "es")
    elif mode == "es" and hasattr(conf, "es") and hasattr(conf.es, "eval"):
        delattr(conf.es, "eval")

    if not mode == "train" and hasattr(conf, "trainer"):
        delattr(conf, "trainer")

    if hasattr(conf, "qiskit"):
        qiskit_ignores = [
            "seed_simulator",
            "seed_transpiler",
            "coupling_map_name",
            "basis_gates_name",
            "est_success_rate",
        ]
        for qiskit_ignore in qiskit_ignores:
            if hasattr(conf.qiskit, qiskit_ignore):
                delattr(conf.qiskit, qiskit_ignore)

    if hasattr(conf, "run"):
        run_ignores = ["device", "workers_per_gpu", "n_epochs"]
        for run_ignore in run_ignores:
            if hasattr(conf.run, run_ignore):
                delattr(conf.run, run_ignore)

    return conf


def get_success_rate(properties, transpiled_circ):
    """
    Estimate the success rate of a transpiled quantum circuit.

    Args:
        properties (list): List of gate error properties.
        transpiled_circ (QuantumCircuit): The transpiled quantum circuit.

    Returns:
        float: The estimated success rate.
    """
    raise NotImplementedError


def get_provider(backend_name, hub=None):
    """
    Get the provider object for a specific backend from IBM Quantum.

    Args:
        backend_name (str): Name of the backend.
        hub (str): Optional hub name.

    Returns:
        IBMQProvider: The provider object.
    """
    # mass-inst-tech-1 or MIT-1
    if backend_name in ["ibmq_casablanca", "ibmq_rome", "ibmq_bogota", "ibmq_jakarta"]:
        if hub == "mass" or hub is None:
            provider = QiskitRuntimeService(
                channel="ibm_quantum", instance="ibm-q-research/mass-inst-tech-1/main"
            )
        elif hub == "mit":
            provider = QiskitRuntimeService(
                channel="ibm_quantum", instance="ibm-q-research/MIT-1/main"
            )
        else:
            raise ValueError(f"not supported backend {backend_name} in hub " f"{hub}")
    elif backend_name in [
        "ibmq_paris",
        "ibmq_toronto",
        "ibmq_manhattan",
        "ibmq_guadalupe",
        "ibmq_montreal",
    ]:
        provider = QiskitRuntimeService(
            channel="ibm_quantum", instance="ibm-q-ornl/anl/csc428"
        )
    else:
        if hub == "mass" or hub is None:
            try:
                provider = QiskitRuntimeService(
                    channel="ibm_quantum",
                    instance="ibm-q-research/mass-inst-tech-1/main",
                )
            except QiskitError:
                # logger.warning(f"Cannot use MIT backend, roll back to open")
                logger.warning("Use the open backend")
                provider = QiskitRuntimeService(
                    channel="ibm_quantum", instance="ibm-q/open/main"
                )
        elif hub == "mit":
            provider = QiskitRuntimeService(
                channel="ibm_quantum", instance="ibm-q-research/MIT-1/main"
            )
        else:
            provider = QiskitRuntimeService(
                channel="ibm_quantum", instance="ibm-q/open/main"
            )

    return provider


def get_provider_hub_group_project(hub="ibm-q", group="open", project="main"):
    provider = QiskitRuntimeService(
        channel="ibm_quantum", instance=f"{hub}/{group}/{project}"
    )
    return provider


def normalize_statevector(states):
    """
    Normalize a statevector to ensure the square magnitude of the statevector sums to 1.

    Args:
        states (torch.Tensor): The statevector tensor.

    Returns:
        torch.Tensor: The normalized statevector tensor.
    """
    # make sure the square magnitude of statevector sum to 1
    # states = states.contiguous()
    original_shape = states.shape
    states_reshape = states.reshape(states.shape[0], -1)

    # for states with no energy, need to set all zero state as energy 1
    energy = (abs(states_reshape) ** 2).sum(dim=-1)
    if energy.min() == 0:
        for k, val in enumerate(energy):
            if val == 0:
                states_reshape[k][0] = 1

    factors = torch.sqrt(1 / ((abs(states_reshape) ** 2).sum(dim=-1))).unsqueeze(-1)
    states = (states_reshape * factors).reshape(original_shape)

    return states


def get_circ_stats(circ):
    depth = circ.depth()
    width = circ.width()
    size = circ.size()
    n_single_gates = 0
    n_two_gates = 0
    n_three_more_gates = 0
    n_gates_dict = {}
    n_measure = 0

    for gate in circ.data:
        op_name = gate[0].name
        wires = [circ.find_bit(qb).index for qb in gate.qubits]
        if op_name in n_gates_dict.keys():
            n_gates_dict[op_name] += 1
        else:
            n_gates_dict[op_name] = 1

        if op_name == "measure":
            n_measure += 1
        elif len(wires) == 1:
            n_single_gates += 1
        elif len(wires) == 2:
            n_two_gates += 1
        else:
            n_three_more_gates += 1

    return {
        "depth": depth,
        "size": size,
        "width": width,
        "n_single_gates": n_single_gates,
        "n_two_gates": n_two_gates,
        "n_three_more_gates": n_three_more_gates,
        "n_gates_dict": n_gates_dict,
    }


def partial_trace(
    q_device: QuantumDevice,
    keep_indices: list[int],
) -> torch.Tensor:
    """Returns a density matrix with only some qubits kept.
    Args:
        q_device: The q_device to take the partial trace over.
        keep_indices: Which indices to take the partial trace of the
            state_vector on.
    Returns:
        A density matrix with only the qubits specified by keep_indices.
    """
    n_wires = q_device.n_wires
    keep_indices = np.array(keep_indices) + 1
    dm_left_index = np.arange(n_wires + 1)
    dm_right_index = np.arange(n_wires + 1)
    dm_right_index[keep_indices] *= -1

    dm_out_index = np.array([0])
    dm_out_index = np.concatenate((dm_out_index, keep_indices))
    dm_out_index = np.concatenate((dm_out_index, -keep_indices))

    dm = contract(
        q_device.states,
        dm_left_index,
        q_device.states.conj(),
        dm_right_index,
        dm_out_index,
    )
    return dm


def tensor_form(dm: torch.Tensor):
    """Returns the tensor form of a density matrix.
    Args:
        dm: a (batched) density matrix.
    Returns:
        The tensor form of the density matrix.
    """
    size = dm.size()
    batched = len(size) % 2 == 1
    if batched:
        n_wires = int(np.log2(np.prod(size) / size[0]) / 2)
        shape_list = [size[0]] + [2] * (2 * n_wires)
    else:
        n_wires = int(np.log2(np.prod(size)) / 2)
        shape_list = [2] * (2 * n_wires)
    return dm.reshape(shape_list)


def matrix_form(dm: torch.Tensor):
    """Returns the matrix form of a density matrix.
    Args:
        dm: a (batched) density matrix.
    Returns:
        The matrix form of the density matrix.
    """
    size = dm.size()
    batched = len(size) % 2 == 1
    if batched:
        n_wires = int(np.log2(np.prod(size) / size[0]) / 2)
        shape_list = [size[0]] + [2**n_wires] * 2
    else:
        n_wires = int(np.log2(np.prod(size)) / 2)
        shape_list = [2**n_wires] * 2
    return dm.reshape(shape_list)


def dm_to_mixture_of_state(dm: torch.Tensor, atol=1e-10):
    """
    Args:
        q_device: The state vector to take the partial trace over.
        keep_indices: Which indices to take the partial trace of the
            state_vector on.
        atol: The tolerance for determining that a factored state is pure.
    """
    size = dm.size()
    batched = len(size) % 2 == 1
    if batched:
        dims = int(np.log2(np.prod(size) / size[0]) / 2)

        eigvals, eigvecs = torch.linalg.eigh(dm)
        result_list = []
        for batch in range(size[0]):
            mixture = tuple(
                zip(
                    eigvals[batch],
                    [vec.reshape([2] * dims) for vec in eigvecs[batch].T],
                )
            )
            result_list.append(
                tuple([(float(p[0]), p[1]) for p in mixture if p[0] > 1e-10])
            )
        return result_list
    else:

        dims = int(np.log2(np.prod(size)) / 2)

        eigvals, eigvecs = torch.linalg.eigh(dm)
        mixture = tuple(zip(eigvals, [vec.reshape([2] * dims) for vec in eigvecs.T]))
        return tuple([(float(p[0]), p[1]) for p in mixture if p[0] > 1e-10])


def partial_trace_test():
    """
    Test function for performing partial trace on a quantum device.

    This function demonstrates how to use the `partial_trace` function from `torchquantum.functional`
    to perform partial trace on a quantum device.

    The function applies Hadamard gate on the first qubit and a CNOT gate between the first and second qubits.
    Then, it performs partial trace on the first qubit and converts the resulting density matrices into
    mixtures of states.

    Prints the resulting mixture of states.

    Note: This function assumes that you have already imported the necessary modules and functions.

    Returns:
        None
    """
    import torchquantum.functional as tqf

    n_wires = 4
    q_device = tq.QuantumDevice(n_wires=n_wires)

    tqf.hadamard(q_device, wires=0)
    tqf.cnot(q_device, wires=[0, 1])

    dms = partial_trace(q_device, [0])
    dms = matrix_form(dms)
    mixture = dm_to_mixture_of_state(dms)

    print(mixture)


def pauli_string_to_matrix(pauli: str, device=torch.device("cpu")) -> torch.Tensor:
    mat_dict = {
        "paulix": torch.tensor([[0, 1], [1, 0]], dtype=C_DTYPE),
        "pauliy": torch.tensor([[0, -1j], [1j, 0]], dtype=C_DTYPE),
        "pauliz": torch.tensor([[1, 0], [0, -1]], dtype=C_DTYPE),
        "i": torch.tensor([[1, 0], [0, 1]], dtype=C_DTYPE),
    }
    paulix = mat_dict["paulix"]
    pauliy = mat_dict["pauliy"]
    pauliz = mat_dict["pauliz"]
    iden = mat_dict["i"]
    pauli_dict = {"X": paulix, "Y": pauliy, "Z": pauliz, "I": iden}

    pauli = pauli.upper()

    matrix = pauli_dict[pauli[0]].to(device)

    for op in pauli[1:]:
        matrix = torch.kron(matrix, pauli_dict[op].to(device))
    return matrix


if __name__ == "__main__":
    build_module_description_test()
    switch_little_big_endian_matrix_test()
    switch_little_big_endian_state_test()


def parameter_shift_gradient(
    model, input_data, expectation_operator, shift_rate=np.pi * 0.5, shots=1024
):
    """
    This function calculates the gradient of a parametrized circuit using the parameter shift rule to be fed into
    a classical optimizer, its formula is given by
    gradient for the ith parameter =( expectation_value(the_ith_parameter + shift_rate)-expectation_value(the_ith_parameter - shift_rate) ) *0.5
    Args:
        model(tq.QuantumModule): the model that you want to use, which includes the quantum device and the parameters
        input(torch.tensor): the input data that you are using
        expectation_operator(str): the observable that you want to calculate the expectation value of, usually the Z operator
        (i.e 'ZZZ' for 3 qubits or 3 wires)
        shift_rate(float , optional): the rate that you would like to shift the parameter with at every iteration, by default pi*0.5
        shots(int , optional): the number of shots to use per parameter ,(for 10 parameters and 1024 shots = 10240 shots in total)
        by default = 1024.
    Returns:
        torch.tensor : An array of the gradients of all the parameters in the circuit.
    """
    par_num = []
    for (
        p
    ) in (
        model.parameters()
    ):  # since the model.parameters() Returns an iterator over module parameters,to get the number of parameter i have to iterate over all of them
        par_num.append(p)
    gradient_of_par = torch.zeros(len(par_num))

    def clone_model(
        model_to_clone,
    ):  # i have to note:this clone_model function was made with GPT
        cloned_model = type(model_to_clone)()  # Create a new instance of the same class
        cloned_model.load_state_dict(
            model_to_clone.state_dict()
        )  # Copy the state dictionary
        return cloned_model

    # Clone the models
    model_plus_shift = clone_model(model)
    model_minus_shift = clone_model(model)
    state_dict_plus_shift = model_plus_shift.state_dict()
    state_dict_minus_shift = model_minus_shift.state_dict()
    #####################

    for idx, key in enumerate(state_dict_plus_shift):
        if idx < 2:  # Skip the first two keys because they are not parameters
            continue
        state_dict_plus_shift[key] += shift_rate
        state_dict_minus_shift[key] -= shift_rate

        model_plus_shift.load_state_dict(state_dict_plus_shift)
        model_minus_shift.load_state_dict(state_dict_minus_shift)

        model_plus_shift.forward(input_data)
        model_minus_shift.forward(input_data)

        state_dict_plus_shift = model_plus_shift.state_dict()
        state_dict_minus_shift = model_minus_shift.state_dict()

        expectation_plus_shift = tq.expval_joint_sampling(
            model_plus_shift.q_device, observable=expectation_operator, n_shots=shots
        )
        expectation_minus_shift = tq.expval_joint_sampling(
            model_minus_shift.q_device, observable=expectation_operator, n_shots=shots
        )

        state_dict_plus_shift[key] -= shift_rate
        state_dict_minus_shift[key] += shift_rate

        gradient_of_par[idx - 2] = (
            expectation_plus_shift - expectation_minus_shift
        ) * 0.5
    return gradient_of_par
