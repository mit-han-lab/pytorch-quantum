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

from typing import Iterable, Optional

import numpy as np
import qiskit
import qiskit.circuit.library.standard_gates as qiskit_gate
import symengine
import sympy
import torch
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import CircuitInstruction, Parameter
from qiskit.circuit.parameter import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit_aer import AerSimulator
from torchpack.utils.logging import logger

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import mat_dict
from torchquantum.util import (
    find_global_phase,
    switch_little_big_endian_matrix,
    switch_little_big_endian_state,
)

__all__ = [
    "tq2qiskit",
    "tq2qiskit_parameterized",
    "qiskit2tq",
    "qiskit2tq_op_history",
    "qiskit2tq_Operator",
    "tq2qiskit_measurement",
    "tq2qiskit_expand_params",
    "qiskit_assemble_circs",
    "tq2qiskit_initialize",
    "append_parameterized_gate",
    "append_fixed_gate",
    "op_history2qiskit",
    "op_history2qiskit_expand_params",
    "op_history2qasm",
]


def qiskit2tq_op_history(circ):
    if getattr(circ, "_layout", None) is not None:
        try:
            p2v_orig = circ._layout.final_layout.get_physical_bits().copy()
        except AttributeError:
            p2v_orig = circ._layout.get_physical_bits().copy()
        p2v = {}
        for p, v in p2v_orig.items():
            if v.register.name == "q":
                p2v[p] = v.index
            else:
                p2v[p] = f"{v.register.name}.{v.index}"
    else:
        p2v = {}
        for p in range(circ.num_qubits):
            p2v[p] = p

    ops = []
    for gate in circ.data:
        op_name = gate[0].name
        wires = [circ.find_bit(qb).index for qb in gate.qubits]
        wires = [p2v[wire] for wire in wires]
        # sometimes the gate.params is ParameterExpression class
        init_params = (
            list(map(float, gate[0].params)) if len(gate[0].params) > 0 else None
        )
        print(
            op_name,
        )

        if op_name in [
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "sx",
            "cx",
            "cz",
            "cy",
            "swap",
            "cswap",
            "ccx",
        ]:
            ops.append(
                {
                    "name": op_name,  # type: ignore
                    "wires": np.array(wires),
                    "params": None,
                    "inverse": False,
                    "trainable": False,
                }
            )
        elif op_name in [
            "rx",
            "ry",
            "rz",
            "rxx",
            "xx",
            "ryy",
            "yy",
            "rzz",
            "zz",
            "rzx",
            "zx",
            "p",
            "cp",
            "crx",
            "cry",
            "crz",
            "u1",
            "cu1",
            "u2",
            "u3",
            "cu3",
            "u",
            "cu",
        ]:
            ops.append(
                {
                    "name": op_name,  # type: ignore
                    "wires": np.array(wires),
                    "params": init_params,
                    "inverse": False,
                    "trainable": True,
                }
            )
        elif op_name in ["barrier", "measure"]:
            continue
        else:
            raise NotImplementedError(
                f"{op_name} conversion to tq is currently not supported."
            )
    return ops


def qiskit_assemble_circs(encoders, fixed_layer, measurement):
    circs_all = []
    n_qubits = len(fixed_layer.qubits)
    for encoder in encoders:
        if len(encoder.cregs) == 0:
            cregs = ClassicalRegister(n_qubits, "c")
            encoder.add_register(cregs)
        if len(fixed_layer.cregs) == 0:
            cregs = ClassicalRegister(n_qubits, "c")
            fixed_layer.add_register(cregs)
        circ = encoder.compose(fixed_layer).compose(measurement)
        circs_all.append(circ)

    return circs_all


def append_parameterized_gate(func, circ, input_idx, params, wires):
    if func == "rx":
        circ.rx(theta=params[input_idx[0]], qubit=wires[0])
    elif func == "ry":
        circ.ry(theta=params[input_idx[0]], qubit=wires[0])
    elif func == "rz":
        circ.rz(phi=params[input_idx[0]], qubit=wires[0])
    elif func == "rxx":
        circ.rxx(theta=params[input_idx[0]], qubit1=wires[0], qubit2=wires[1])
    elif func == "ryy":
        circ.ryy(theta=params[input_idx[0]], qubit1=wires[0], qubit2=wires[1])
    elif func == "rzz":
        circ.rzz(theta=params[input_idx[0]], qubit1=wires[0], qubit2=wires[1])
    elif func == "rzx":
        circ.rzx(theta=params[input_idx[0]], qubit1=wires[0], qubit2=wires[1])
    elif func == "phaseshift":
        circ.p(theta=params[input_idx[0]], qubit=wires[0])
    elif func == "crx":
        circ.crx(
            theta=params[input_idx[0]], control_qubit=wires[0], target_qubit=wires[1]
        )
    elif func == "cry":
        circ.cry(
            theta=params[input_idx[0]], control_qubit=wires[0], target_qubit=wires[1]
        )
    elif func == "crz":
        circ.cry(
            theta=params[input_idx[0]], control_qubit=wires[0], target_qubit=wires[1]
        )
    elif func == "u1":
        circ.p(theta=params[input_idx[0]], qubit=wires[0])
    elif func == "cu1":
        circ.cu1(
            theta=params[input_idx[0]], control_qubit=wires[0], target_qubit=wires[1]
        )
    elif func == "u2":
        from qiskit.circuit.library import U2Gate

        circ.append(
            U2Gate(phi=params[input_idx[0]], lam=params[input_idx[1]]), wires, []
        )
        # circ.u2(phi=params[input_idx[0]], lam=params[input_idx[1]], qubit=wires[0])
    elif func == "u3":
        circ.u(
            theta=params[input_idx[0]],
            phi=params[input_idx[1]],
            lam=params[input_idx[2]],
            qubit=wires[0],
        )
    elif func == "cu3":
        circ.cu3(
            theta=params[input_idx[0]],
            phi=params[input_idx[1]],
            lam=params[input_idx[2]],
            qubit=wires[0],
        )
    else:
        raise NotImplementedError(
            f"{func} cannot be converted to parameterized Qiskit QuantumCircuit"
        )


def append_fixed_gate(circ, func, params, wires, inverse):
    if not isinstance(wires, Iterable):
        wires = [wires]
    # if not isinstance(params, Iterable):
    # params = [params]

    if func in ["hadamard", "h"]:
        circ.h(*wires)
    elif func in ["shadamard", "sh"]:
        circ.ry(np.pi / 4, *wires)
    elif func in ["paulix", "x"]:
        circ.x(*wires)
    elif func in ["pauliy", "y"]:
        circ.y(*wires)
    elif func in ["pauliz", "z"]:
        circ.z(*wires)
    elif func == "s":
        circ.s(*wires)
    elif func == "t":
        circ.t(*wires)
    elif func == "sx":
        circ.sx(*wires)
    elif func in ["cnot", "cx"]:
        circ.cx(*wires)
    elif func == "cz":
        circ.cz(*wires)
    elif func == "cy":
        circ.cy(*wires)
    elif func == "rx":
        circ.rx(params, *wires)
    elif func == "ry":
        circ.ry(params, *wires)
    elif func == "rz":
        circ.rz(params, *wires)
    elif func in ["rxx", "xx"]:
        circ.rxx(params, *wires)
    elif func in ["ryy", "yy"]:
        circ.ryy(params, *wires)
    elif func in ["rzz", "zz"]:
        circ.rzz(params, *wires)
    elif func == ["rzx", "zx"]:
        circ.rzx(params, *wires)
    elif func == "swap":
        circ.swap(*wires)
    elif func == "sswap":
        # square root of swap
        from torchquantum.plugin.qiskit.qiskit_unitary_gate import UnitaryGate

        mat = mat_dict["sswap"].detach().cpu().numpy()
        mat = switch_little_big_endian_matrix(mat)
        circ.append(UnitaryGate(mat), *wires, [])
    elif func == "cswap":
        circ.cswap(*wires)
    elif func in ["toffoli", "ccx"]:
        circ.ccx(*wires)
    elif func in ["phaseshift", "p"]:
        circ.p(params, *wires)
    elif func == "crx":
        circ.crx(params, *wires)
    elif func == "cry":
        circ.cry(params, *wires)
    elif func == "crz":
        circ.crz(params, *wires)
    elif func == "u1":
        circ.u1(params, *wires)
    elif func in ["cu1", "cp", "cr", "cphase"]:
        circ.cu1(params, *wires)
    elif func == "u2":
        from qiskit.circuit.library import U2Gate

        circ.append(U2Gate(phi=params[0], lam=params[1]), wires, [])
        # circ.u2(*list(params), *wires)
    elif func == "u3":
        circ.u(*list(params), *wires)
    elif func == "cu3":
        circ.cu3(*list(params), *wires)
    elif (
        func == "qubitunitary"
        or func == "qubitunitaryfast"
        or func == "qubitunitarystrict"
    ):
        from torchquantum.plugin.qiskit.qiskit_unitary_gate import UnitaryGate

        mat = np.array(params)
        mat = switch_little_big_endian_matrix(mat)
        circ.append(UnitaryGate(mat), wires, [])
    elif func == "multicnot":
        circ.mcx(wires[:-1], wires[-1])  # type: ignore
    elif func == "multixcnot":
        controls = wires[:-1]  # type: ignore
        target = wires[-1]  # type: ignore
        num_ctrl_qubits = len(controls)

        gate = qiskit_gate.MCXGrayCode(
            num_ctrl_qubits, ctrl_state="0" * num_ctrl_qubits
        )
        circ.append(gate, controls + [target], [])
    else:
        logger.exception(f"{func} cannot be converted to Qiskit.")
        raise NotImplementedError(func)

    if inverse:
        data = list(circ.data[-1])
        del circ.data[-1]
        circ.data.append(tuple([data[0].inverse()] + data[1:]))
    return circ


def tq2qiskit_initialize(q_device: tq.QuantumDevice, all_states):
    """Call the qiskit initialize function and encoder the current quantum state
     using initialize and return circuits

    Args:
        q_device:

    Returns:

    """
    bsz = all_states.shape[0]
    circ_all = []
    for k in range(bsz):
        circ = QuantumCircuit(q_device.n_wires)
        state = all_states[k]
        state = np.complex128(state)
        state = state / (np.absolute(state) ** 2).sum()
        state = switch_little_big_endian_state(state)
        qiskit.circuit.library.data_preparation.state_preparation._EPS = 1e-7
        circ.initialize(state, circ.qubits)
        circ_all.append(circ)
    return circ_all


def tq2qiskit_expand_params(
    q_device: tq.QuantumDevice,
    x: torch.Tensor,
    func_list,
):
    """Expand the input classical values to fixed rotation angles in gates. No
        Qiskit.circuit.Parameter is used. All rotations are hard coded in
        gates. This will solve the issue of qiskit bugs.

    Args:
        q_device (tq.QuantumDevice): Quantum device
        x (torch.Tensor): Input classical values waited to be embedded in the
            circuits.
        func_list (List): Information about how the classical values should be
            encoded.

    Returns:
        circ_all (List[Qiskit.QiskitCircuit]): expand the parameters into encodings
            and return the hard coded circuits.
    """
    # the bsz determines how many QuantumCircuit will be constructed
    bsz = x.shape[0]
    circ_all = []
    for k in range(bsz):
        classical_values = x[k].detach().cpu().numpy()
        circ = QuantumCircuit(q_device.n_wires)
        for info in func_list:
            input_idx = info["input_idx"]
            func = info["func"]
            wires = info["wires"]
            append_parameterized_gate(func, circ, input_idx, classical_values, wires)

        circ_all.append(circ)

    return circ_all


# construct a QuantumCircuit object according to the tq module
def tq2qiskit(
    q_device: tq.QuantumDevice,
    m: tq.QuantumModule,
    x=None,
    draw=False,
    remove_ops=False,
    remove_ops_thres=1e-4,
):
    # build the module list without changing the statevector of QuantumDevice
    original_wires_per_block = m.wires_per_block
    original_static_mode = m.static_mode
    m.static_off()
    m.static_on(wires_per_block=q_device.n_wires)
    m.is_graph_top = False

    # forward to register all modules and parameters
    if x is None:
        m.forward(q_device)
    else:
        m.forward(q_device, x)

    m.is_graph_top = True
    m.graph.build_flat_module_list()

    module_list = m.graph.flat_module_list
    m.static_off()

    if original_static_mode:
        m.static_on(wires_per_block=original_wires_per_block)

    # circ = QuantumCircuit(q_device.n_wires, q_device.n_wires)
    circ = QuantumCircuit(q_device.n_wires)

    for module in module_list:
        try:
            # no params in module or batch size == 1, because we will
            # generate only one qiskit QuantumCircuit
            assert module.params is None or module.params.shape[0] == 1
        except AssertionError:
            logger.exception("Cannot convert batch model tq module")

    n_removed_ops = 0

    for module in module_list:
        if remove_ops:
            if module.name in [
                "RX",
                "RY",
                "RZ",
                "RXX",
                "RYY",
                "RZZ",
                "RZX",
                "PhaseShift",
                "CRX",
                "CRY",
                "CRZ",
                "U1",
                "CU1",
            ]:
                param = module.params[0][0].item()
                param = param % (4 * np.pi)
                param = param - 4 * np.pi if param > 2 * np.pi else param
                if abs(param) < remove_ops_thres:
                    n_removed_ops += 1
                    continue

            elif module.name in ["U2", "U3", "CU3"]:
                param = module.params[0].data.cpu().numpy()
                param = param % (4 * np.pi)
                param[param > 2 * np.pi] -= 4 * np.pi
                if all(abs(param) < remove_ops_thres):
                    n_removed_ops += 1
                    continue

        if module.name == "Hadamard":
            circ.h(*module.wires)
        elif module.name == "SHadamard":
            circ.ry(np.pi / 4, *module.wires)
        elif module.name == "PauliX":
            circ.x(*module.wires)
        elif module.name == "PauliY":
            circ.y(*module.wires)
        elif module.name == "PauliZ":
            circ.z(*module.wires)
        elif module.name == "S":
            circ.s(*module.wires)
        elif module.name == "T":
            circ.t(*module.wires)
        elif module.name == "SX":
            circ.sx(*module.wires)
        elif module.name == "CNOT":
            circ.cx(*module.wires)
        elif module.name == "CZ":
            circ.cz(*module.wires)
        elif module.name == "CY":
            circ.cy(*module.wires)
        elif module.name == "RX":
            circ.rx(module.params[0][0].item(), *module.wires)
        elif module.name == "RY":
            circ.ry(module.params[0][0].item(), *module.wires)
        elif module.name == "RZ":
            circ.rz(module.params[0][0].item(), *module.wires)
        elif module.name == "RXX":
            circ.rxx(module.params[0][0].item(), *module.wires)
        elif module.name == "RYY":
            circ.ryy(module.params[0][0].item(), *module.wires)
        elif module.name == "RZZ":
            circ.rzz(module.params[0][0].item(), *module.wires)
        elif module.name == "RZX":
            circ.rzx(module.params[0][0].item(), *module.wires)
        elif module.name == "SWAP":
            circ.swap(*module.wires)
        elif module.name == "SSWAP":
            # square root of swap
            from torchquantum.plugin.qiskit.qiskit_unitary_gate import UnitaryGate

            mat = module.matrix.data.cpu().numpy()
            mat = switch_little_big_endian_matrix(mat)
            circ.append(UnitaryGate(mat), module.wires, [])
        elif module.name == "CSWAP":
            circ.cswap(*module.wires)
        elif module.name == "Toffoli":
            circ.ccx(*module.wires)
        elif module.name == "PhaseShift":
            circ.p(module.params[0][0].item(), *module.wires)
        elif module.name == "CRX":
            circ.crx(module.params[0][0].item(), *module.wires)
        elif module.name == "CRY":
            circ.cry(module.params[0][0].item(), *module.wires)
        elif module.name == "CRZ":
            circ.crz(module.params[0][0].item(), *module.wires)
        elif module.name == "U1":
            circ.u1(module.params[0][0].item(), *module.wires)
        elif module.name == "CU1":
            circ.cu1(module.params[0][0].item(), *module.wires)
        elif module.name == "U2":
            from qiskit.circuit.library import U2Gate

            circ.append(
                U2Gate(
                    phi=module.params[0].data.cpu().numpy()[0],
                    lam=module.params[0].data.cpu().numpy()[0],
                ),
                module.wires,
                [],
            )
            # circ.u2(*list(module.params[0].data.cpu().numpy()), *module.wires)
        elif module.name == "U3":
            circ.u3(*list(module.params[0].data.cpu().numpy()), *module.wires)
        elif module.name == "CU3":
            circ.cu3(*list(module.params[0].data.cpu().numpy()), *module.wires)
        elif (
            module.name == "QubitUnitary"
            or module.name == "QubitUnitaryFast"
            or module.name == "TrainableUnitary"
            or module.name == "TrainableUnitaryStrict"
        ):
            from torchquantum.plugin.qiskit.qiskit_unitary_gate import UnitaryGate

            mat = module.params[0].data.cpu().numpy()
            mat = switch_little_big_endian_matrix(mat)
            circ.append(UnitaryGate(mat), module.wires, [])
        elif module.name == "MultiCNOT":
            circ.mcx(module.wires[:-1], module.wires[-1])
        elif module.name == "MultiXCNOT":
            controls = module.wires[:-1]
            target = module.wires[-1]
            num_ctrl_qubits = len(controls)

            gate = qiskit_gate.MCXGrayCode(
                num_ctrl_qubits, ctrl_state="0" * num_ctrl_qubits
            )
            circ.append(gate, controls + [target], [])
        else:
            logger.exception(f"{module.name} cannot be converted to Qiskit.")
            raise NotImplementedError(module.name)

        if module.inverse:
            data = list(circ.data[-1])
            del circ.data[-1]
            circ.data.append(tuple([data[0].inverse()] + data[1:]))
    if draw:
        import matplotlib.pyplot as plt

        circ.draw()
        plt.show()

    if n_removed_ops > 0:
        logger.warning(
            f"Remove {n_removed_ops} operations with small parameter magnitude."
        )

    return circ


def tq2qiskit_measurement(q_device: tq.QuantumDevice, q_layer_measure):
    circ = QuantumCircuit(q_device.n_wires, q_device.n_wires)
    v_c_reg_mapping = q_layer_measure.v_c_reg_mapping

    if v_c_reg_mapping is not None:
        for q_reg, c_reg in v_c_reg_mapping["v2c"].items():
            circ.measure(q_reg, c_reg)
    else:
        circ.measure(list(range(q_device.n_wires)), list(range(q_device.n_wires)))
    return circ


def tq2qiskit_parameterized(q_device: tq.QuantumDevice, func_list):
    """
    construct parameterized qiskit QuantumCircuit,
    useful in the classical-quantum encoder
    """
    # circ = QuantumCircuit(q_device.n_wires, q_device.n_wires)
    circ = QuantumCircuit(q_device.n_wires)

    params = {}
    for info in func_list:
        input_idx = info["input_idx"]
        for idx in input_idx:
            param = Parameter(f"param{idx}")
            params[idx] = param

        func = info["func"]
        wires = info["wires"]
        wires = wires if isinstance(wires, Iterable) else [wires]

        append_parameterized_gate(func, circ, input_idx, params, wires)

    return circ, params


def op_history2qiskit(n_wires, op_history):
    """convert a tq op_history to a qiskit QuantumCircuit
    Args:
        n_wires: number of wires
        op_history: a list of tq QuantumModule objects
    Returns:
        a qiskit QuantumCircuit object
    """
    circ = QuantumCircuit(n_wires)
    for op in op_history:
        append_fixed_gate(circ, op["name"], op["params"], op["wires"], op["inverse"])
    return circ


def op_history2qasm(n_wires, op_history):
    """convert a tq op_history to a qasm string
    Args:
        n_wires: number of wires
        op_history: a list of tq QuantumModule objects
    Returns:
        a qasm string
    """
    circ = op_history2qiskit(n_wires, op_history)
    return circ.qasm()


def op_history2qiskit_expand_params(n_wires, op_history, bsz):
    """convert a tq op_history to a qiskit QuantumCircuit
    Args:
        n_wires: number of wires
        op_history: a list of tq operations
        bsz: batch size
    Returns:
        a qiskit QuantumCircuit object
    """
    assert bsz == len(op_history[0]["params"])
    circs_all = []
    for i in range(bsz):
        circ = QuantumCircuit(n_wires)
        for op in op_history:
            if "params" in op.keys() and op["params"] is not None:
                param = op["params"][i]
            else:
                param = None

            append_fixed_gate(circ, op["name"], param, op["wires"], op["inverse"])

        circs_all.append(circ)

    return circs_all


# construct a tq QuantumModule object according to the qiskit QuantumCircuit
# object
def qiskit2tq_Operator(circ: QuantumCircuit, initial_parameters=None):
    if getattr(circ, "_layout", None) is not None:
        try:
            p2v_orig = circ._layout.final_layout.get_physical_bits().copy()
        except AttributeError:
            try:
                p2v_orig = circ._layout.get_physical_bits().copy()
            except AttributeError:
                p2v_orig = circ._layout.initial_layout.get_physical_bits().copy()
        p2v = {}
        for p, v in p2v_orig.items():
            if v.register.name == "q":
                p2v[p] = v.index
            else:
                p2v[p] = f"{v.register.name}.{v.index}"
    else:
        p2v = {}
        for p in range(circ.num_qubits):
            p2v[p] = p

    if initial_parameters is None:
        initial_parameters = torch.nn.init.uniform_(
            torch.ones(len(circ.parameters)), -np.pi, np.pi
        )

    param_to_index = {}
    for i, param in enumerate(circ.parameters):
        param_to_index[param] = i

    ops = []
    for gate in circ.data:
        op_name = gate[0].name
        wires = [circ.find_bit(qb).index for qb in gate.qubits]
        wires = [p2v[wire] for wire in wires]

        init_params = qiskit2tq_translate_qiskit_params(
            gate, initial_parameters, param_to_index
        )

        if op_name in [
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "sx",
            "cx",
            "cz",
            "cy",
            "swap",
            "cswap",
            "ccx",
        ]:
            ops.append(tq.op_name_dict[op_name](wires=wires))
        elif op_name in [
            "rx",
            "ry",
            "rz",
            "rxx",
            "xx",
            "ryy",
            "yy",
            "rzz",
            "zz",
            "rzx",
            "zx",
            "p",
            "cp",
            "crx",
            "cry",
            "crz",
            "u1",
            "cu1",
            "u2",
            "u3",
            "cu3",
            "u",
            "cu",
        ]:
            ops.append(
                tq.op_name_dict[op_name](
                    has_params=True,
                    trainable=True,
                    init_params=init_params,
                    wires=wires,
                )
            )
        elif op_name in ["barrier", "measure"]:
            continue
        else:
            raise NotImplementedError(
                f"{op_name} conversion to tq is currently not supported."
            )

    return ops


def qiskit2tq_translate_qiskit_params(
    circuit_instruction: CircuitInstruction, initial_parameters, param_to_index
):
    parameters = []
    for p in circuit_instruction.operation.params:
        if isinstance(p, Parameter) or isinstance(p, ParameterVectorElement):
            parameters.append(initial_parameters[param_to_index[p]])
        elif isinstance(p, ParameterExpression):
            if len(p.parameters) == 0:
                parameters.append(float(p))
                continue

            expr = p.sympify().simplify()
            if isinstance(expr, symengine.Expr):  # qiskit uses symengine if available
                expr = expr._sympy_()  # sympy.Expr

            for free_symbol in expr.free_symbols:
                # replace names: theta[0] -> theta_0
                # ParameterVector creates symbols with brackets like theta[0]
                # but sympy.lambdify does not allow brackets in symbol names
                free_symbol.name = free_symbol.name.replace("[", "_").replace("]", "")

            parameter_list = list(p.parameters)
            sympy_symbols = [param._symbol_expr for param in parameter_list]
            # replace names again: theta[0] -> theta_0
            sympy_symbols = [
                sympy.Symbol(str(symbol).replace("[", "_").replace("]", ""))
                for symbol in sympy_symbols
            ]
            lam_f = sympy.lambdify(sympy_symbols, expr, modules="math")
            parameters.append(
                lam_f(
                    *[
                        initial_parameters[param_to_index[param]]
                        for param in parameter_list
                    ]
                )
            )
        else:  # non-parameterized gate
            parameters.append(p)
    return parameters


def qiskit2tq(
    circ: QuantumCircuit, initial_parameters: Optional[list[torch.nn.Parameter]] = None
):
    ops = qiskit2tq_Operator(circ, initial_parameters)
    return tq.QuantumModuleFromOps(ops)


def test_qiskit2tq():
    import pdb

    pdb.set_trace()
    n_wires = 4
    q_dev = tq.QuantumDevice(n_wires=n_wires)

    circ = QuantumCircuit(n_wires, n_wires)
    circ.h(0)
    circ.h(0)

    circ.rx(theta=0.1, qubit=2)
    circ.ry(theta=0.2, qubit=3)
    circ.rz(phi=0.3, qubit=2)
    circ.sx(2)
    circ.sx(3)

    circ.crx(theta=0.4, control_qubit=0, target_qubit=1)
    circ.cx(control_qubit=2, target_qubit=1)

    circ.u3(theta=-0.1, phi=-0.2, lam=-0.4, qubit=3)
    circ.cx(control_qubit=3, target_qubit=0)
    circ.cx(control_qubit=0, target_qubit=2)
    circ.x(2)
    circ.x(3)
    circ.u2(phi=-0.2, lam=-0.9, qubit=3)
    circ.x(0)

    m = qiskit2tq(circ)

    backend = AerSimulator(method="unitary")
    circ = transpile(circ, backend)
    circ.save_unitary()
    result = backend.run(circ).result()
    unitary_qiskit = result.get_unitary(circ)

    unitary_tq = m.get_unitary(q_dev)
    unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

    circ_from_m = tq2qiskit(q_dev, m)
    assert circ_from_m == circ

    phase = find_global_phase(unitary_tq, unitary_qiskit, 1e-4)

    assert np.allclose(unitary_tq * phase, unitary_qiskit, atol=1e-6)


class T00(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.gate = tq.Hadamard()

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        self.gate(q_device, wires=0)


class TQAll(tq.QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.submodules = tq.QuantumModuleList()
        self.n_gate = n_gate
        self.t00 = T00()
        for k in range(self.n_gate):
            self.submodules.append(op())

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for k in range(self.n_gate - 1):
            self.submodules[k](q_device, wires=[k, k + 1])
        self.submodules[-1](q_device, wires=[self.n_gate - 1, 0])
        self.t00(q_device)


class TestModule(tq.QuantumModule):
    def __init__(self, q_device: tq.QuantumDevice = None):
        super().__init__()
        self.q_device = q_device
        self.n_gate = 10
        self.gate0 = tq.CNOT()
        # self.gate1 = tq.CNOT()
        self.submodules = tq.QuantumModuleList()
        self.q_layer0 = TQAll(self.n_gate, tq.CNOT)
        for k in range(self.n_gate):
            self.submodules.append(tq.RY())
        # for k in range(self.n_gate):
        #     self.submodules.append(tq.CNOT())
        # self.gate0 = tq.RY(has_params=False, trainable=False)
        # self.gate1 = tq.RX(has_params=False, trainable=False)
        # self.gate2 = tq.RZ(has_params=False, trainable=False)
        self.gate1 = tq.RX(has_params=True, trainable=True)
        self.gate2 = tq.RZ(has_params=True, trainable=True)
        self.gate3 = tq.RY(has_params=True, trainable=True)
        # self.gate3 = tq.CNOT()
        self.gate4 = tq.RX(has_params=True, trainable=True)
        self.gate5 = tq.RZ(has_params=True, trainable=True)
        self.gate6 = tq.RY(has_params=True, trainable=True)
        self.gate7 = tq.RX()
        self.gate8 = tq.U2(has_params=True, trainable=True)
        self.gate9 = tq.TrainableUnitary(has_params=True, trainable=True, n_wires=3)
        self.gate10 = tq.MultiXCNOT(n_wires=5)
        self.gate11 = tq.MultiCNOT(n_wires=3)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.gate1(q_device, wires=3)
        self.gate2(q_device, wires=4)
        self.gate3(q_device, wires=3)
        self.gate4(q_device, wires=3)
        self.gate5(q_device, wires=3)
        self.gate6(q_device, wires=3, inverse=True)
        self.gate7(q_device, wires=4, params=x, inverse=True)
        self.gate8(q_device, wires=2)
        self.gate9(q_device, wires=[2, 3, 4])

        self.q_layer0(q_device)
        tqf.qubitunitary(
            self.q_device,
            wires=[1, 2],
            params=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            static=self.static_mode,
            parent_graph=self.graph,
        )
        tqf.qubitunitary(
            self.q_device,
            wires=[1, 2],
            params=[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            static=self.static_mode,
            parent_graph=self.graph,
        )
        self.gate10(q_device, wires=[4, 5, 6, 7, 1])
        self.gate11(q_device, wires=[2, 1, 9])

        # self.gate0(q_device, wires=[7, 4])
        # self.gate1(q_device, wires=[3, 9])

        # self.gate0(q_device, wires=1, params=x[:, 2])
        # self.gate1(q_device, wires=5, params=x[:, 0])
        # self.gate2(q_device, wires=7, params=x[:, 6])

        # self.gate2(q_device, wires=5)
        # self.gate3(q_device, wires=[3, 5])
        # self.gate4(q_device, wires=5)


class TestModuleParameterized(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        # self.func_list = [
        #     {'input_idx': [0], 'func': 'ry', 'wires': [0]},
        #     {'input_idx': [1], 'func': 'ry', 'wires': [1]},
        #     {'input_idx': [2], 'func': 'ry', 'wires': [2]},
        #     {'input_idx': [3], 'func': 'ry', 'wires': [3]},
        #     {'input_idx': [4], 'func': 'rz', 'wires': [0]},
        #     {'input_idx': [5], 'func': 'rz', 'wires': [1]},
        #     {'input_idx': [6], 'func': 'rz', 'wires': [2]},
        #     {'input_idx': [7], 'func': 'rz', 'wires': [3]},
        #     {'input_idx': [8], 'func': 'rx', 'wires': [0]},
        #     {'input_idx': [9], 'func': 'rx', 'wires': [1]},
        #     {'input_idx': [10], 'func': 'rx', 'wires': [2]},
        #     {'input_idx': [11], 'func': 'rx', 'wires': [3]},
        #     {'input_idx': [12], 'func': 'ry', 'wires': [0]},
        #     {'input_idx': [13], 'func': 'ry', 'wires': [1]},
        #     {'input_idx': [14], 'func': 'ry', 'wires': [2]},
        #     {'input_idx': [15], 'func': 'ry', 'wires': [3]}
        # ]
        self.func_list = [
            {"input_idx": [6, 5, 4], "func": "u3", "wires": [1]},
            {"input_idx": [7], "func": "u1", "wires": [1]},
            {"input_idx": [0, 1, 2], "func": "u3", "wires": [0]},
            {"input_idx": [3], "func": "u1", "wires": [0]},
            {"input_idx": [8, 9, 10], "func": "u3", "wires": [2]},
            {"input_idx": [11], "func": "u1", "wires": [2]},
            {"input_idx": [12, 13, 14], "func": "u3", "wires": [3]},
            {"input_idx": [15], "func": "u1", "wires": [3]},
        ]
        self.encoder = tq.GeneralEncoder(self.func_list)

    @tq.static_support
    def forward(self, q_device, x):
        self.q_device = q_device
        self.encoder(q_device, x)


def test_tq2qiskit():
    import pdb

    pdb.set_trace()
    inputs = torch.ones((1, 1)) * 0.42
    q_dev = tq.QuantumDevice(n_wires=10)
    test_module = TestModule(q_dev)

    circuit = tq2qiskit(test_module, inputs)

    backend = AerSimulator(method="unitary")
    circuit = transpile(circuit, backend)
    circuit.save_unitary()
    result = backend.run(circuit).result()
    unitary_qiskit = result.get_unitary(circuit)

    unitary_tq = test_module.get_unitary(q_dev, inputs)
    unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

    print(unitary_qiskit)
    print(unitary_tq)
    assert np.allclose(unitary_qiskit, unitary_tq, atol=1e-6)


def test_tq2qiskit_parameterized():
    import pdb

    pdb.set_trace()
    inputs = torch.randn((1, 16))
    q_dev = tq.QuantumDevice(n_wires=4)
    test_module = TestModuleParameterized()
    test_module(q_dev, inputs)
    unitary_tq = test_module.get_unitary(q_dev, inputs)
    unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

    circuit, params = tq2qiskit_parameterized(q_dev, test_module.encoder.func_list)
    binds = {}
    for k, x in enumerate(inputs[0]):
        binds[params[k]] = x.item()

    backend = AerSimulator(method="unitary")
    circuit = transpile(circuit, backend)
    circuit.save_unitary()
    result = backend.run(circuit, parameter_binds=[binds]).result()
    unitary_qiskit = result.get_unitary(circuit)

    # print(unitary_qiskit)
    # print(unitary_tq)
    assert np.allclose(unitary_qiskit, unitary_tq, atol=1e-6)


if __name__ == "__main__":
    # test_tq2qiskit_parameterized()
    test_qiskit2tq()
