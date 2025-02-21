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

import argparse
import pdb
import torch
import torchquantum as tq
import numpy as np

from torchpack.utils.logging import logger
from torchquantum.operator import op_name_dict
from torchquantum.functional import func_name_dict
from torchquantum.macro import F_DTYPE
from torchquantum.plugin.qiskit import (
    QISKIT_INCOMPATIBLE_FUNC_NAMES,
    QISKIT_INCOMPATIBLE_OPS,
)


class QLayer(tq.QuantumModule):
    def __init__(
        self,
        n_wires,
        wires,
        n_ops_rd,
        n_ops_cin,
        n_funcs,
        qiskit_compatible=False,
    ):
        super().__init__()
        self.n_wires = n_wires
        self.wires = wires
        self.n_ops_rd = n_ops_rd
        self.n_ops_cin = n_ops_cin
        self.n_funcs = n_funcs
        self.rd_layer = None
        if self.n_ops_rd > 0:
            self.rd_layer = tq.RandomLayerAllTypes(
                n_ops=n_ops_rd, wires=wires, qiskit_compatible=qiskit_compatible
            )

        self.cin_op_types = [
            tq.RX,
            tq.RY,
            tq.RZ,
            tq.PhaseShift,
            tq.Rot,
            tq.MultiRZ,
            tq.CRX,
            tq.CRY,
            tq.CRZ,
            tq.CRot,
            tq.U1,
            tq.U2,
            tq.U3,
        ]
        self.funcs = [
            "hadamard",
            "paulix",
            "pauliy",
            "pauliz",
            "s",
            "t",
            "sx",
            "cnot",
            "cz",
            "cy",
            "rx",
            "ry",
            "rz",
            "swap",
            "cswap",
            "toffoli",
            "phaseshift",
            "rot",
            "multirz",
            "crx",
            "cry",
            "crz",
            "crot",
            "u1",
            "u2",
            "u3",
            "qubitunitary",
            "qubitunitaryfast",
            "multicnot",
            "multixcnot",
        ]

        if qiskit_compatible:
            for op in QISKIT_INCOMPATIBLE_OPS:
                self.cin_op_types.remove(op)
            for func in QISKIT_INCOMPATIBLE_FUNC_NAMES:
                self.funcs.remove(func)

        self.x_idx = 0
        self.cin_op_list = tq.QuantumModuleList()
        self.rand_mat = np.random.randn(2**self.n_wires, 2**self.n_wires)
        self.build_random_cin_layer()

        self.func_list = []
        self.func_wires_list = []
        self.build_random_funcs()

        self.cin_op_inverse = np.random.randint(2, size=n_ops_cin)
        self.func_inverse = np.random.randint(2, size=n_funcs)

    def build_random_cin_layer(self):
        cnt = 0
        while cnt < self.n_ops_cin:
            op = np.random.choice(self.cin_op_types)
            n_op_wires = op.num_wires
            if n_op_wires > self.n_wires:
                continue
            if n_op_wires == -1:
                is_AnyWire = True
                n_op_wires = self.n_wires
            else:
                is_AnyWire = False

            op_wires = list(
                np.random.choice(self.wires, size=n_op_wires, replace=False)
            )

            if is_AnyWire:
                operation = op(n_wires=n_op_wires, wires=op_wires)
            else:
                operation = op(wires=op_wires)

            self.cin_op_list.append(operation)
            cnt += 1

    def build_random_funcs(self):
        cnt = 0
        while cnt < self.n_funcs:
            func = np.random.choice(self.funcs)
            # print(f"Selected function: {func}")

            """
            ORIGINAL: n_func_wires = op_name_dict[func]().num_wires
            Changed to avoid initialization error with QubitUnitaryFast which requires 
            parameters during instantiation. Instead, we access num_wires directly 
            from the class since it's a class attribute.
            """

            op_class = op_name_dict[func]
            # print(f"Operator class: {op_class}")
            # print(f"Number of wires: {op_class.num_wires}")
            n_func_wires = op_class.num_wires
            
            if n_func_wires > self.n_wires:
                continue
            cnt += 1

            if n_func_wires == -1:
                n_func_wires = self.n_wires

            func_wires = list(
                np.random.choice(self.wires, size=n_func_wires, replace=False)
            )
            self.func_list.append(func)
            self.func_wires_list.append(func_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device

        # test random generated parameterized layers
        if self.n_ops_rd > 0:
            self.rd_layer(q_device)

        # test layer with classical inputs
        for op, is_inverse in zip(self.cin_op_list, self.cin_op_inverse):
            n_op_params = op.num_params
            if n_op_params == -1:
                n_op_params = 2
            params = x[:, self.x_idx : self.x_idx + n_op_params]
            self.x_idx += n_op_params
            op(self.q_device, params=params, inverse=is_inverse)

        # test functional layers
        for func, func_wires, is_inverse in zip(
            self.func_list, self.func_wires_list, self.func_inverse
        ):
            n_func_wires = len(func_wires)
            op_class = op_name_dict[func]
            # n_func_params = op_name_dict[func]().num_params
            n_func_params = op_class.num_params
            if n_func_params == 0:
                if func in ["multicnot", "multixcnot"]:
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        n_wires=n_func_wires,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )
                else:
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )
            elif n_func_params == -1:
                # qubitunitary:
                if func in ["qubitunitary", "qubitunitaryfast", "qubitunitarystrict"]:
                    u, s, v = np.linalg.svd(
                        self.rand_mat[: 2**n_func_wires, : 2**n_func_wires]
                    )
                    params = u @ v
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        n_wires=n_func_wires,
                        params=params,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )
                else:
                    raise NotImplementedError
            else:
                params = x[:, self.x_idx : self.x_idx + n_func_params]
                self.x_idx += n_func_params
                if func in ["multirz"]:
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        n_wires=n_func_wires,
                        params=params,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )
                else:
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        params=params,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )

        self.x_idx = 0


def static_mode_dynamic_vs_static_gradients_test():
    bsz = 7

    for n_wires in range(2, 10):
        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_dev0 = tq.QuantumDevice(n_wires=n_wires)
        q_dev1 = tq.QuantumDevice(n_wires=n_wires)
        q_dev0.reset_all_eq_states(bsz)
        q_layer = QLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
        )
        q_layer(q_dev0, x)
        loss = q_dev0.states.abs().sum()
        loss.backward()
        q_layer_dynamic_grads = {}
        for name, params in q_layer.named_parameters():
            q_layer_dynamic_grads[name] = params.grad.data.clone()
            params.grad = None

        for wires_per_block in range(1, n_wires + 1):
            q_dev1.reset_all_eq_states(bsz)
            q_layer.static_off()
            q_layer.static_on(wires_per_block=wires_per_block)
            q_layer(q_dev1, x)
            loss = q_dev1.states.abs().sum()
            loss.backward()
            q_layer_static_grads = {}
            for name, params in q_layer.named_parameters():
                q_layer_static_grads[name] = params.grad.data.clone()
                params.grad = None

            name = None
            try:
                for name, _ in q_layer.named_parameters():
                    diff = q_layer_dynamic_grads[name] - q_layer_static_grads[name]
                    assert torch.allclose(
                        q_layer_dynamic_grads[name],
                        q_layer_static_grads[name],
                        atol=1e-4,
                    )
                    logger.info(f"Diff: {diff}")
                    logger.info(
                        f"PASS: [n_wires]={n_wires}, dynamic VS "
                        f"static run with [wires_per_block]="
                        f"{wires_per_block}. Params: {name}"
                    )
            except AssertionError:
                logger.exception(
                    f"FAIL: [n_wires]={n_wires}, dynamic VS "
                    f"static run with "
                    f"[wires_per_block]={wires_per_block}. "
                    f"Params: {name}"
                )
                raise AssertionError

            q_layer.static_off()


def static_mode_dynamic_vs_static_test():
    bsz = 7

    for n_wires in range(2, 10):
        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_dev0 = tq.QuantumDevice(n_wires=n_wires)
        q_dev1 = tq.QuantumDevice(n_wires=n_wires)
        q_dev0.reset_all_eq_states(bsz)
        q_layer = QLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
        )
        q_layer(q_dev0, x)
        state_dyna = q_dev0.states.clone()

        for wires_per_block in range(1, n_wires + 1):
            q_dev1.reset_all_eq_states(bsz)
            q_layer.static_off()
            q_layer.static_on(wires_per_block=wires_per_block)
            q_layer(q_dev1, x)
            state_static = q_dev1.states.clone()

            try:
                assert torch.allclose(state_dyna, state_static, atol=1e-5)
                logger.info(
                    f"PASS: [n_wires]={n_wires}, dynamic VS static "
                    f"run with [wires_per_block]={wires_per_block}"
                )
            except AssertionError:
                logger.exception(
                    f"FAIL: [n_wires]={n_wires}, dynamic VS "
                    f"static run with "
                    f"[wires_per_block]={wires_per_block}"
                )
                raise AssertionError

            q_layer.static_off()


def static_mode_dynamic_vs_static_vs_get_unitary_test():
    for n_wires in range(2, 11):
        q_dev0 = tq.QuantumDevice(n_wires=n_wires)
        q_dev1 = tq.QuantumDevice(n_wires=n_wires)
        q_dev2 = tq.QuantumDevice(n_wires=n_wires)
        q_dev0.reset_identity_states()
        q_dev1.reset_identity_states()

        q_layer = tq.RandomLayerAllTypes(n_ops=500, wires=list(range(n_wires)))
        q_layer(q_dev0)
        u_dyna = q_dev0.states.clone().reshape(2**n_wires, 2**n_wires).permute(1, 0)

        u_static_get = q_layer.get_unitary(q_dev1)
        try:
            assert torch.allclose(u_dyna, u_static_get, atol=1e-5)
            logger.info(f"PASS: [n_wires]={n_wires}, dynamic VS get_unitary()")
        except AssertionError:
            logger.exception(f"FAIL: [n_wires]={n_wires}, dynamic VS " f"get_unitary()")
            raise AssertionError

        for wires_per_block in range(1, n_wires + 1):
            q_dev2.reset_identity_states()
            q_layer.static_on(wires_per_block=wires_per_block)
            q_layer(q_dev2)
            u_static_run = (
                q_dev2.states.clone().reshape(2**n_wires, 2**n_wires).permute(1, 0)
            )
            q_layer.static_off()
            try:
                assert torch.allclose(u_dyna, u_static_run, atol=1e-5)
                logger.info(
                    f"PASS: [n_wires]={n_wires}, dynamic VS static "
                    f"run with [wires_per_block]={wires_per_block}"
                )
            except AssertionError:
                logger.exception(
                    f"FAIL: [n_wires]={n_wires}, dynamic VS "
                    f"static run with [wires_per_block]"
                    f"={wires_per_block}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="pdb")
    args = parser.parse_args()

    if args.pdb:
        pdb.set_trace()

    torch.manual_seed(42)
    np.random.seed(42)

    static_mode_dynamic_vs_static_vs_get_unitary_test()
    static_mode_dynamic_vs_static_test()
    static_mode_dynamic_vs_static_gradients_test()
