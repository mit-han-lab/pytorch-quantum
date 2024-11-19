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
from test.static_mode_test import QLayer as AllRandomLayer

import numpy as np
import torch
from qiskit_aer import AerSimulator
from torchpack.utils.logging import logger

import torchquantum as tq
from torchquantum.macro import F_DTYPE
from torchquantum.plugin import tq2qiskit
from torchquantum.util import (
    find_global_phase,
    get_expectations_from_counts,
    switch_little_big_endian_matrix,
    switch_little_big_endian_state,
)


def unitary_tq_vs_qiskit_test():
    for n_wires in range(2, 10):
        q_dev = tq.QuantumDevice(n_wires=n_wires)
        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_layer = AllRandomLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
            qiskit_compatible=True,
        )

        unitary_tq = q_layer.get_unitary(q_dev, x)
        unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

        # qiskit
        circ = tq2qiskit(q_layer, x)
        simulator = AerSimulator(method="unitary")
        circ.save_unitary()
        result = simulator.run(circ).result()
        unitary_qiskit = result.get_unitary(circ)

        stable_threshold = 1e-5
        try:
            # WARNING: need to remove the global phase! The qiskit simulated
            # results sometimes has global phase shift.
            global_phase = find_global_phase(
                unitary_tq, unitary_qiskit, stable_threshold
            )

            if global_phase is None:
                logger.exception(
                    f"Cannot find a stable enough factor to "
                    f"reduce the global phase, increase the "
                    f"stable_threshold and try again"
                )
                raise RuntimeError

            assert np.allclose(unitary_tq * global_phase, unitary_qiskit, atol=1e-6)
            logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

        except AssertionError:
            logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
            raise AssertionError

        except RuntimeError:
            raise RuntimeError

    logger.info(f"PASS tq vs qiskit unitary test")


def state_tq_vs_qiskit_test():
    bsz = 1
    for n_wires in range(2, 10):
        q_dev = tq.QuantumDevice(n_wires=n_wires)
        q_dev.reset_states(bsz=bsz)

        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_layer = AllRandomLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
            qiskit_compatible=True,
        )

        q_layer(q_dev, x)
        state_tq = q_dev.states.reshape(bsz, -1)
        state_tq = switch_little_big_endian_state(state_tq.data.numpy())

        # qiskit
        circ = tq2qiskit(q_layer, x)
        # Select the StatevectorSimulator from the Aer provider
        simulator = AerSimulator(method="statevector")
        circ.save_statevector()

        # Execute and get counts
        result = simulator.run(circ).result()
        state_qiskit = result.get_statevector(circ)

        stable_threshold = 1e-5
        try:
            # WARNING: need to remove the global phase! The qiskit simulated
            # results sometimes has global phase shift.
            global_phase = find_global_phase(
                state_tq, np.expand_dims(state_qiskit, 0), stable_threshold
            )

            if global_phase is None:
                logger.exception(
                    f"Cannot find a stable enough factor to "
                    f"reduce the global phase, increase the "
                    f"stable_threshold and try again"
                )
                raise RuntimeError

            assert np.allclose(state_tq * global_phase, state_qiskit, atol=1e-6)
            logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

        except AssertionError:
            logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
            raise AssertionError

        except RuntimeError:
            raise RuntimeError

    logger.info(f"PASS tq vs qiskit statevector test")


def measurement_tq_vs_qiskit_test():
    bsz = 1
    for n_wires in range(2, 10):
        q_dev = tq.QuantumDevice(n_wires=n_wires)
        q_dev.reset_states(bsz=bsz)

        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_layer = AllRandomLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
            qiskit_compatible=True,
        )

        q_layer(q_dev, x)
        measurer = tq.MeasureAll(obs=tq.PauliZ)
        # flip because qiskit is from N to 0, tq is from 0 to N
        measured_tq = np.flip(measurer(q_dev).data[0].numpy())

        # qiskit
        circ = tq2qiskit(q_layer, x)
        circ.measure(list(range(n_wires)), list(range(n_wires)))

        simulator = AerSimulator()

        # Execute and get counts
        result = simulator.run(circ, shots=1000000).result()
        counts = result.get_counts(circ)
        measured_qiskit = get_expectations_from_counts(counts, n_wires=n_wires)

        try:
            # WARNING: the measurement has randomness, so tolerate larger
            # differences (MAX 20%) between tq and qiskit
            # typical mean difference is less than 1%
            diff = np.abs(measured_tq - measured_qiskit).mean()
            diff_ratio = (
                np.abs((measured_tq - measured_qiskit) / measured_qiskit)
            ).mean()
            logger.info(f"Diff: tq vs qiskit {diff} \t Diff Ratio: " f"{diff_ratio}")
            assert np.allclose(measured_tq, measured_qiskit, atol=1e-4, rtol=2e-1)
            logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

        except AssertionError:
            logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
            raise AssertionError

    logger.info(f"PASS tq vs qiskit measurement test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="pdb")
    args = parser.parse_args()

    # seed = 45
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    if args.pdb:
        pdb.set_trace()

    unitary_tq_vs_qiskit_test()
    state_tq_vs_qiskit_test()
    measurement_tq_vs_qiskit_test()
