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

import datetime
import itertools

import numpy as np
import pathos.multiprocessing as multiprocessing
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from torchpack.utils.logging import logger
from tqdm import tqdm

import torchquantum as tq
from torchquantum.util import (
    get_circ_stats,
    get_expectations_from_counts,
    get_provider_hub_group_project,
)

from .qiskit_macros import IBMQ_NAMES
from .qiskit_plugin import tq2qiskit, tq2qiskit_measurement, tq2qiskit_parameterized


class EmptyPassManager(PassManager):
    def run(self, circuits, output_name: str = None, callback=None):
        return circuits


def run_job_worker(data):
    while True:
        try:
            job = AerSimulator(**(data))
            result = job.result()
            counts = result.get_counts()
            # circ_num = len(data['parameter_binds'])
            # logger.info(
            #     f'run job worker successful, circuit number = {circ_num}')
            break
        except Exception as e:
            if "Job was cancelled" in str(e):
                logger.warning(f"Job is cancelled manually.")
                return None
            else:
                logger.warning(f"Job failed because {e}, rerun now.")

    return counts


class QiskitProcessor(object):
    def __init__(
        self,
        use_real_qc=False,
        backend_name=None,
        backend=None,
        noise_model_name=None,
        noise_model=None,
        coupling_map_name=None,
        coupling_map=None,
        basis_gates_name=None,
        basis_gates=None,
        n_shots=8192,
        initial_layout=None,
        seed_transpiler=42,
        seed_simulator=42,
        optimization_level=None,
        max_jobs=5,
        remove_ops=False,
        remove_ops_thres=1e-4,
        transpile_with_ancilla=True,
        hub="ibm-q",
        group="open",
        project="main",
        layout_method=None,
        routing_method=None,
    ):
        self.use_real_qc = use_real_qc
        self.noise_model_name = noise_model_name
        self.backend_name = backend_name
        self.coupling_map_name = coupling_map_name
        self.basis_gates_name = basis_gates_name
        self.n_shots = n_shots
        self.initial_layout = initial_layout
        self.seed_transpiler = seed_transpiler
        self.seed_simulator = seed_simulator
        self.optimization_level = optimization_level
        self.max_jobs = max_jobs
        self.transpile_with_ancilla = transpile_with_ancilla

        self.layout_method = layout_method
        self.routing_method = routing_method

        self.hub = hub
        self.group = group
        self.project = project
        self.backend = backend
        self.provider = None
        self.noise_model = noise_model
        self.coupling_map = coupling_map
        self.basis_gates = basis_gates
        self.properties = None
        self.empty_pass_manager = EmptyPassManager()

        self.transpiled_circs = None

        self.remove_ops = remove_ops
        self.remove_ops_thres = remove_ops_thres

        self.qiskit_init()

    def get_noise_model(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            self.properties = backend.properties()
            noise_model = NoiseModel.from_backend(backend)
        else:
            noise_model = None

        return noise_model

    def get_coupling_map(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            coupling_map = backend.configuration().coupling_map
        else:
            if name == "four_all":
                coupling_map = [
                    [0, 1],
                    [1, 0],
                    [0, 2],
                    [2, 0],
                    [0, 3],
                    [3, 0],
                    [1, 2],
                    [2, 1],
                    [1, 3],
                    [3, 1],
                    [2, 3],
                    [3, 2],
                ]
            else:
                coupling_map = None

        return coupling_map

    def get_basis_gates(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            basis_gates = backend.configuration().basis_gates
        else:
            basis_gates = None

        return basis_gates

    def qiskit_init(self):
        self.provider = None
        self.properties = None

        if self.backend is None:
            # initialize now
            self.provider = get_provider_hub_group_project(
                hub=self.hub,
                group=self.group,
                project=self.project,
            )
            if self.use_real_qc:
                self.backend = self.provider.get_backend(self.backend_name)
                self.properties = self.backend.properties()
                self.coupling_map = self.get_coupling_map(self.backend_name)
            else:
                # use simulator
                self.backend = AerSimulator(max_parallel_experiments=0)
                self.noise_model = self.get_noise_model(self.noise_model_name)
                self.coupling_map = self.get_coupling_map(self.coupling_map_name)
                self.basis_gates = self.get_basis_gates(self.basis_gates_name)
        else:
            # predefined backend
            self.backend_name = self.backend.name()
            print(f"Use backend: {self.backend_name}")
            if self.coupling_map is None:
                self.coupling_map = self.backend.configuration().coupling_map
            if self.basis_gates is None:
                self.basis_gates = self.backend.configuration().basis_gates

    def set_layout(self, layout):
        self.initial_layout = layout

    def set_backend(self, backend):
        self.backend = backend

    def transpile(self, circs):
        if not self.transpile_with_ancilla and self.coupling_map is not None:
            # only use same number of physical qubits as virtual qubits
            # !! the risk is that the remaining graph is not a connected graph,
            # need fix this later
            coupling_map = []
            for pair in self.coupling_map:
                if all([p_wire < len(circs.qubits) for p_wire in pair]):
                    coupling_map.append(pair)
        else:
            coupling_map = self.coupling_map
        transpiled_circs = transpile(
            circuits=circs,
            backend=self.backend,
            basis_gates=self.basis_gates,
            coupling_map=coupling_map,
            initial_layout=self.initial_layout,
            seed_transpiler=self.seed_transpiler,
            optimization_level=self.optimization_level,
        )
        return transpiled_circs

    def preprocess_parameterized(
        self,
        q_device,
        q_layer_parameterized,
        q_layer_fixed,
        q_layer_measure,
        x,
    ):
        circ_parameterized, params = tq2qiskit_parameterized(
            q_device, q_layer_parameterized.func_list
        )
        circ_fixed = tq2qiskit(
            q_device,
            q_layer_fixed,
            remove_ops=self.remove_ops,
            remove_ops_thres=self.remove_ops_thres,
        )

        circ_measurement = tq2qiskit_measurement(q_device, q_layer_measure)
        circ = circ_parameterized + circ_fixed + circ_measurement

        logger.info(f"Before transpile: {get_circ_stats(circ)}")
        transpiled_circ = self.transpile(circ)
        logger.info(f"After transpile: {get_circ_stats(transpiled_circ)}")
        self.transpiled_circs = [transpiled_circ]
        # construct the parameter_binds
        binds_all = []
        for inputs_single in x:
            binds = {}
            for k, input_single in enumerate(inputs_single):
                binds[params[k]] = input_single.item()
            binds_all.append(binds)

        return transpiled_circ, binds_all

    def process_parameterized(
        self,
        q_device: tq.QuantumDevice,
        q_layer_parameterized: tq.QuantumModule,
        q_layer_fixed: tq.QuantumModule,
        q_layer_measure: tq.QuantumModule,
        x,
        parallel=True,
    ):
        """
        separate the conversion, encoder part will be converted to a
        parameterized Qiskit QuantumCircuit. The remaining part will be a
        non-parameterized QuantumCircuit. In this case, only one time of
        compilation is required.

        q_layer_parameterized needs to have a func_list to specify the gates

        for parallel:
        JobManager has bugs when submitting job, so use multiprocessing instead
        """
        transpiled_circ, binds_all = self.preprocess_parameterized(
            q_device, q_layer_parameterized, q_layer_fixed, q_layer_measure, x
        )

        if parallel:
            if hasattr(self.backend.configuration(), "max_experiments"):
                chunk_size = self.backend.configuration().max_experiments
            else:
                # using simulator, apply multithreading
                chunk_size = len(binds_all) // self.max_jobs

            if chunk_size == 0:
                split_binds = [binds_all]
            else:
                split_binds = [
                    binds_all[i : i + chunk_size]
                    for i in range(0, len(binds_all), chunk_size)
                ]

            feed_dicts = []
            for split_bind in split_binds:
                feed_dict = {
                    "experiments": transpiled_circ,
                    "backend": self.backend,
                    "pass_manager": self.empty_pass_manager,
                    "shots": self.n_shots,
                    "seed_simulator": self.seed_simulator,
                    "noise_model": self.noise_model,
                    "parameter_binds": split_bind,
                }
                feed_dicts.append(feed_dict)

            p = multiprocessing.Pool(self.max_jobs)
            results = p.map(run_job_worker, feed_dicts)
            p.close()

            if all(isinstance(result, dict) for result in results):
                counts = results
            else:
                if isinstance(results[-1], dict):
                    results[-1] = [results[-1]]
                counts = list(itertools.chain(*results))
        else:
            job = self.backend.run(
                transpiled_circ,
                pass_manager=self.empty_pass_manager,
                shots=self.n_shots,
                seed_simulator=self.seed_simulator,
                noise_model=self.noise_model,
                parameter_binds=binds_all,
            )

            result = job.result()
            counts = result.get_counts()

        measured_qiskit = get_expectations_from_counts(counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)

        return measured_qiskit

    def preprocess_parameterized_and_shift(
        self,
        q_device,
        q_layer_parameterized,
        q_layer_fixed,
        q_layer_measure,
        x,
        shift_encoder,
        shift_this_step,
    ):
        circ_parameterized, params = tq2qiskit_parameterized(
            q_device, q_layer_parameterized.func_list
        )
        circ_fixed_list = []
        circ_fixed = tq2qiskit(
            q_device,
            q_layer_fixed,
            remove_ops=self.remove_ops,
            remove_ops_thres=self.remove_ops_thres,
        )
        circ_fixed_list.append(circ_fixed)

        # not shift encoder ==> shift fixed layer
        if not shift_encoder:
            for i, named_param in enumerate(q_layer_fixed.named_parameters()):
                if shift_this_step[i]:
                    param = named_param[-1]
                    param.copy_(param + np.pi * 0.5)
                    circ_fixed = tq2qiskit(
                        q_device,
                        q_layer_fixed,
                        remove_ops=self.remove_ops,
                        remove_ops_thres=self.remove_ops_thres,
                    )
                    circ_fixed_list.append(circ_fixed)
                    param.copy_(param - np.pi)
                    circ_fixed = tq2qiskit(
                        q_device,
                        q_layer_fixed,
                        remove_ops=self.remove_ops,
                        remove_ops_thres=self.remove_ops_thres,
                    )
                    circ_fixed_list.append(circ_fixed)
                    param.copy_(param + np.pi * 0.5)

        self.transpiled_circs = []
        for circ_fixed in circ_fixed_list:
            circ = circ_parameterized + circ_fixed
            v_c_reg_mapping = q_layer_measure.v_c_reg_mapping
            if v_c_reg_mapping is not None:
                for q_reg, c_reg in v_c_reg_mapping["v2c"].items():
                    circ.measure(q_reg, c_reg)
            else:
                circ.measure(
                    list(range(q_device.n_wires)), list(range(q_device.n_wires))
                )

            transpiled_circ = self.transpile(circ)
            self.transpiled_circs.append(transpiled_circ)
        # construct the parameter_binds
        binds_all = []
        if shift_encoder:
            for idx in range(x.size()[1]):
                x[:, idx] += np.pi * 0.5
                for inputs_single in x:
                    binds = {}
                    for k, input_single in enumerate(inputs_single):
                        binds[params[k]] = input_single.item()
                    binds_all.append(binds)

                x[:, idx] -= np.pi
                for inputs_single in x:
                    binds = {}
                    for k, input_single in enumerate(inputs_single):
                        binds[params[k]] = input_single.item()
                    binds_all.append(binds)

                x[:, idx] += np.pi * 0.5
        else:
            for inputs_single in x:
                binds = {}
                for k, input_single in enumerate(inputs_single):
                    binds[params[k]] = input_single.item()
                binds_all.append(binds)

        return self.transpiled_circs, binds_all

    def process_parameterized_and_shift(
        self,
        q_device: tq.QuantumDevice,
        q_layer_parameterized: tq.QuantumModule,
        q_layer_fixed: tq.QuantumModule,
        q_layer_measure: tq.QuantumModule,
        x,
        shift_encoder=False,
        parallel=True,
        shift_this_step=None,
    ):
        """
        separate the conversion, encoder part will be converted to a
        parameterized Qiskit QuantumCircuit. The remaining part will be a
        non-parameterized QuantumCircuit. In this case, only one time of
        compilation is required.

        q_layer_parameterized needs to have a func_list to specify the gates

        for parallel:
        JobManager has bugs when submitting job, so use multiprocessing instead
        """
        transpiled_circs, binds_all = self.preprocess_parameterized_and_shift(
            q_device,
            q_layer_parameterized,
            q_layer_fixed,
            q_layer_measure,
            x,
            shift_encoder,
            shift_this_step,
        )

        time_spent_list = []

        if parallel:
            if hasattr(self.backend.configuration(), "max_experiments"):
                chunk_size = self.backend.configuration().max_experiments
            else:
                # using simulator, apply multithreading
                chunk_size = len(binds_all) // self.max_jobs

            split_binds = [
                binds_all[i : i + chunk_size]
                for i in range(0, len(binds_all), chunk_size)
            ]

            feed_dicts = []
            for split_bind in split_binds:
                feed_dict = {
                    "experiments": transpiled_circs,
                    "backend": self.backend,
                    "pass_manager": self.empty_pass_manager,
                    "shots": self.n_shots,
                    "seed_simulator": self.seed_simulator,
                    "noise_model": self.noise_model,
                    "parameter_binds": split_bind,
                }
                feed_dicts.append(feed_dict)

            p = multiprocessing.Pool(self.max_jobs)
            results = p.map(run_job_worker, feed_dicts)
            p.close()

            if all(isinstance(result, dict) for result in results):
                counts = results
            else:
                if isinstance(results[-1], dict):
                    results[-1] = [results[-1]]
                counts = list(itertools.chain(*results))
        else:
            chunk_size = 75 // len(binds_all)
            split_circs = [
                transpiled_circs[i : i + chunk_size]
                for i in range(0, len(transpiled_circs), chunk_size)
            ]
            counts = []
            total_time_spent = datetime.timedelta()
            total_cont = 0
            for circ in split_circs:
                while True:
                    try:
                        job = self.backend.run(
                            circ,
                            pass_manager=self.empty_pass_manager,
                            shots=self.n_shots,
                            seed_simulator=self.seed_simulator,
                            noise_model=self.noise_model,
                            parameter_binds=binds_all,
                        )

                        result = (
                            job.result()
                        )  # qiskit.providers.ibmq.job.exceptions.IBMQJobFailureError:Job has failed. Use the error_message() method to get more details
                        counts = counts + result.get_counts()
                        # time_per_step = job.time_per_step()
                        # time_spent = time_per_step['COMPLETED'] - time_per_step['RUNNING'] + time_per_step['QUEUED'] - job.time_per_step()['CREATING']
                        # time_spent_list.append(time_spent)
                        # print(time_spent)
                        # total_time_spent += time_spent
                        # total_cont += 1
                        # print(total_time_spent / total_cont)
                        break
                    except QiskitError as e:
                        logger.warning("Job failed, rerun now.")
                        print(e.message)

        measured_qiskit = get_expectations_from_counts(counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)

        return measured_qiskit, time_spent_list

    def process_multi_measure(
        self,
        q_device: tq.QuantumDevice,
        q_layer: tq.QuantumModule,
        q_layer_measure: tq.QuantumModule,
    ):
        obs_list = q_layer_measure.obs_list
        circ_fixed = tq2qiskit(
            q_device,
            q_layer,
            remove_ops=self.remove_ops,
            remove_ops_thres=self.remove_ops_thres,
        )

        transpiled_circ_fixed = self.transpile(circ_fixed)

        circ_all = []

        for hamil in obs_list:
            circ_diagonalize = QuantumCircuit(q_device.n_wires, q_device.n_wires)

            # diagonalize the measurements
            for wire, observable in zip(hamil["wires"], hamil["observables"]):
                if observable == "x":
                    circ_diagonalize.h(qubit=wire)
                elif observable == "y":
                    circ_diagonalize.z(qubit=wire)
                    circ_diagonalize.s(qubit=wire)
                    circ_diagonalize.h(qubit=wire)

            circ_measurement = tq2qiskit_measurement(q_device, q_layer_measure)

            circ_diagonalize = circ_diagonalize + circ_measurement

            transpiled_circ_diagonalize = self.transpile(circ_diagonalize)
            circ_all.append(transpiled_circ_fixed + transpiled_circ_diagonalize)

        self.transpiled_circs = circ_all

        if hasattr(self.backend.configuration(), "max_experiments"):
            chunk_size = self.backend.configuration().max_experiments
        else:
            # using simulator, apply multithreading
            chunk_size = len(circ_all) // self.max_jobs

        split_circs = [
            circ_all[i : i + chunk_size] for i in range(0, len(circ_all), chunk_size)
        ]

        feed_dicts = []
        for split_circ in split_circs:
            feed_dict = {
                "experiments": split_circ,
                "backend": self.backend,
                "pass_manager": self.empty_pass_manager,
                "shots": self.n_shots,
                "seed_simulator": self.seed_simulator,
                "noise_model": self.noise_model,
            }
            feed_dicts.append(feed_dict)

        p = multiprocessing.Pool(self.max_jobs)
        results = p.map(run_job_worker, feed_dicts)
        p.close()

        if all(isinstance(result, dict) for result in results):
            counts = results
        else:
            if isinstance(results[-1], dict):
                results[-1] = [results[-1]]
            counts = list(itertools.chain(*results))

        measured_qiskit = get_expectations_from_counts(counts, n_wires=q_device.n_wires)

        measured_qiskit = torch.tensor(measured_qiskit, device=q_device.state.device)

        return measured_qiskit

    def process(
        self,
        q_device: tq.QuantumDevice,
        q_layer: tq.QuantumModule,
        q_layer_measure: tq.QuantumModule,
        x,
    ):
        circs = []
        for i, x_single in tqdm(enumerate(x)):
            circ = tq2qiskit(q_device, q_layer, x_single.unsqueeze(0))
            circ_measurement = tq2qiskit_measurement(q_device, q_layer_measure)
            circ = circ + circ_measurement

            circs.append(circ)

        transpiled_circs = self.transpile(circs)
        self.transpiled_circs = transpiled_circs

        job = self.backend.run(
            transpiled_circs,
            shots=self.n_shots,
            # initial_layout=self.initial_layout,
            seed_transpiler=self.seed_transpiler,
            seed_simulator=self.seed_simulator,
            coupling_map=self.coupling_map,
            basis_gates=self.basis_gates,
            noise_model=self.noise_model,
            optimization_level=self.optimization_level,
        )

        result = job.result()
        counts = result.get_counts()

        measured_qiskit = get_expectations_from_counts(counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)

        return measured_qiskit

    def process_ready_circs_get_counts(self, circs_all, parallel=True):
        circs_all_transpiled = []
        for circ in tqdm(circs_all):
            circs_all_transpiled.append(self.transpile(circ))

        circs_all = circs_all_transpiled

        if parallel:
            if hasattr(self.backend.configuration(), "max_experiments"):
                chunk_size = self.backend.configuration().max_experiments
            else:
                # using simulator, apply multithreading
                chunk_size = len(circs_all) // self.max_jobs
                if chunk_size == 0:
                    chunk_size = 1

            split_circs = [
                circs_all[i : i + chunk_size]
                for i in range(0, len(circs_all), chunk_size)
            ]

            feed_dicts = []
            for split_circ in split_circs:
                feed_dict = {
                    "experiments": split_circ,
                    "backend": self.backend,
                    "pass_manager": self.empty_pass_manager,
                    "shots": self.n_shots,
                    "seed_simulator": self.seed_simulator,
                    "noise_model": self.noise_model,
                }
                feed_dicts.append(feed_dict)

            p = multiprocessing.Pool(self.max_jobs)
            results = p.map(run_job_worker, feed_dicts)
            p.close()

            if all(isinstance(result, dict) for result in results):
                counts = results
            else:
                if isinstance(results[-1], dict):
                    results[-1] = [results[-1]]
                counts = list(itertools.chain(*results))
        else:
            job = self.backend.run(
                circs_all,
                pass_manager=self.empty_pass_manager,
                shots=self.n_shots,
                seed_simulator=self.seed_simulator,
                noise_model=self.noise_model,
            )

            result = job.result()
            counts = [result.get_counts()]
        return counts

    def process_ready_circs(self, q_device, circs_all, parallel=True):
        counts = self.process_ready_circs_get_counts(circs_all, parallel=parallel)
        measured_qiskit = get_expectations_from_counts(counts, n_wires=q_device.n_wires)
        measured_torch = torch.tensor(measured_qiskit)
        return measured_torch

    def process_circs_get_joint_expval(self, circs_all, observable, parallel=True):
        """
        This function is used to compute the joint expectation value of a list of observables
        we add diagonalizing gates before sending them to the backend
        """
        observable = observable.upper()
        circs_all_diagonalized = []
        for circ_ in circs_all:
            circ = circ_.copy()
            for k, obs in enumerate(observable):
                if obs == "X":
                    circ.h(k)
                elif obs == "Y":
                    circ.z(k)
                    circ.s(k)
                    circ.h(k)
            circ.measure_all()
            circs_all_diagonalized.append(circ)

        expval_all = []

        mask = np.ones(len(observable), dtype=bool)
        mask[np.array([*observable]) == "I"] = False

        counts = self.process_ready_circs_get_counts(
            circs_all_diagonalized, parallel=parallel
        )

        # here we need to switch the little and big endian of distribution bitstrings
        distributions = []
        for count in counts:
            distribution = {}
            for k, v in count.items():
                distribution[k[::-1]] = v
            distributions.append(distribution)

        for distri in distributions:
            n_eigen_one = 0
            n_eigen_minus_one = 0
            for bitstring, n_count in distri.items():
                if (
                    np.dot(list(map(lambda x: eval(x), [*bitstring])), mask).sum() % 2
                    == 0
                ):
                    n_eigen_one += n_count
                else:
                    n_eigen_minus_one += n_count

            expval = (
                n_eigen_one / self.n_shots + (-1) * n_eigen_minus_one / self.n_shots
            )
            expval_all.append(expval)

        return expval_all


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    circ = QuantumCircuit(3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.rx(0.1, 0)

    qiskit_processor = QiskitProcessor(use_real_qc=False)

    qiskit_processor.process_circs_get_joint_expval([circ], "XII")

    qdev = tq.QuantumDevice(n_wires=3, bsz=1)
    qdev.h(0)
    qdev.cx([0, 1])
    qdev.cx([1, 2])
    qdev.rx(0, 0.1)

    from torchquantum.measurement import expval_joint_sampling

    print(expval_joint_sampling(qdev, "XII", n_shots=8192))
