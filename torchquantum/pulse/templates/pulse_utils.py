import copy
from typing import Union

import numpy as np
import qiskit
from qiskit import QuantumCircuit, pulse
from qiskit.compiler import assemble
from qiskit.providers.fake_provider import *
from qiskit.pulse import Schedule
from qiskit.pulse.instructions import Instruction
from qiskit.pulse.transforms import block_to_schedule
from scipy.optimize import LinearConstraint


def is_parametric_pulse(t0, *inst: Union["Schedule", Instruction]):
    inst = t0[1]
    t0 = t0[0]
    if isinstance(inst, pulse.Play):
        return True
    else:
        return False


def extract_ampreal(pulse_prog):
    # extract the real part of pulse amplitude, igonred the imaginary part.
    amp_list = list(
        map(
            lambda x: x[1].pulse.amp,
            pulse_prog.filter(is_parametric_pulse).instructions,
        )
    )
    amp_list = np.array(amp_list)
    ampa_list = np.angle(np.array(amp_list))
    return ampa_list


def extract_amp(pulse_prog):
    # extract the pulse amplitdue.
    amp_list = list(
        map(
            lambda x: x[1].pulse.amp,
            pulse_prog.filter(is_parametric_pulse).instructions,
        )
    )
    amp_list = np.array(amp_list)
    ampa_list = np.angle(np.array(amp_list))
    ampn_list = np.abs(np.array(amp_list))
    amps_list = []
    for i, j in zip(ampn_list, ampa_list):
        amps_list.append(i)
        amps_list.append(j)
    amps_list = np.array(amps_list)
    return amps_list


def is_phase_pulse(t0, *inst: Union["Schedule", Instruction]):
    inst = t0[1]
    t0 = t0[0]
    if isinstance(inst, pulse.ShiftPhase):
        return True
    return False


def extract_phase(pulse_prog):

    for _, ShiftPhase in pulse_prog.filter(is_phase_pulse).instructions:
        # print(play.pulse.amp)
        pass
    instructions = pulse_prog.filter(is_phase_pulse).instructions

    phase_list = list(
        map(
            lambda x: x[1]._operands[0],
            pulse_prog.operands[0].filter(is_phase_pulse).instructions,
        )
    )
    return phase_list


def cir2pul(circuit, backend):
    # transform quantum circuit to pulse schedule
    with pulse.build(backend) as pulse_prog:
        pulse.call(circuit)
    return pulse_prog


def snp(qubit, backend):
    circuit = QuantumCircuit(qubit + 1)
    circuit.h(qubit)
    sched = cir2pul(circuit, backend)
    sched = block_to_schedule(sched)
    return sched


def tnp(qubit, cqubit, backend):
    circuit = QuantumCircuit(cqubit + 1)
    circuit.cx(qubit, cqubit)
    sched = cir2pul(circuit, backend)
    sched = block_to_schedule(sched)
    return sched


def pul_append(sched1, sched2):
    sched = sched1.append(sched2)
    return sched


def map_amp(pulse_ansatz, modified_list):
    sched = Schedule()
    for inst, amp in zip(
        pulse_ansatz.filter(is_parametric_pulse).instructions, modified_list
    ):
        inst[1].pulse._amp = amp
    for i in pulse_ansatz.instructions:
        if is_parametric_pulse(i):
            sched += copy.deepcopy(i[1])
    return sched


def get_from(d: dict, key: str):

    value = 0
    if key in d:
        value = d[key]
    return value


def run_pulse_sim(measurement_pulse):
    measure_result = []
    for measure_pulse in measurement_pulse:
        shots = 1024
        pulse_sim = qiskit_aer.PulseSimulator.from_backend(FakeJakarta())
        pul_sim = assemble(
            measure_pulse,
            backend=pulse_sim,
            shots=1024,
            meas_level=2,
            meas_return="single",
        )
        results = pulse_sim.run(pul_sim).result()

        counts = results.get_counts()
        expectation_value = (
            (get_from(counts, "00") + get_from(counts, "11"))
            - (get_from(counts, "10") + get_from(counts, "01"))
        ) / shots
        measure_result.append(expectation_value)
    return measure_result


def gen_LC(parameters_array):
    dim_design = int(len(parameters_array))
    Mid = int(len(parameters_array) / 2)
    bound = np.ones((dim_design, 2)) * np.array([0, 0.9])
    bound[-Mid:] = bound[-Mid:] * np.pi * 2
    tol = 1e-3  # tolerance for optimization precision.
    lb = bound[:, 0]
    ub = bound[:, 1]
    LC = LinearConstraint(np.eye(dim_design), lb, ub, keep_feasible=False)
    return LC


def observe_genearte(pulse_ansatz, backend):
    qubits = 0, 1
    with pulse.build(backend) as pulse_measurez0:
        # z measurement of qubit 0 and 1
        pulse.call(pulse_ansatz)
        pulse.barrier(0, 1)
        pulse.measure(0)
    with pulse.build(backend) as pulse_measurez1:
        # z measurement of qubit 0 and 1
        pulse.call(pulse_ansatz)
        pulse.barrier(0, 1)
        pulse.measure(1)
    with pulse.build(backend) as pulse_measurez:
        # z measurement of qubit 0 and 1
        pulse.call(pulse_ansatz)
        pulse.barrier(0, 1)
        pulse.measure(qubits)
    with pulse.build(backend) as pulse_measurex:
        # x measurement of qubit 0 and 1
        pulse.call(pulse_ansatz)
        pulse.barrier(0, 1)
        pulse.u2(0, np.pi, 0)
        pulse.u2(0, np.pi, 1)
        pulse.barrier(0, 1)
        pulse.measure(qubits)

    with pulse.build(backend) as pulse_measurey:
        # y measurement of qubit 0 and 1
        pulse.call(pulse_ansatz)
        pulse.barrier(0, 1)
        pulse.u2(np.pi / 2, 0, 0)
        pulse.u2(np.pi / 2, 0, 1)
        pulse.barrier(0, 1)
        pulse.measure(qubits)
    measurement_pulse = [
        pulse_measurez0,
        pulse_measurez1,
        pulse_measurez,
        pulse_measurex,
    ]
    return measurement_pulse
