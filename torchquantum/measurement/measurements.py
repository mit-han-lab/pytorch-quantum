import random

import torch
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from torchquantum.macro import F_DTYPE

from typing import Union, List
from collections import Counter, OrderedDict

from torchquantum.functional import mat_dict
# from ..operator import op_name_dict, Observable
import torchquantum.operator as op
from copy import deepcopy
import matplotlib.pyplot as plt

__all__ = [
    "find_observable_groups",
    "expval_joint_sampling_grouping",
    "expval_joint_analytical",
    "expval_joint_sampling",
    "expval",
    "MeasureAll",
    "MeasureMultipleTimes",
    "MeasureMultiPauliSum",
    "MeasureMultiQubitPauliSum",
    "gen_bitstrings",
    "measure",
]


def gen_bitstrings(n_wires):
    return ["{:0{}b}".format(k, n_wires) for k in range(2 ** n_wires)]


def measure(qdev, n_shots=1024, draw_id=None):
    """Measure the target state and obtain classical bitstream distribution
    Args:
        q_state: input tq.QuantumDevice
        n_shots: number of simulated shots
    Returns:
        distribution of bitstrings
    """

    """
    In measure function, the statevector is copied to the CPU and 
    a list of all possible 2^n bitstrings is constructed to do the sampling. 
    This is again a huge CPU memory and runtime overhead since sampling can done on the GPU directly and efficiently. 
    Here is a sketch of how that might look like in PyTorch using Inverse transform sampling method: 
    Calculate squared amplitudes on GPU using troch.abs and troch.square. 
    Calculate the cumulative distribution function using troch.cumsum. 
    Generate random numbers (as many as the required number of samples) between 0 and 1 using torch.rand. 
    Find the index of each random number inside the cumulative distribution using troch.searchsorted. 
    Copy the indices to the CPU and convert each number to its bitstring binary representation.
    """

    bitstring_candidates = gen_bitstrings(qdev.n_wires)  # length is 2 to the power of n_wires
    if isinstance(qdev, tq.QuantumDevice):
        # length is 2 to the power of n_wires
        state_mag = qdev.get_states_1d().abs().detach().cpu().numpy()
    elif isinstance(qdev, tq.NoiseDevice):
        '''
        Measure the density matrix in the computational basis
        '''
        state_mag = qdev.get_probs_1d().abs().detach().cpu().numpy()
    distri_all = []

    for state_mag_one in state_mag:
        state_prob_one = np.abs(state_mag_one) ** 2
        measured = random.choices(
            population=bitstring_candidates,
            weights=state_prob_one,
            k=n_shots,
        )
        counter = Counter(measured)
        counter.update({key: 0 for key in bitstring_candidates})
        distri = dict(counter)
        distri = OrderedDict(sorted(distri.items()))
        distri_all.append(distri)

    if draw_id is not None:
        plt.bar(distri_all[draw_id].keys(), distri_all[draw_id].values())
        plt.xticks(rotation="vertical")
        plt.xlabel("bitstring [qubit0, qubit1, ..., qubitN]")
        plt.title("distribution of measured bitstrings")
        plt.show()
    return distri_all


def find_observable_groups(observables):
    # the group is not unique
    # ["XXII", "IIZZ", "ZZII"] can be grouped as ["XXZZ", "ZZII"] or ["ZZZZ", "XXZZ"]
    groups = {}
    for observable in observables:
        matched = False
        for group, elements in groups.items():
            group_new = deepcopy(group)
            for k in range(len(observable)):
                if (observable[k] == 'I') or (observable[k] == group[k]):  # not finish yet
                    continue
                elif observable[k] != group[k]:
                    if group[k] == "I":
                        if k < len(observable) - 1:  # not finish yet
                            group_new = group_new[:k] + observable[k] + group_new[k + 1:]
                        else:
                            group_new = group_new[:k] + observable[k]
                        continue
                    else:
                        break
            else:  # for this group, the observable is matched or I be replaced, so no need to try other groups
                matched = True
                break
        if matched:
            # change the group name of matched one
            elements = groups[group]
            del groups[group]
            elements.append(observable)
            groups[group_new] = elements
        else:
            # no matched group, creata a new one
            groups[observable] = [observable]

    return groups


def expval_joint_sampling_grouping(
        qdev: tq.QuantumDevice,
        observables: List[str],
        n_shots_per_group=1024,
):
    assert len(observables) == len(set(observables)), "each observable should be unique"
    # key is the group, values is the list of sub-observables  
    obs = []
    for observable in observables:
        obs.append(observable.upper())
    # firstly find the groups
    groups = find_observable_groups(obs)

    # rotation to the desired basis
    n_wires = qdev.n_wires
    paulix = op.op_name_dict["paulix"]
    pauliy = op.op_name_dict["pauliy"]
    pauliz = op.op_name_dict["pauliz"]
    iden = op.op_name_dict["i"]
    pauli_dict = {"X": paulix, "Y": pauliy, "Z": pauliz, "I": iden}

    expval_all_obs = {}
    for obs_group, obs_elements in groups.items():
        # for each group need to clone a new qdev and its states
        qdev_clone = tq.QuantumDevice(n_wires=qdev.n_wires, bsz=qdev.bsz, device=qdev.device)
        qdev_clone.clone_states(qdev.states)

        for wire in range(n_wires):
            for rotation in pauli_dict[obs_group[wire]]().diagonalizing_gates():
                rotation(qdev_clone, wires=wire)
        # measure
        distributions = measure(qdev_clone, n_shots=n_shots_per_group)
        # interpret the distribution for different observable elements
        for obs_element in obs_elements:
            expval_all = []
            mask = np.ones(len(obs_element), dtype=bool)
            mask[np.array([*obs_element]) == "I"] = False

            for distri in distributions:
                n_eigen_one = 0
                n_eigen_minus_one = 0
                for bitstring, n_count in distri.items():
                    if np.dot(list(map(lambda x: eval(x), [*bitstring])), mask).sum() % 2 == 0:
                        n_eigen_one += n_count
                    else:
                        n_eigen_minus_one += n_count

                expval = n_eigen_one / n_shots_per_group + (-1) * n_eigen_minus_one / n_shots_per_group

                expval_all.append(expval)
            expval_all_obs[obs_element] = torch.tensor(expval_all, dtype=F_DTYPE)

    return expval_all_obs


def expval_joint_sampling(
        qdev: tq.QuantumDevice,
        observable: str,
        n_shots=1024,
):
    """
    Compute the expectation value of a joint observable from sampling 
    the measurement bistring
    Args:
        qdev: the quantum device
        observable: the joint observable, on the qubit 0, 1, 2, 3, etc in this order
    Returns:
        the expectation value
    Examples:
    >>> import torchquantum as tq
    >>> import torchquantum.functional as tqf
    >>> x = tq.QuantumDevice(n_wires=2)
    >>> tqf.hadamard(x, wires=0)
    >>> tqf.x(x, wires=1)
    >>> tqf.cnot(x, wires=[0, 1])
    >>> print(expval_joint_sampling(x, 'II', n_shots=8192))
    tensor([[0.9997]])
    >>> print(expval_joint_sampling(x, 'XX', n_shots=8192))
    tensor([[0.9991]])
    >>> print(expval_joint_sampling(x, 'ZZ', n_shots=8192))
    tensor([[-0.9980]])
    """
    # rotation to the desired basis
    n_wires = qdev.n_wires
    paulix = op.op_name_dict["paulix"]
    pauliy = op.op_name_dict["pauliy"]
    pauliz = op.op_name_dict["pauliz"]
    iden = op.op_name_dict["i"]
    pauli_dict = {"X": paulix, "Y": pauliy, "Z": pauliz, "I": iden}

    qdev_clone = tq.QuantumDevice(n_wires=qdev.n_wires, bsz=qdev.bsz, device=qdev.device)
    qdev_clone.clone_states(qdev.states)

    observable = observable.upper()
    for wire in range(n_wires):
        for rotation in pauli_dict[observable[wire]]().diagonalizing_gates():
            rotation(qdev_clone, wires=wire)

    mask = np.ones(len(observable), dtype=bool)
    mask[np.array([*observable]) == "I"] = False

    expval_all = []
    # measure
    distributions = measure(qdev_clone, n_shots=n_shots)
    for distri in distributions:
        n_eigen_one = 0
        n_eigen_minus_one = 0
        for bitstring, n_count in distri.items():
            if np.dot(list(map(lambda x: eval(x), [*bitstring])), mask).sum() % 2 == 0:
                n_eigen_one += n_count
            else:
                n_eigen_minus_one += n_count

        expval = n_eigen_one / n_shots + (-1) * n_eigen_minus_one / n_shots
        expval_all.append(expval)

    return torch.tensor(expval_all, dtype=F_DTYPE)


def expval_joint_analytical(
        qdev: tq.QuantumDevice,
        observable: str,
):
    """
    Compute the expectation value of a joint observable in analytical way, assuming the
    statevector is available.
    Args:
        qdev: the quantum device
        observable: the joint observable, on the qubit 0, 1, 2, 3, etc in this order
    Returns:
        the expectation value
    Examples:
    >>> import torchquantum as tq
    >>> import torchquantum.functional as tqf
    >>> x = tq.QuantumDevice(n_wires=2)
    >>> tqf.hadamard(x, wires=0)
    >>> tqf.x(x, wires=1)
    >>> tqf.cnot(x, wires=[0, 1])
    >>> print(expval_joint_analytical(x, 'II'))
    tensor([[1.0000]])
    >>> print(expval_joint_analytical(x, 'XX'))
    tensor([[1.0000]])
    >>> print(expval_joint_analytical(x, 'ZZ'))
    tensor([[-1.0000]])
    """
    # compute the hamiltonian matrix
    paulix = mat_dict["paulix"]
    pauliy = mat_dict["pauliy"]
    pauliz = mat_dict["pauliz"]
    iden = mat_dict["i"]
    pauli_dict = {"X": paulix, "Y": pauliy, "Z": pauliz, "I": iden}

    observable = observable.upper()
    assert len(observable) == qdev.n_wires
    states = qdev.get_states_1d()

    hamiltonian = pauli_dict[observable[0]].to(states.device)
    for op in observable[1:]:
        hamiltonian = torch.kron(hamiltonian, pauli_dict[op].to(states.device))

    # torch.mm(states, torch.mm(hamiltonian, states.conj().transpose(0, 1))).real

    return (
        (states.conj() * torch.mm(hamiltonian, states.transpose(0, 1)).transpose(0, 1))
        .sum(-1)
        .real
    )


def expval(
        qdev: tq.QuantumDevice,
        wires: Union[int, List[int]],
        observables: Union[op.Observable, List[op.Observable]],
):
    all_dims = np.arange(qdev.states.dim())
    if isinstance(wires, int):
        wires = [wires]
        observables = [observables]

    # rotation to the desired basis
    for wire, observable in zip(wires, observables):
        for rotation in observable.diagonalizing_gates():
            rotation(qdev, wires=wire)

    states = qdev.states
    # compute magnitude
    state_mag = torch.abs(states) ** 2

    expectations = []
    for wire, observable in zip(wires, observables):
        # compute marginal magnitude
        reduction_dims = np.delete(all_dims, [0, wire + 1])
        if reduction_dims.size == 0:
            probs = state_mag
        else:
            probs = state_mag.sum(list(reduction_dims))
        res = probs.mv(observable.eigvals.real.to(probs.device))
        expectations.append(res)

    return torch.stack(expectations, dim=-1)


class MeasureAll(tq.QuantumModule):
    """Obtain the expectation value of all the qubits."""

    def __init__(self, obs, v_c_reg_mapping=None):
        super().__init__()
        self.obs = obs
        self.v_c_reg_mapping = v_c_reg_mapping

    def forward(self, qdev: tq.QuantumDevice):
        x = expval(qdev, list(range(qdev.n_wires)), [self.obs()] * qdev.n_wires)

        if self.v_c_reg_mapping is not None:
            c2v_mapping = self.v_c_reg_mapping["c2v"]
            """
            the measurement is not normal order, need permutation
            """
            perm = []
            for k in range(x.shape[-1]):
                if k in c2v_mapping.keys():
                    perm.append(c2v_mapping[k])
            x = x[:, perm]

        if self.noise_model_tq is not None and self.noise_model_tq.is_add_noise:
            return self.noise_model_tq.apply_readout_error(x)
        else:
            return x

    def set_v_c_reg_mapping(self, mapping):
        self.v_c_reg_mapping = mapping


class MeasureMultipleTimes(tq.QuantumModule):
    """
    obs list:
    list of dict: example
    [{'wires': [0, 2, 3, 1], 'observables': ['x', 'y', 'z', 'i']
    },
    {'wires': [0, 2, 3, 1], 'observables': ['x', 'y', 'z', 'i']
    },
    ]
    """

    def __init__(self, obs_list, v_c_reg_mapping=None):
        super().__init__()
        self.obs_list = obs_list
        self.v_c_reg_mapping = v_c_reg_mapping

    def forward(self, qdev: tq.QuantumDevice):
        res_all = []

        for layer in self.obs_list:
            # create a new q device for each time of measurement
            qdev_new = tq.QuantumDevice(n_wires=qdev.n_wires)
            qdev_new.clone_states(existing_states=qdev.states)
            qdev_new.state = qdev.state

            observables = []
            for wire in range(qdev.n_wires):
                observables.append(tq.I())

            for wire, observable in zip(layer["wires"], layer["observables"]):
                observables[wire] = tq.op_name_dict[observable]()

            res = expval(
                qdev_new,
                wires=list(range(qdev.n_wires)),
                observables=observables,
            )

            if self.v_c_reg_mapping is not None:
                c2v_mapping = self.v_c_reg_mapping["c2v"]
                """
                the measurement is not normal order, need permutation
                """
                perm = []
                for k in range(res.shape[-1]):
                    if k in c2v_mapping.keys():
                        perm.append(c2v_mapping[k])
                res = res[:, perm]
            res_all.append(res)

        return torch.cat(res_all)

    def set_v_c_reg_mapping(self, mapping):
        self.v_c_reg_mapping = mapping


class MeasureMultiPauliSum(tq.QuantumModule):
    """
    similar to qiskit.opflow PauliSumOp
    obs list:
    list of dict: example
    [{'wires': [0, 2, 3, 1],
    'observables': ['x', 'y', 'z', 'i'],
    'coefficient': [1, 0.5, 0.4, 0.3]
    },
    {'wires': [0, 2, 3, 1],
    'observables': ['x', 'y', 'z', 'i'],
    'coefficient': [1, 0.5, 0.4, 0.3]
    },
    ]
    """

    def __init__(self, obs_list, v_c_reg_mapping=None):
        super().__init__()
        self.obs_list = obs_list
        self.v_c_reg_mapping = v_c_reg_mapping
        self.measure_multiple_times = MeasureMultipleTimes(
            obs_list=obs_list, v_c_reg_mapping=v_c_reg_mapping
        )

    def forward(self, qdev: tq.QuantumDevice):
        res_all = self.measure_multiple_times(qdev)

        return res_all.sum(-1)


class MeasureMultiQubitPauliSum(tq.QuantumModule):
    """obs list:
    list of dict: example
    [{'coefficient': [0.5, 0.2]},
    {'wires': [0, 2, 3, 1],
    'observables': ['x', 'y', 'z', 'i'],
    },
    {'wires': [0, 2, 3, 1],
    'observables': ['y', 'x', 'z', 'i'],
    },
    ]
    Measures 0.5 * <x y z i> + 0.2 * <y x z i>
    """

    def __init__(self, obs_list, v_c_reg_mapping=None):
        super().__init__()
        self.obs_list = obs_list
        self.v_c_reg_mapping = v_c_reg_mapping
        self.measure_multiple_times = MeasureMultipleTimes(
            obs_list=obs_list[1:], v_c_reg_mapping=v_c_reg_mapping
        )

    def forward(self, qdev: tq.QuantumDevice):
        res_all = self.measure_multiple_times(qdev)
        return (res_all * self.obs_list[0]["coefficient"]).sum(-1)


if __name__ == '__main__':
    import pdb

    pdb.set_trace()
    qdev = tq.QuantumDevice(n_wires=2, bsz=5, device="cpu", record_op=True)  # use device='cuda' for GPU
    qdev.h(wires=0)
    qdev.cnot(wires=[0, 1])
    tqf.h(qdev, wires=1)
    tqf.x(qdev, wires=1)
    op = tq.RX(has_params=True, trainable=True, init_params=0.5)
    op(qdev, wires=0)

    # measure the state on z basis
    print(tq.measure(qdev, n_shots=1024))

    # obtain the expval on a observable
    expval = expval_joint_sampling(qdev, 'II', 100000)
    expval_ana = expval_joint_analytical(qdev, 'II')
    print(expval, expval_ana)
