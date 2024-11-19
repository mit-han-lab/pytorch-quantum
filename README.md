<p align="center">
<img src="https://github.com/mit-han-lab/torchquantum/blob/main/torchquantum_logo.jpg?raw=true" alt="torchquantum Logo" width="450">
</p>

<h2><p align="center">Quantum Computing in PyTorch</p></h2>
<h3><p align="center">Faster, Scalable, Easy Debugging, Easy Deployment on Real Machine</p></h3>


<p align="center">
    <a href="https://torchquantum.readthedocs.io/">
        <img alt="Documentation" src="https://img.shields.io/readthedocs/torchquantum/main">
    </a>
    <a href="https://github.com/mit-han-lab/torchquantum/blob/master/LICENSE">
        <img alt="MIT License" src="https://img.shields.io/github/license/mit-han-lab/torchquantum">
    </a>
    <a href="https://join.slack.com/t/torchquantum/shared_invite/zt-1ghuf283a-OtP4mCPJREd~367VX~TaQQ">
        <img alt="Chat @ Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://discord.gg/VTHZAB5E">
        <img alt="Chat @ Discord" src="https://img.shields.io/badge/contact-me-blue?logo=discord&logoColor=white">
    </a>
<!--     <a href="https://qmlsys.hanruiwang.me">
        <img alt="Forum" src="https://img.shields.io/discourse/status?server=https%3A%2F%2Fqmlsys.hanruiwang.me%2F">
    </a> -->
    <a href="https://qmlsys.mit.edu">
        <img alt="Website" src="https://img.shields.io/website?up_message=qmlsys&url=https%3A%2F%2Fqmlsys.mit.edu">
    </a>
   <a href="https://pypi.org/project/torchquantum/">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/torchquantum">
    </a>
       <a href="https://unitary.fund/">
        <img alt="Pypi" src="https://img.shields.io/badge/supported%20by-Unitary%20Fund-green">
    </a>
   </a>
       <a href="https://pytorch.org/ecosystem/">
        <img alt="Pypi" src="https://img.shields.io/badge/integration%20-PyTorch%20Ecosystem-blue">
    </a>
    </a>
       <a href="https://qiskit.org/ecosystem/">
        <img alt="Pypi" src="https://img.shields.io/badge/integration%20-Qiskit%20Ecosystem-blue">
    </a>

</p>
<br />



# 👋 Welcome

#### What it is doing

Simulate quantum computations on classical hardware using PyTorch. It supports statevector simulation and pulse simulation on GPUs. It can scale up to the simulation of 30+ qubits with multiple GPUs.
#### Who will benefit

Researchers on quantum algorithm design, parameterized quantum circuit training, quantum optimal control, quantum machine learning, quantum neural networks.
#### Differences from Qiskit/Pennylane

Dynamic computation graph, automatic gradient computation, fast GPU support, batch model tensorized processing.

## News
- Torchquantum is used in the winning team for ACM Quantum Computing for Drug Discovery Challenge.
- Torchquantum is highlighted in [UnitaryHack](https://unitaryhack.dev/projects/torchquantum/).
- TorchQuantum received [UnitaryFund](https://unitary.fund/).
- TorchQuantum is integrated to [IBM Qiskit Ecosystem](https://qiskit.github.io/ecosystem/).
- TorchQuantum is integrated to [PyTorch Ecosystem](https://pytorch.org/ecosystem/).
- v0.1.8 Available!
- Check the [dev branch](https://github.com/mit-han-lab/torchquantum/tree/dev) for new latest features on quantum layers and quantum algorithms.
- Join our [Slack](https://join.slack.com/t/torchquantum/shared_invite/zt-1ghuf283a-OtP4mCPJREd~367VX~TaQQ) for real time support!
- Welcome to contribute! Please contact us or post in the Github Issues if you want to have new examples implemented by TorchQuantum or any other questions.
- Qmlsys website goes online: [qmlsys.mit.edu](https://qmlsys.mit.edu) and [torchquantum.org](https://torchquantum.org)

## Features

- Easy construction and simulation of quantum circuits in **PyTorch**
- **Dynamic computation graph** for easy debugging
- **Gradient support** via autograd
- **Batch mode** inference and training on **CPU/GPU**
- Easy **deployment on real quantum devices** such as IBMQ
- **Easy hybrid classical-quantum** model construction
- (coming soon) **pulse-level simulation**


## Installation
```bash
git clone https://github.com/mit-han-lab/torchquantum.git
cd torchquantum
pip install --editable .
```

## Basic Usage

```python
import torchquantum as tq
import torchquantum.functional as tqf

qdev = tq.QuantumDevice(n_wires=2, bsz=5, device="cpu", record_op=True) # use device='cuda' for GPU

# use qdev.op
qdev.h(wires=0)
qdev.cnot(wires=[0, 1])

# use tqf
tqf.h(qdev, wires=1)
tqf.x(qdev, wires=1)

# use tq.Operator
op = tq.RX(has_params=True, trainable=True, init_params=0.5)
op(qdev, wires=0)

# print the current state (dynamic computation graph supported)
print(qdev)

# obtain the qasm string
from torchquantum.plugin import op_history2qasm
print(op_history2qasm(qdev.n_wires, qdev.op_history))

# measure the state on z basis
print(tq.measure(qdev, n_shots=1024))

# obtain the expval on a observable by stochastic sampling (doable on simulator and real quantum hardware)
from torchquantum.measurement import expval_joint_sampling
expval_sampling = expval_joint_sampling(qdev, 'ZX', n_shots=1024)
print(expval_sampling)

# obtain the expval on a observable by analytical computation (only doable on classical simulator)
from torchquantum.measurement import expval_joint_analytical
expval = expval_joint_analytical(qdev, 'ZX')
print(expval)

# obtain gradients of expval w.r.t. trainable parameters
expval[0].backward()
print(op.params.grad)


# Apply gates to qdev with tq.QuantumModule
ops = [
    {'name': 'hadamard', 'wires': 0}, 
    {'name': 'cnot', 'wires': [0, 1]},
    {'name': 'rx', 'wires': 0, 'params': 0.5, 'trainable': True},
    {'name': 'u3', 'wires': 0, 'params': [0.1, 0.2, 0.3], 'trainable': True},
    {'name': 'h', 'wires': 1, 'inverse': True}
]

qmodule = tq.QuantumModule.from_op_history(ops)
qmodule(qdev)
```


<!--
## Basic Usage 2

```python
import torchquantum as tq
import torchquantum.functional as tqf

x = tq.QuantumDevice(n_wires=2)

tqf.hadamard(x, wires=0)
tqf.x(x, wires=1)
tqf.cnot(x, wires=[0, 1])

# print the current state (dynamic computation graph supported)
print(x.states)

# obtain the classical bitstring distribution
print(tq.measure(x, n_shots=2048))
```
 -->


## Guide to the examples

We also prepare many example and tutorials using TorchQuantum.

For **beginning level**, you may check [QNN for MNIST](examples/mnist), [Quantum Convolution (Quanvolution)](examples/quanvolution) and [Quantum Kernel Method](examples/quantum_kernel_method), and [Quantum Regression](examples/regression).

For **intermediate level**, you may check [Amplitude Encoding for MNIST](examples/amplitude_encoding_mnist), [Clifford gate QNN](examples/clifford_qnn), [Save and Load QNN models](examples/save_load_example), [PauliSum Operation](examples/PauliSumOp), [How to convert tq to Qiskit](examples/converter_tq_qiskit).

For **expert**, you may check [Parameter Shift on-chip Training](examples/param_shift_onchip_training), [VQA Gradient Pruning](examples/gradient_pruning), [VQE](examples/vqe),  [VQA for State Prepration](examples/train_state_prep), [QAOA (Quantum Approximate Optimization Algorithm)](examples/qaoa).


## Usage

Construct parameterized quantum circuit models as simple as constructing a normal pytorch model.
```python
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.n_wires = 4
    self.measure = tq.MeasureAll(tq.PauliZ)

    self.encoder_gates = [tqf.rx] * 4 + [tqf.ry] * 4 + \
                         [tqf.rz] * 4 + [tqf.rx] * 4
    self.rx0 = tq.RX(has_params=True, trainable=True)
    self.ry0 = tq.RY(has_params=True, trainable=True)
    self.rz0 = tq.RZ(has_params=True, trainable=True)
    self.crx0 = tq.CRX(has_params=True, trainable=True)

  def forward(self, x):
    bsz = x.shape[0]
    # down-sample the image
    x = F.avg_pool2d(x, 6).view(bsz, 16)

    # create a quantum device to run the gates
    qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

    # encode the classical image to quantum domain
    for k, gate in enumerate(self.encoder_gates):
      gate(qdev, wires=k % self.n_wires, params=x[:, k])

    # add some trainable gates (need to instantiate ahead of time)
    self.rx0(qdev, wires=0)
    self.ry0(qdev, wires=1)
    self.rz0(qdev, wires=3)
    self.crx0(qdev, wires=[0, 2])

    # add some more non-parameterized gates (add on-the-fly)
    qdev.h(wires=3)
    qdev.sx(wires=2)
    qdev.cnot(wires=[3, 0])
    qdev.qubitunitary(wires=[1, 2], params=[[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 0, 1j],
                                            [0, 0, -1j, 0]])

    # perform measurement to get expectations (back to classical domain)
    x = self.measure(qdev).reshape(bsz, 2, 2)

    # classification
    x = x.sum(-1).squeeze()
    x = F.log_softmax(x, dim=1)

    return x

```

## VQE Example

Train a quantum circuit to perform VQE task.
Quito quantum computer as in [simple_vqe.py](./examples/simple_vqe/simple_vqe.py)
script:
```python
cd examples/vqe
python vqe.py
```

## MNIST Example

Train a quantum circuit to perform MNIST classification task and deploy on the real IBM
Quito quantum computer as in [mnist_example.py](./examples/simple_mnist/mnist_example_no_binding.py)
script:
```python
cd examples/mnist
python mnist.py
```

## Files

| File      | Description |
| ----------- | ----------- |
| devices.py      | QuantumDevice class which stores the statevector |
| encoding.py   | Encoding layers to encode classical values to quantum domain |
| functional.py   | Quantum gate functions |
| operators.py   | Quantum gate classes |
| layers.py   | Layer templates such as RandomLayer |
| measure.py   | Measurement of quantum states to get classical values |
| graph.py   | Quantum gate graph used in static mode |
| super_layer.py   | Layer templates for SuperCircuits |
| plugins/qiskit*   | Convertors and processors for easy deployment on IBMQ |
| examples/| More examples for training QML and VQE models |

## Coding Style

torchquantum uses pre-commit hooks to ensure Python style consistency and prevent common mistakes in its codebase.

To enable it pre-commit hooks please reproduce:
```bash
pip install pre-commit
pre-commit install
```


[comment]: <> (## More Examples)

[comment]: <> (The `examples/` folder contains more examples to train the QML and VQE)

[comment]: <> (models. Example usage for a QML circuit:)

[comment]: <> (```python)

[comment]: <> (# train the circuit with 36 params in the U3+CU3 space)

[comment]: <> (python examples/train.py examples/configs/mnist/four0123/train/baseline/u3cu3_s0/rand/param36.yml)

[comment]: <> (# evaluate the circuit with torchquantum)

[comment]: <> (python examples/eval.py examples/configs/mnist/four0123/eval/tq/all.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36)

[comment]: <> (# evaluate the circuit with real IBMQ-Yorktown quantum computer)

[comment]: <> (python examples/eval.py examples/configs/mnist/four0123/eval/x2/real/opt2/300.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36)

[comment]: <> (```)

[comment]: <> (Example usage for a VQE circuit:)

[comment]: <> (```python)

[comment]: <> (# Train the VQE circuit for h2)

[comment]: <> (python examples/train.py examples/configs/vqe/h2/train/baseline/u3cu3_s0/human/param12.yml)

[comment]: <> (# evaluate the VQE circuit with torchquantum)

[comment]: <> (python examples/eval.py examples/configs/vqe/h2/eval/tq/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/)

[comment]: <> (# evaluate the VQE circuit with real IBMQ-Yorktown quantum computer)

[comment]: <> (python examples/eval.py examples/configs/vqe/h2/eval/x2/real/opt2/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/)

[comment]: <> (```)

[comment]: <> (Detailed documentations coming soon.)

[comment]: <> (## QuantumNAS)

[comment]: <> (Quantum noise is the key challenge in Noisy Intermediate-Scale Quantum &#40;NISQ&#41; computers. Previous work for mitigating noise has primarily focused on gate-level or pulse-level noise-adaptive compilation. However, limited research efforts have explored a higher level of optimization by making the quantum circuits themselves resilient to noise. We propose QuantumNAS, a comprehensive framework for noise-adaptive co-search of the variational circuit and qubit mapping. Variational quantum circuits are a promising approach for constructing QML and quantum simulation. However, finding the best variational circuit and its optimal parameters is challenging due to the large design space and parameter training cost. We propose to decouple the circuit search and parameter training by introducing a novel SuperCircuit. The SuperCircuit is constructed with multiple layers of pre-defined parameterized gates and trained by iteratively sampling and updating the parameter subsets &#40;SubCircuits&#41; of it. It provides an accurate estimation of SubCircuits performance trained from scratch. Then we perform an evolutionary co-search of SubCircuit and its qubit mapping. The SubCircuit performance is estimated with parameters inherited from SuperCircuit and simulated with real device noise models. Finally, we perform iterative gate pruning and finetuning to remove redundant gates. Extensively evaluated with 12 QML and VQE benchmarks on 10 quantum comput, QuantumNAS significantly outperforms baselines. For QML, QuantumNAS is the first to demonstrate over 95% 2-class, 85% 4-class, and 32% 10-class classification accuracy on real QC. It also achieves the lowest eigenvalue for VQE tasks on H2, H2O, LiH, CH4, BeH2 compared with UCCSD. We also open-source torchquantum for fast training of parameterized quantum circuits to facilitate future research.)

[comment]: <> (<p align="center">)

[comment]: <> (<img src="https://hanruiwang.me/project_pages/quantumnas/assets/teaser.jpg" alt="torchquantum teaser" width="550">)

[comment]: <> (</p>)

[comment]: <> (QuantumNAS Framework overview:)

[comment]: <> (<p align="center">)

[comment]: <> (<img src="https://hanruiwang.me/project_pages/quantumnas/assets/overview.jpg" alt="torchquantum overview" width="1000">)

[comment]: <> (</p>)

[comment]: <> (QuantumNAS models achieve higher robustness and accuracy than other baseline models:)

[comment]: <> (<p align="center">)

[comment]: <> (<img src="https://hanruiwang.me/project_pages/quantumnas/assets/results.jpg" alt="torchquantum results" width="550">)

[comment]: <> (</p>)

## Papers using TorchQuantum

- [HPCA'22] [Wang et al., "QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits"](https://arxiv.org/abs/2107.10845)
- [DAC'22] [Wang et al., "QuantumNAT: Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization"](https://arxiv.org/abs/2110.11331)
- [DAC'22] [Wang et al., "QOC: Quantum On-Chip Training with Parameter Shift and Gradient Pruning"](https://arxiv.org/abs/2202.13239)
- [QCE'22] [Liang et al., "Variational Quantum Pulse Learning"](https://arxiv.org/abs/2203.17267)
- [ICCAD'22] [Hu et al., "Quantum Neural Network Compression"](https://arxiv.org/abs/2207.01578)
- [ICCAD'22] [Wang et al., "QuEst: Graph Transformer for Quantum Circuit Reliability Estimation"](https://arxiv.org/abs/2210.16724)
- [ICML Workshop] [Yun et al., "Slimmable Quantum Federated Learning"](https://dynn-icml2022.github.io/spapers/paper_7.pdf)
- [IEEE ICDCS] [Yun et al., "Quantum Multi-Agent Reinforcement Learning via Variational Quantum Circuit Design"](https://ieeexplore.ieee.org/document/9912289)
- [QCE'23] [Zhan et al., "Quantum Sensor Network Algorithms for Transmitter Localization"](https://ieeexplore.ieee.org/abstract/document/10313806)
<details>
  <summary>Manuscripts</summary>

  ## Manuscripts

  - [Yun et al., "Projection Valued Measure-based Quantum Machine Learning for Multi-Class Classification"](https://arxiv.org/abs/2210.16731)
  - [Baek et al., "3D Scalable Quantum Convolutional Neural Networks for Point Cloud Data Processing in Classification Applications"](https://arxiv.org/abs/2210.09728)
  - [Baek et al., "Scalable Quantum Convolutional Neural Networks"](https://arxiv.org/abs/2209.12372)
  - [Yun et al., "Quantum Multi-Agent Meta Reinforcement Learning"](https://arxiv.org/abs/2208.11510)

</details>



## Dependencies

- 3.9 >= Python >= 3.7 (Python 3.10 may have the `concurrent` package issue for Qiskit)
- PyTorch >= 1.8.0
- configargparse >= 0.14
- GPU model training requires NVIDIA GPUs

## Contact

TorchQuantum [Forum](https://qmlsys.hanruiwang.me)

Hanrui Wang [hanrui@mit.edu](mailto:hanrui@mit.edu)

## Contributors

Jiannan Cao, Jessica Ding, Jiai Gu, Song Han, Zhirui Hu, Zirui Li, Zhiding Liang, Pengyu Liu, Yilian Liu, Mohammadreza Tavasoli, Hanrui Wang, Zhepeng Wang, Zhuoyang Ye

## Citation
```
@inproceedings{hanruiwang2022quantumnas,
    title     = {Quantumnas: Noise-adaptive search for robust quantum circuits},
    author    = {Wang, Hanrui and Ding, Yongshan and Gu, Jiaqi and Li, Zirui and Lin, Yujun and Pan, David Z and Chong, Frederic T and Han, Song},
    booktitle = {The 28th IEEE International Symposium on High-Performance Computer Architecture (HPCA-28)},
    year      = {2022}
}
```
