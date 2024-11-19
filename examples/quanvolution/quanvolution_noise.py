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

import torchquantum as tq
import torchquantum.functional as tqf

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from torchquantum.dataset import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR


class QuanvolutionFilter(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll_density(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        qdev = tq.NoiseDevice(self.n_wires, bsz=bsz, device=x.device,
                                noise_model=tq.NoiseModel(kraus_dict={"Bitflip": 0.22, "Phaseflip": 0.22}))
        size = 28
        x = x.view(bsz, size, size)

        data_list = []

        for c in range(0, size, 2):
            for r in range(0, size, 2):
                data = torch.transpose(
                    torch.cat(
                        (x[:, c, r], x[:, c, r + 1], x[:, c + 1, r], x[:, c + 1, r + 1])
                    ).view(4, bsz),
                    0,
                    1,
                )
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        qdev, self.encoder, self.q_layer, self.measure, data
                    )
                else:
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    data = self.measure(qdev)

                data_list.append(data.view(bsz, 4))

        result = torch.cat(data_list, dim=1).float()

        return result


class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qf = QuanvolutionFilter()
        self.linear = torch.nn.Linear(4 * 14 * 14, 10)

    def forward(self, x, use_qiskit=False):
        with torch.no_grad():
            x = self.qf(x, use_qiskit)
        x = self.linear(x)
        return F.log_softmax(x, -1)


class HybridModel_without_qf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 10)

    def forward(self, x, use_qiskit=False):
        x = x.view(-1, 28 * 28)
        x = self.linear(x)
        return F.log_softmax(x, -1)


def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow["train"]:
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end="\r")


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict["image"].to(device)
            targets = feed_dict["digit"].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")

    return accuracy, loss


def main():
    train_model_without_qf = True
    n_epochs = 15

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    dataset = MNIST(
        root="./mnist_data",
        train_valid_split_ratio=[0.9, 0.1],
        n_test_samples=300,
        n_train_samples=500,
    )
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=10,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = HybridModel().to(device)
    model_without_qf = HybridModel_without_qf().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    accu_list1 = []
    loss_list1 = []
    accu_list2 = []
    loss_list2 = []
    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer)
        print(optimizer.param_groups[0]["lr"])

        # valid
        accu, loss = valid_test(
            dataflow,
            "test",
            model,
            device,
        )
        accu_list1.append(accu)
        loss_list1.append(loss)
        scheduler.step()

    if train_model_without_qf:
        optimizer = optim.Adam(
            model_without_qf.parameters(), lr=5e-3, weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
        for epoch in range(1, n_epochs + 1):
            # train
            print(f"Epoch {epoch}:")
            train(dataflow, model_without_qf, device, optimizer)
            print(optimizer.param_groups[0]["lr"])

            # valid
            accu, loss = valid_test(dataflow, "test", model_without_qf, device)
            accu_list2.append(accu)
            loss_list2.append(loss)

            scheduler.step()

    # run on real QC
    try:
        from qiskit import IBMQ
        from torchquantum.plugin import QiskitProcessor

        # firstly perform simulate
        print(f"\nTest with Qiskit Simulator")
        processor_simulation = QiskitProcessor(use_real_qc=False)
        model.qf.set_qiskit_processor(processor_simulation)
        valid_test(dataflow, "test", model, device, qiskit=True)
        # then try to run on REAL QC
        backend_name = "ibmq_quito"
        print(f"\nTest on Real Quantum Computer {backend_name}")
        processor_real_qc = QiskitProcessor(use_real_qc=True, backend_name=backend_name)
        model.qf.set_qiskit_processor(processor_real_qc)
        valid_test(dataflow, "test", model, device, qiskit=True)
    except ImportError:
        print(
            "Please install qiskit, create an IBM Q Experience Account and "
            "save the account token according to the instruction at "
            "'https://github.com/Qiskit/qiskit-ibmq-provider', "
            "then try again."
        )


if __name__ == "__main__":
    main()
