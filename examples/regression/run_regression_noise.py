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

import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import torchquantum as tq

from torch.optim.lr_scheduler import CosineAnnealingLR

import random
import numpy as np

# data is cos(theta)|000> + e^(j * phi)sin(theta) |111>

from torchpack.datasets.dataset import Dataset


def gen_data(L, N):
    omega_0 = np.zeros([2 ** L], dtype="complex_")
    omega_0[0] = 1 + 0j

    omega_1 = np.zeros([2 ** L], dtype="complex_")
    omega_1[-1] = 1 + 0j

    states = np.zeros([N, 2 ** L], dtype="complex_")

    thetas = 2 * np.pi * np.random.rand(N)
    phis = 2 * np.pi * np.random.rand(N)

    for i in range(N):
        states[i] = (
                np.cos(thetas[i]) * omega_0
                + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        )

    X = np.sin(2 * thetas) * np.cos(phis)

    return states, X


class RegressionDataset:
    def __init__(self, split, n_samples, n_wires):
        self.split = split
        self.n_samples = n_samples
        self.n_wires = n_wires

        self.states, self.Xlabel = gen_data(self.n_wires, self.n_samples)

    def __getitem__(self, index: int):
        instance = {"states": self.states[index], "Xlabel": self.Xlabel[index]}
        return instance

    def __len__(self) -> int:
        return self.n_samples


class Regression(Dataset):
    def __init__(self, n_train, n_valid, n_wires):
        n_samples_dict = {"train": n_train, "valid": n_valid}
        super().__init__(
            {
                split: RegressionDataset(
                    split=split, n_samples=n_samples_dict[split], n_wires=n_wires
                )
                for split in ["train", "valid"]
            }
        )


class QModel(tq.QuantumModule):
    def __init__(self, n_wires, n_blocks, add_fc=False):
        super().__init__()
        # inside one block, we have one u3 layer one each qubit and one layer
        # cu3 layer with ring connection
        self.n_wires = n_wires
        self.n_blocks = n_blocks
        self.u3_layers = tq.QuantumModuleList()
        self.cu3_layers = tq.QuantumModuleList()
        for _ in range(n_blocks):
            self.u3_layers.append(
                tq.Op1QAllLayer(
                    op=tq.U3,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.cu3_layers.append(
                tq.Op2QAllLayer(
                    op=tq.CU3,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                    circular=True,
                )
            )
        self.measure = tq.MeasureAll_density(tq.PauliZ)
        self.add_fc = add_fc
        if add_fc:
            self.fc_layer = torch.nn.Linear(n_wires, 1)

    def forward(self, input_states):
        qdev = tq.NoiseDevice(
            n_wires=self.n_wires, bsz=input_states.shape[0], device=input_states.device,
            noise_model=tq.NoiseModel(kraus_dict={"Bitflip": 0.22, "Phaseflip": 0.22}))
        # firstly set the qdev
        bsz = input_states.shape[0]
        input_states = torch.reshape(input_states, [bsz] + [2] * self.n_wires)

        qdev.clone_from_states(input_states)
        for k in range(self.n_blocks):
            self.u3_layers[k](qdev)
            self.cu3_layers[k](qdev)

        res = self.measure(qdev)
        if self.add_fc:
            res = self.fc_layer(res)
        else:
            res = res[:, 1]
        return res


def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow["train"]:
        inputs = feed_dict["states"].to(device).to(torch.complex64)
        targets = feed_dict["Xlabel"].to(device).to(torch.float)

        outputs = model(inputs)

        loss = F.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}")


def valid_test(dataflow, split, model, device):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict["states"].to(device).to(torch.complex64)
            targets = feed_dict["Xlabel"].to(device).to(torch.float)

            outputs = model(inputs)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    loss = F.mse_loss(output_all, target_all)

    print(f"{split} set loss: {loss}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="debug with pdb")
    parser.add_argument(
        "--bsz", type=int, default=32, help="batch size for training and validation"
    )
    parser.add_argument("--n_wires", type=int, default=3, help="number of qubits")
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=2,
        help="number of blocks, each contain one layer of "
             "U3 gates and one layer of CU3 with "
             "ring connections",
    )
    parser.add_argument(
        "--n_train", type=int, default=300, help="number of training samples"
    )
    parser.add_argument(
        "--n_valid", type=int, default=1000, help="number of validation samples"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--addfc", action="store_true", help="add a final classical FC layer"
    )

    args = parser.parse_args()

    if args.pdb:
        import pdb

        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = Regression(
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_wires=args.n_wires,
    )

    dataflow = dict()

    for split in dataset:
        if split == "train":
            sampler = torch.utils.data.RandomSampler(dataset[split])
        else:
            sampler = torch.utils.data.SequentialSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=args.bsz,
            sampler=sampler,
            num_workers=1,
            pin_memory=True,
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QModel(n_wires=args.n_wires, n_blocks=args.n_blocks, add_fc=args.addfc).to(
        device
    )

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        train(dataflow, model, device, optimizer)

        # valid
        valid_test(dataflow, "valid", model, device)
        scheduler.step()

    # final valid
    valid_test(dataflow, "valid", model, device)


if __name__ == "__main__":
    main()
