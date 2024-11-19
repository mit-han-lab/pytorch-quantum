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
import torch
from torchquantum.util.vqe_utils import parse_hamiltonian_file
import random
import numpy as np
import argparse
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchquantum.measurement import expval_joint_analytical_density


class QVQEModel(tq.QuantumModule):
    def __init__(self, arch, hamil_info):
        super().__init__()
        self.arch = arch
        self.hamil_info = hamil_info
        self.n_wires = hamil_info["n_wires"]
        self.n_blocks = arch["n_blocks"]
        self.u3_layers = tq.QuantumModuleList()
        self.cu3_layers = tq.QuantumModuleList()
        for _ in range(self.n_blocks):
            self.u3_layers.append(
                tq.Op1QAllLayer(
                    op=tq.U3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.cu3_layers.append(
                tq.Op2QAllLayer(
                    op=tq.CU3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    circular=True,
                )
            )

    def forward(self):
        qdev = tq.NoiseDevice(
            n_wires=self.n_wires, bsz=1, device=next(self.parameters()).device,
            noise_model=tq.NoiseModel(kraus_dict={"Bitflip": 0.22, "Phaseflip": 0.22})
        )

        for k in range(self.n_blocks):
            self.u3_layers[k](qdev)
            self.cu3_layers[k](qdev)

        expval = 0
        for hamil in self.hamil_info["hamil_list"]:
            expval += (
                expval_joint_analytical_density(qdev, observable=hamil["pauli_string"])
                * hamil["coeff"]
            )

        return expval


def train(model, optimizer, n_steps=1):
    for _ in range(n_steps):
        loss = model()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Expectation of energy: {loss.item()}")


def valid_test(model):
    with torch.no_grad():
        loss = model()

    print(f"validation: expectation of energy: {loss.item()}")


def process_hamil_info(hamil_info):
    hamil_list = hamil_info["hamil_list"]
    n_wires = hamil_info["n_wires"]
    all_info = []

    for hamil in hamil_list:
        pauli_string = ""
        for i in range(n_wires):
            if i in hamil["wires"]:
                wire = hamil["wires"].index(i)
                pauli_string += hamil["observables"][wire].upper()
            else:
                pauli_string += "I"
        all_info.append({"pauli_string": pauli_string, "coeff": hamil["coefficient"]})
    hamil_info["hamil_list"] = all_info
    return hamil_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="debug with pdb")
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=2,
        help="number of blocks, each contain one layer of "
        "U3 gates and one layer of CU3 with "
        "ring connections",
    )
    parser.add_argument(
        "--steps_per_epoch", type=int, default=10, help="number of training epochs"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--hamil_filename",
        type=str,
        default="h2.txt",
        help="number of training epochs",
    )

    args = parser.parse_args()

    if args.pdb:
        import pdb

        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hamil_info = process_hamil_info(parse_hamiltonian_file(args.hamil_filename))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = QVQEModel(arch={"n_blocks": args.n_blocks}, hamil_info=hamil_info)

    model.to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        train(model, optimizer, n_steps=args.steps_per_epoch)

        scheduler.step()

    # final valid
    valid_test(model)


if __name__ == "__main__":
    main()
