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

"""
author: Vivek Yanamadula @Vivekyy
"""

import torch
import torch.nn.functional as F

import torchquantum as tq

from torchquantum.dataset import MNIST
from torchquantum.operator import op_name_dict
from typing import List


class TQNet(tq.QuantumModule):
    def __init__(self, layers: List[tq.QuantumModule], encoder=None, use_softmax=False):
        super().__init__()

        self.encoder = encoder
        self.use_softmax = use_softmax

        self.layers = tq.QuantumModuleList()

        for layer in layers:
            self.layers.append(layer)

        self.service = "TorchQuantum"
        self.measure = tq.MeasureAll_density(tq.PauliZ)

    def forward(self, device, x):
        bsz = x.shape[0]
        device.reset_states(bsz)

        x = F.avg_pool2d(x, 6)
        x = x.view(bsz, 16)

        if self.encoder:
            self.encoder(device, x)

        for layer in self.layers:
            layer(device)

        meas = self.measure(device)

        if self.use_softmax:
            meas = F.log_softmax(meas, dim=1)

        return meas


class TQLayer(tq.QuantumModule):
    def __init__(self, gates: List[tq.QuantumModule]):
        super().__init__()

        self.service = "TorchQuantum"

        self.layer = tq.QuantumModuleList()
        for gate in gates:
            self.layer.append(gate)

    @tq.static_support
    def forward(self, q_device):
        for gate in self.layer:
            gate(q_device)


def train_tq(model, device, train_dl, epochs, loss_fn, optimizer):
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0
        for batch_dict in train_dl:
            x = batch_dict["image"]
            y = batch_dict["digit"]

            y = y.to(torch.long)

            x = x.to(torch_device)
            y = y.to(torch_device)

            optimizer.zero_grad()

            preds = model(device, x)

            loss = loss_fn(preds, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            batches += 1

            print(f"Epoch {epoch + 1} | Loss: {running_loss/batches}", end="\r")

        print(f"Epoch {epoch + 1} | Loss: {running_loss/batches}")
        losses.append(running_loss / batches)

    return losses


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# encoder = None
# encoder = tq.AmplitudeEncoder()
encoder = tq.MultiPhaseEncoder(["u3", "u3", "u3", "u3"])


random_layer = tq.RandomLayer(n_ops=50, wires=list(range(4)))
trainable_layer = [
    op_name_dict["rx"](trainable=True, has_params=True, wires=[0]),
    op_name_dict["ry"](trainable=True, has_params=True, wires=[1]),
    op_name_dict["rz"](trainable=True, has_params=True, wires=[3]),
    op_name_dict["crx"](trainable=True, has_params=True, wires=[0, 2]),
]
trainable_layer = TQLayer(trainable_layer)
layers = [random_layer, trainable_layer]

device = tq.NoiseDevice(n_wires=4,
                        noise_model=tq.NoiseModel(kraus_dict={"Bitflip": 0.08, "Phaseflip": 0.08})).to(torch_device)

model = TQNet(layers=layers, encoder=encoder, use_softmax=True).to(torch_device)

loss_fn = F.nll_loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

dataset = MNIST(
    root="./mnist_data",
    train_valid_split_ratio=[0.9, 0.1],
    digits_of_interest=[0, 1, 3, 6],
    n_test_samples=200,
)

train_dl = torch.utils.data.DataLoader(
    dataset["train"],
    batch_size=32,
    sampler=torch.utils.data.RandomSampler(dataset["train"]),
)
val_dl = torch.utils.data.DataLoader(
    dataset["valid"],
    batch_size=32,
    sampler=torch.utils.data.RandomSampler(dataset["valid"]),
)
test_dl = torch.utils.data.DataLoader(
    dataset["test"],
    batch_size=32,
    sampler=torch.utils.data.RandomSampler(dataset["test"]),
)

print("--Training--")
train_losses = train_tq(model, device, train_dl, 1, loss_fn, optimizer)

