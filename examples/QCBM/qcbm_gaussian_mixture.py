import matplotlib.pyplot as plt
import numpy as np
import torch
from torchquantum.algorithm import QCBM, MMDLoss
import torchquantum as tq
import argparse
import os
from pprint import pprint


# Reproducibility
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_wires", type=int, default=6, help="Number of wires used in the circuit"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--n_blocks", type=int, default=6, help="Number of blocks in ansatz"
    )
    parser.add_argument(
        "--n_layers_per_block",
        type=int,
        default=1,
        help="Number of layers per block in ansatz",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Visualize the predicted probability distribution",
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer class from torch.optim"
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    return parser


# Function to create a gaussian mixture
def gaussian_mixture_pdf(x, mus, sigmas):
    mus, sigmas = np.array(mus), np.array(sigmas)
    vars = sigmas**2
    values = [
        (1 / np.sqrt(2 * np.pi * v)) * np.exp(-((x - m) ** 2) / (2 * v))
        for m, v in zip(mus, vars)
    ]
    values = np.sum([val / sum(val) for val in values], axis=0)
    return values / np.sum(values)


def main():
    set_seed()
    parser = _setup_parser()
    args = parser.parse_args()

    print("Configuration:")
    pprint(vars(args))

    # Create a gaussian mixture
    n_wires = args.n_wires
    assert n_wires >= 1, "Number of wires must be at least 1"

    x_max = 2**n_wires
    x_input = np.arange(x_max)
    mus = [(2 / 8) * x_max, (5 / 8) * x_max]
    sigmas = [x_max / 10] * 2
    data = gaussian_mixture_pdf(x_input, mus, sigmas)

    # This is the target distribution that the QCBM will learn
    target_probs = torch.tensor(data, dtype=torch.float32)

    # Ansatz
    layers = tq.RXYZCXLayer0(
        {
            "n_blocks": args.n_blocks,
            "n_wires": n_wires,
            "n_layers_per_block": args.n_layers_per_block,
        }
    )

    qcbm = QCBM(n_wires, layers)

    # To train QCBMs, we use MMDLoss with radial basis function kernel.
    bandwidth = torch.tensor([0.25, 60])
    space = torch.arange(2**n_wires)
    mmd = MMDLoss(bandwidth, space)

    # Optimization
    optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer = optimizer_class(qcbm.parameters(), lr=args.lr)

    for i in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred_probs = qcbm()
        loss = mmd(pred_probs, target_probs)
        loss.backward()
        optimizer.step()
        print(i, loss.item())

    # Visualize the results
    if args.plot:
        with torch.no_grad():
            pred_probs = qcbm()

        plt.plot(x_input, target_probs, linestyle="-.", label=r"$\pi(x)$")
        plt.bar(x_input, pred_probs, color="green", alpha=0.5, label="samples")
        plt.xlabel("Samples")
        plt.ylabel("Prob. Distribution")

        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
