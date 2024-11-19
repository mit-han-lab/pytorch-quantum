import torch
import torch.nn as nn

import torchquantum as tq

__all__ = ["QCBM", "MMDLoss"]


class MMDLoss(nn.Module):
    """Squared maximum mean discrepancy with radial basis function kerne"""

    def __init__(self, scales, space):
        """
        Initialize MMDLoss object. Calculates and stores the kernel matrix.

        Args:
            scales: Bandwidth parameters.
            space: Basis input space.
        """
        super().__init__()

        gammas = 1 / (2 * (scales**2))

        # squared Euclidean distance
        sq_dists = torch.abs(space[:, None] - space[None, :]) ** 2

        # Kernel matrix
        self.K = sum(torch.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
        self.scales = scales

    def k_expval(self, px, py):
        """
        Kernel expectation value

        Args:
            px: First probability distribution
            py: Second probability distribution

        Returns:
            Expectation value of the RBF Kernel.
        """

        return px @ self.K @ py

    def forward(self, px, py):
        """
        Squared MMD loss.

        Args:
            px: First probability distribution
            py: Second probability distribution

        Returns:
            Squared MMD loss.
        """
        pxy = px - py
        return self.k_expval(pxy, pxy)


class QCBM(nn.Module):
    """
    Quantum Circuit Born Machine (QCBM)

    Attributes:
        ansatz: An Ansatz object
        n_wires: Number of wires in the ansatz used.

    Methods:
        __init__: Initialize the QCBM object.
        forward: Returns the probability distribution (output from measurement).
    """

    def __init__(self, n_wires, ansatz):
        """
        Initialize QCBM object

        Args:
            ansatz (Ansatz): An Ansatz object
            n_wires (int): Number of wires in the ansatz used.
        """
        super().__init__()

        self.ansatz = ansatz
        self.n_wires = n_wires

    def forward(self):
        """
        Execute and obtain the probability distribution

        Returns:
            Probabilities (torch.Tensor)
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device="cpu")
        self.ansatz(qdev)
        probs = torch.abs(qdev.states.flatten()) ** 2
        return probs
