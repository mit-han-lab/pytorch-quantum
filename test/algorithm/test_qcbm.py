from torchquantum.algorithm.qcbm import QCBM, MMDLoss
import torchquantum as tq
import torch


def test_qcbm_forward():
    n_wires = 3
    n_layers = 3
    ops = []
    for l in range(n_layers):
        for q in range(n_wires):
            ops.append({"name": "rx", "wires": q, "params": 0.0, "trainable": True})
        for q in range(n_wires - 1):
            ops.append({"name": "cnot", "wires": [q, q + 1]})

    data = torch.ones(2**n_wires)
    qmodule = tq.QuantumModule.from_op_history(ops)
    qcbm = QCBM(n_wires, qmodule)
    probs = qcbm()
    expected = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
    assert torch.allclose(probs, expected)


def test_mmd_loss():
    n_wires = 2
    bandwidth = torch.tensor([0.1, 1.0])
    space = torch.arange(2**n_wires)

    mmd = MMDLoss(bandwidth, space)
    loss = mmd(torch.zeros(4), torch.zeros(4))
    assert torch.isclose(loss, torch.tensor(0.0), rtol=1e-5)
