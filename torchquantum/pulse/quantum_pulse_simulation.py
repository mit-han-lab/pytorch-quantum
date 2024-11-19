import torch
import torch.optim as optim
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumPulseDemo(tq.QuantumModule):
    """
    Quantum pulse demonstration module.

    This module defines a parameterized quantum pulse and applies it to a quantum device.
    """

    def __init__(self):
        """
        Initializes the QuantumPulseDemo module.

        Args:
            n_wires (int): Number of quantum wires (qubits).
            n_steps (int): Number of steps for the quantum pulse.
            hamil (list): Hamiltonian for the quantum pulse.
        """
        super().__init__()
        self.n_wires = 2

        # Quantum encoder
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'rx', 'wires': [0]},
            {'input_idx': [1], 'func': 'rx', 'wires': [1]}
        ])

        # Define parameterized quantum pulse
        self.pulse = tq.pulse.QuantumPulseDirect(n_steps=4, hamil=[[0, 1], [1, 0]])

    def forward(self, x):
        """
        Forward pass through the QuantumPulseDemo module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Measurement result from the quantum device.
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        self.apply_pulse(qdev)
        return tq.measure(qdev)

    def apply_pulse(self, qdev):
        """
        Applies the parameterized quantum pulse to the quantum device.

        Args:
            qdev (tq.QuantumDevice): Quantum device to apply the pulse to.
        """
        pulse_params = self.pulse.pulse_shape.detach().cpu().numpy()
        # Apply pulse to the quantum device (adjust based on actual pulse application logic)
        for params in pulse_params:
            tqf.rx(qdev, wires=0, params=params)
            tqf.rx(qdev, wires=1, params=params)

class OM_EOM_Simulation:
    """
    Optical modulation with electro-optic modulator (EOM) simulation.

    This class simulates a sequence of optical pulses with or without EOM modulation.
    """

    def __init__(self, pulse_duration, modulation_bandwidth=None, eom_mode=False):
        """
        Initializes the OM_EOM_Simulation.

        Args:
            pulse_duration (float): Duration of each pulse.
            modulation_bandwidth (float, optional): Bandwidth of modulation. Defaults to None.
            eom_mode (bool, optional): Whether to simulate EOM mode. Defaults to False.
        """
        self.pulse_duration = pulse_duration
        self.modulation_bandwidth = modulation_bandwidth
        self.eom_mode = eom_mode

    def simulate_sequence(self):
        """
        Simulates a sequence of optical pulses with specified parameters.

        Returns:
            list: Sequence of pulses and delays.
        """
        # Initialize the sequence
        sequence = []

        # Add pulses and delays to the sequence
        if self.modulation_bandwidth:
            # Apply modulation bandwidth effect
            sequence.append(('Delay', 0))
            sequence.append(('Pulse', 'NoisyChannel'))
        for _ in range(3):
            # Apply pulses with specified duration
            sequence.append(('Delay', self.pulse_duration))
            if self.eom_mode:
                # Apply EOM mode operation
                sequence.append(('Pulse', 'EOM'))
            else:
                # Apply regular pulse
                sequence.append(('Pulse', 'Regular'))
            # Apply a delay between pulses
            sequence.append(('Delay', 0))

        return sequence

class QuantumPulseDemoRunner:
    """
    Runner for training the QuantumPulseDemo model and simulating the OM_EOM_Simulation.
    """

    def __init__(self, pulse_duration, modulation_bandwidth=None, eom_mode=False):
        """
        Initializes the QuantumPulseDemoRunner.

        Args:
            pulse_duration (float): Duration of each pulse.
            modulation_bandwidth (float, optional): Bandwidth of modulation. Defaults to None.
            eom_mode (bool, optional): Whether to simulate EOM mode. Defaults to False.
        """
        self.model = QuantumPulseDemo()
        self.optimizer = optim.Adam(params=self.model.pulse.parameters(), lr=5e-3)
        self.target_unitary = self._initialize_target_unitary()
        self.simulator = OM_EOM_Simulation(pulse_duration, modulation_bandwidth, eom_mode)

    def _initialize_target_unitary(self):
        """
        Initializes the target unitary matrix.

        Returns:
            torch.Tensor: Target unitary matrix.
        """
        theta = 0.6
        return torch.tensor(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=torch.complex64,
        )

    def train(self, epochs=1000):
        """
        Trains the QuantumPulseDemo model.

        Args:
            epochs (int, optional): Number of training epochs. Defaults to 1000.
        """
        for epoch in range(epochs):
            x = torch.tensor([[np.pi, np.pi]], dtype=torch.float32)

            qdev = self.model(x)

            loss = (
                1
                - (
                    torch.trace(self.model.pulse.get_unitary() @ self.target_unitary)
                    / self.target_unitary.shape[0]
                ).abs()
                ** 2
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
                print('Current Pulse Shape:', self.model.pulse.pulse_shape)
                print('Current Unitary:\n', self.model.pulse.get_unitary())

    def run_simulation(self):
        """
        Runs the OM_EOM_Simulation.
        """
        sequence = self.simulator.simulate_sequence()
        for step in sequence:
            print(step)

# Example usage
runner = QuantumPulseDemoRunner(pulse_duration=100, modulation_bandwidth=5, eom_mode=True)
runner.train()
runner.run_simulation()
