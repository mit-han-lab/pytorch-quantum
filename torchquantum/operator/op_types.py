import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional.functionals as tqf
import numpy as np
from abc import ABCMeta
from ..macro import C_DTYPE, F_DTYPE
from typing import Iterable, Union, List
from enum import IntEnum

__all__ = [
    "Operator",
    "Operation",
    "DiagonalOperation",
    "Observable",
    "WiresEnum",
    "NParamsEnum",
    "AnyNParams",
    "AllWires",
    "AnyWires",
]


class WiresEnum(IntEnum):
    """Integer enumeration class
    to represent the number of wires
    an operation acts on."""

    AnyWires = -1
    AllWires = 0


class NParamsEnum(IntEnum):
    """Integer enumeration class
    to represent the number of wires
    an operation acts on"""

    AnyNParams = -1


AnyNParams = NParamsEnum.AnyNParams


AllWires = WiresEnum.AllWires
"""IntEnum: An enumeration which represents all wires in the
subsystem. It is equivalent to an integer with value 0."""

AnyWires = WiresEnum.AnyWires
"""IntEnum: An enumeration which represents any wires in the
subsystem. It is equivalent to an integer with value -1."""


class Operator(tq.QuantumModule):
    """The class for quantum operators."""

    @property
    def name(self):
        """String for the name of the operator."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name of the operator.

        Args:
            value (str): operator name.

        """
        self._name = value

    def __init__(
        self,
        has_params: bool = False,
        trainable: bool = False,
        init_params=None,
        n_wires=None,
        wires=None,
        inverse=False,
    ):
        """__init__ function for Operator.

        Args:
            has_params (bool, optional): Whether the operations has parameters.
                Defaults to False.
            trainable (bool, optional): Whether the parameters are trainable
                (if contains parameters). Defaults to False.
            init_params (torch.Tensor, optional): Initial parameters.
                Defaults to None.
            n_wires (int, optional): Number of qubits. Defaults to None.
            wires (Union[int, List[int]], optional): Which qubit the operation
                is applied to. Defaults to None.
        """
        super().__init__()
        self.params = None
        # number of wires of the operator
        # n_wires is used in gates that can be applied to arbitrary number
        # of qubits such as MultiRZ
        self.n_wires = n_wires
        # wires that the operator applies to
        self.wires = wires
        self._name = self.__class__.__name__
        # for static mode
        self.static_matrix = None
        self.inverse = inverse
        self.clifford_quantization = False

        try:
            assert not (trainable and not has_params)
        except AssertionError:
            has_params = True
            logger.warning(
                f"Module must have parameters to be trainable; "
                f"Switched 'has_params' to True."
            )

        self.has_params = has_params
        self.trainable = trainable
        if self.has_params:
            self.params = self.build_params(trainable=self.trainable)
            self.reset_params(init_params)

    @classmethod
    def _matrix(cls, params):
        """The unitary matrix of the operator.

        Args:
            params (torch.Tensor, optional): The parameters for parameterized
                operators.

        Returns: None.

        """
        raise NotImplementedError

    @property
    def matrix(self):
        """The unitary matrix of the operator."""
        return self._matrix(self.params)

    @classmethod
    def _eigvals(cls, params):
        """The eigenvalues of the unitary matrix of the operator.

        Args:
            params (torch.Tensor, optional): The parameters for parameterized
                operators.

        Returns: None.

        """
        # Warning: The eigenvalues of the operator {cls.__name__} are not defined.
        return None

    @property
    def eigvals(self):
        """The eigenvalues of the unitary matrix of the operator.

        Returns: Eigenvalues.

        """
        return self._eigvals(self.params)

    def _get_unitary_matrix(self):
        """Obtain the unitary matrix of the operator.

        Returns: Unitary matrix.

        """
        return self.matrix

    def set_wires(self, wires):
        """Set which qubits the operator is applied to.

        Args:
            wires (Union[int, List[int]]): Qubits the operator is applied to.

        Returns: None.

        """
        self.wires = [wires] if isinstance(wires, int) else wires

    def forward(
        self, q_device: tq.QuantumDevice, wires=None, params=None, inverse=None
    ):
        """Apply the operator to the quantum device states.

        Args:
            q_device (torchquantum.QuantumDevice): Quantum Device that the
                operator is applied to.
            wires (Union[int, List[int]]): Qubits that the operator is
                applied to.
            params (torch.Tensor): Parameters of the operator
            inverse (bool): Whether inverse the unitary matrix of the operator.

        Returns:

        """
        if inverse is not None:
            logger.warning("replace the inverse flag with the input")
            self.inverse = inverse
        # try:
        #     assert self.name in self.fixed_ops or \
        #            self.has_params ^ (params is not None)
        # except AssertionError as err:
        #     logger.exception(f"Parameterized gate either has its "
        #                      f"own parameters or has input as parameters")
        #     raise err

        # try:
        #     assert not (self.wires is None and wires is None)
        # except AssertionError as err:
        #     logger.exception(f"Need to specify the wires either when "
        #                      f"initialize or when forward")
        #     raise err

        if params is not None:
            self.params = params

        if self.params is not None:
            self.params = (
                self.params.unsqueeze(-1) if self.params.dim() == 1 else self.params
            )

        if wires is not None:
            # update the wires
            wires = [wires] if isinstance(wires, int) else wires
            self.wires = wires

        # self.inverse = inverse

        if self.static_mode:
            self.parent_graph.add_op(self)
            return

        # non-parameterized gate
        if self.params is None:
            if self.n_wires is None:
                self.func(q_device, self.wires, inverse=self.inverse)  # type: ignore
            else:
                self.func(q_device, self.wires, n_wires=self.n_wires, inverse=self.inverse)  # type: ignore
        else:
            if isinstance(self.noise_model_tq, tq.NoiseModelTQPhase):
                params = self.noise_model_tq.add_noise(self.params)
            else:
                params = self.params

            if self.clifford_quantization:
                from torchquantum.util.quantization.clifford_quantization import CliffordQuantizer
                params = CliffordQuantizer.quantize_sse(params)
            if self.n_wires is None:
                self.func(q_device, self.wires, params=params, inverse=self.inverse)
            else:
                self.func(
                    q_device,
                    self.wires,
                    params=params,
                    n_wires=self.n_wires,
                    inverse=self.inverse,
                )

        if self.noise_model_tq is not None and self.noise_model_tq.is_add_noise:
            noise_ops = self.noise_model_tq.sample_noise_op(self)
            if len(noise_ops):
                for noise_op in noise_ops:
                    noise_op(q_device)

    def __repr__(self):
        return f" class: {self.name} \n parameters: {self.params} \n wires: {self.wires} \n inverse: {self.inverse}"


class Observable(Operator, metaclass=ABCMeta):
    """Class for Observables."""

    def __init__(
        self,
        has_params: bool = False,
        trainable: bool = False,
        init_params=None,
        n_wires=None,
        wires=None,
        inverse=False,
    ):
        """Init function of the Observable class

        Args:
            has_params (bool, optional): Whether the operations has parameters.
                Defaults to False.
            trainable (bool, optional): Whether the parameters are trainable
                (if contains parameters). Defaults to False.
            init_params (torch.Tensor, optional): Initial parameters.
                Defaults to None.
            n_wires (int, optional): Number of qubits. Defaults to None.
            wires (Union[int, List[int]], optional): Which qubit the operation
                is applied to. Defaults to None.
        """
        super().__init__(
            has_params=has_params,
            trainable=trainable,
            init_params=init_params,
            n_wires=n_wires,
            wires=wires,
            inverse=inverse,
        )
        self.return_type = None

    def diagonalizing_gates(self):
        """The diagonalizing gates when perform measurements.

        Returns: None.

        """
        raise NotImplementedError


class Operation(Operator, metaclass=ABCMeta):
    """_summary_"""

    def __init__(
        self,
        has_params: bool = False,
        trainable: bool = False,
        init_params=None,
        n_wires=None,
        wires=None,
        inverse=False,
    ):
        """_summary_

        Args:
            has_params (bool, optional): Whether the operations has parameters.
                Defaults to False.
            trainable (bool, optional): Whether the parameters are trainable
                (if contains parameters). Defaults to False.
            init_params (torch.Tensor, optional): Initial parameters.
                Defaults to None.
            n_wires (int, optional): Number of qubits. Defaults to None.
            wires (Union[int, List[int]], optional): Which qubit the operation is applied to.
                Defaults to None.
        """
        super().__init__(
            has_params=has_params,
            trainable=trainable,
            init_params=init_params,
            n_wires=n_wires,
            wires=wires,
            inverse=inverse,
        )
        if type(self.num_wires) == int:
            self.n_wires = self.num_wires

    @property
    def matrix(self):
        """The unitary matrix of the operator."""
        op_matrix = self._matrix(self.params)

        return op_matrix

    @property
    def eigvals(self):
        """ "The eigenvalues of the unitary matrix of the operator.

        Returns:
            torch.Tensor: Eigenvalues.

        """
        op_eigvals = self._eigvals(self.params)

        return op_eigvals

    def init_params(self):
        """Initialize the parameters.

        Raises:
            NotImplementedError: The init param function is not implemented.
        """
        raise NotImplementedError

    def build_params(self, trainable):
        """Build parameters.

        Args:
            trainable (bool): Whether the parameters are trainable.

        Returns:
            torch.Tensor: Built parameters.
        """
        parameters = nn.Parameter(torch.empty([1, self.num_params], dtype=F_DTYPE))
        parameters.requires_grad = True if trainable else False
        # self.register_parameter(f"{self.name}_params", parameters)
        return parameters

    def reset_params(self, init_params=None):
        """Reset parameters.

        Args:
            init_params (torch.Tensor, optional): Input the initialization
                parameters. Defaults to None.
        """
        if init_params is not None:
            if isinstance(init_params, Iterable):
                for k, init_param in enumerate(init_params):
                    torch.nn.init.constant_(self.params[:, k], init_param)
            else:
                torch.nn.init.constant_(self.params, init_params)
        else:
            torch.nn.init.uniform_(self.params, -np.pi, np.pi)


class DiagonalOperation(Operation, metaclass=ABCMeta):
    """Class for Diagonal Operation."""

    @classmethod
    def _eigvals(cls, params):
        """The eigenvalues of the unitary matrix of the operator.

        Args:
            params (torch.Tensor, optional): The parameters for parameterized
                operators.

        Returns: None.
        raise NotImplementedError
        """

    @property
    def eigvals(self):
        """The eigenvalues of the unitary matrix of the operator.

        Returns: Eigenvalues.

        """
        return super().eigvals

    @classmethod
    def _matrix(cls, params):
        """The unitary matrix of the operator.

        Args:
            params (torch.Tensor, optional): The parameters for parameterized
                operators.

        Returns: None.

        """
        return torch.diag(cls._eigvals(params))
