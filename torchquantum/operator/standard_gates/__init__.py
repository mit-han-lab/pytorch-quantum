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

from .ecr import EchoedCrossResonance, ECR
from .global_phase import GlobalPhase
from .i import I
from .iswap import ISWAP

# TODO: Make paulix/y/z alias as X/Y/Z
from .paulix import PauliX, CNOT, C4X, C3X, DCX, MultiCNOT, MultiXCNOT
from .pauliy import PauliY, CY
from .pauliz import PauliZ, CZ, CCZ
from .hadamard import Hadamard, SHadamard, CHadamard, H, SH, CH
from .phase_shift import PhaseShift
from .qft import QFT
from .r import R
from .reset import Reset
from .rot import Rot, CRot
from .rx import RX, RXX, CRX
from .ry import RY, RYY, CRY
from .rz import RZ, MultiRZ, RZZ, RZX, CRZ
from .toffoli import Toffoli, CCX, RC3X, RCCX
from .qubit_unitary import QubitUnitary, QubitUnitaryFast
from .trainable_unitary import TrainableUnitary, TrainableUnitaryStrict
from .s import S, SDG, CS, CSDG
from .single_excitation import SingleExcitation
from .swap import SWAP, SSWAP, CSWAP
from .sx import SX, CSX, C3SX, SXDG
from .t import T, TDG
from .u1 import U1, CU1
from .u2 import U2, CU2
from .u3 import U3, CU3, CU, U
from .xx_min_yy import XXMINYY
from .xx_plus_yy import XXPLUSYY

_all_variables = [
    EchoedCrossResonance,
    ECR,
    GlobalPhase,
    I,
    ISWAP,
    PauliX,
    CNOT,
    C4X,
    C3X,
    DCX,
    MultiCNOT,
    MultiXCNOT,
    PauliY,
    CY,
    PauliZ,
    CZ,
    CCZ,
    Hadamard,
    SHadamard,
    CHadamard,
    H,
    SH,
    CH,
    PhaseShift,
    QFT,
    R,
    Reset,
    Rot,
    CRot,
    RX,
    RXX,
    CRX,
    RY,
    RYY,
    CRY,
    RZ,
    MultiRZ,
    RZZ,
    RZX,
    CRZ,
    Toffoli,
    CCX,
    RC3X,
    RCCX,
    S,
    SDG,
    CS,
    CSDG,
    SingleExcitation,
    SWAP,
    SSWAP,
    CSWAP,
    SX,
    CSX,
    C3SX,
    SXDG,
    T,
    TDG,
    TrainableUnitary,
    TrainableUnitaryStrict,
    U1,
    CU1,
    U2,
    CU2,
    U3,
    CU3,
    CU,
    U,
    XXMINYY,
    XXPLUSYY,
]

# create the operations dictionary
op_name_dict = {_x.op_name: _x for _x in _all_variables}

# add aliases as well
op_name_dict.update(
    {
        "h": H,
        "sh": SH,
        "u": U,
        "qubitunitary": QubitUnitary,
        "qubitunitaryfast": QubitUnitaryFast,
        "x": PauliX,
        "y": PauliY,
        "z": PauliZ,
        "cx": CNOT,
        "xx": RXX,
        "yy": RYY,
        "zz": RZZ,
        "zx": RZX,
        "ccx": Toffoli,
        "p": U1,
        "cp": CU1,
        "cr": CU1,
    }
)

# TODO: make this compatible with aliases
fixed_ops = [_a().__class__.__name__ for _a in _all_variables if _a.num_params == 0]
parameterized_ops = [_a().__class__.__name__ for _a in _all_variables if _a.num_params > 0]
