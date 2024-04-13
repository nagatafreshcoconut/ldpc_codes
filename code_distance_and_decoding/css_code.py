from typing import Tuple, Dict, List, Optional
import numpy as np
from panqec.codes import StabilizerCode

from bposd.css import css_code
import scipy


Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class CSSCode(StabilizerCode):
    """Quantum LPDC with CSS.

    Parameters
    ----------
    Hx_in : int [n_stab, n_qubits]
        Parity check matrix for X stabilizer of CSS code
    Hz_in : int [n_stab, n_qubits]
        Parity check matrix for Z stabilizer of CSS code
    name : str
        Name of code
    L : int
        size of code for identification, has no effect on code itself


    This file needs to be put into codes/CSS/_css.py in the panqec folder
    The folder CSS also needs an empty __init__.py file




    Note: This stores a dense representation of Hx, Hz as this is needed by css_code routine
    In particular, css_code gets slow in computing logical operators beyond ~3000 qubits.
    It may be good to implement sparse way to compute logicals.
    """

    dimension = 1
    deformation_names = []

    def __init__(self, Hx_in, Hz_in, name, L_in=None):
        if scipy.sparse.issparse(Hx_in):
            Hx_in_array = Hx_in.toarray()
        else:
            Hx_in_array = np.array(Hx_in)

        if scipy.sparse.issparse(Hz_in):
            Hz_in_array = Hz_in.toarray()
        else:
            Hz_in_array = np.array(Hz_in)

        n_stab_x, n_qubits = np.shape(Hx_in_array)  ##X stabilizer, relate to Z errors
        n_stab_z, n_qubits = np.shape(Hz_in_array)  ##Z stabilizer, relate to X errors

        ##parity check matrix constructed by panQEC as
        ##( 0 H_z // H_x 0). First rows are z errors, Last rows x errors

        ##use css_code function to generate logicals etc.
        ##note that these are used to set stabilizer and logical operators

        ##css_code does not seem to work with sparse input
        qcode = css_code(Hx_in_array, Hz_in_array)
        qcode.test()

        ##from css_code
        self.lz = qcode.lz
        self.lx = qcode.lx
        self.n_logicals = self.lz.shape[0]  ##number of logicals

        self.n_qubits = n_qubits
        self.n_stab_x = n_stab_x
        self.n_stab_z = n_stab_z
        self.n_stab = n_stab_x + n_stab_z
        self.name = name

        self.qubit_stab_indices_z = [
            np.nonzero(Hz_in_array[x, :])[0] for x in range(self.n_stab_z)
        ]
        self.qubit_stab_indices_x = [
            np.nonzero(Hx_in_array[x, :])[0] for x in range(self.n_stab_x)
        ]

        if L_in == None:
            super().__init__(self.n_qubits)
        else:
            super().__init__(L_in)

    @property
    def label(self) -> str:
        return "QLDPC" + self.name

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []

        # Qubits
        for x in range(self.n_qubits):
            coordinates.append((x,))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []

        # stabilizers
        for x in range(self.n_stab):
            coordinates.append((x,))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")
        (x,) = location
        if x < self.n_stab_z:
            return "Z"
        else:
            return "X"

    def get_stabilizer(self, location) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        (x,) = location
        if self.stabilizer_type(location) == "Z":
            pauli = "Z"
            # H=self.Hz_in
            # shift=0
            qubit_indices = self.qubit_stab_indices_z[x]

        else:
            pauli = "X"
            # H=self.Hx_in
            # shift=self.n_stab_z

            qubit_indices = self.qubit_stab_indices_x[x - self.n_stab_z]

        # ##check on which qubits stabilizer acts non-trivially
        # qubit_indices=np.nonzero(H[x-shift,:])[0]

        operator = dict()
        for k in qubit_indices:
            qubit_location = (k,)
            # print(qubit_indices)
            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location) -> str:
        (x,) = location
        axis = "x"
        return axis

    def get_logicals_x(self) -> List[Operator]:
        """The logical X operators."""

        logicals = []

        for k in range(self.n_logicals):
            target_index = np.nonzero(self.lx[k])[0]
            operator: Operator = dict()
            for x in target_index:
                operator[(x,)] = "X"
            logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        """The logical Z operators."""

        logicals = []

        for k in range(self.n_logicals):
            target_index = np.nonzero(self.lz[k])[0]
            operator: Operator = dict()
            for x in target_index:
                operator[(x,)] = "Z"
            logicals.append(operator)

        return logicals

    # def get_deformation(
    #     self, location: Tuple,
    #     deformation_name: str,
    #     deformation_axis: str = 'y',
    #     **kwargs
    # ) -> Dict:

    #     if deformation_axis not in ['x', 'y']:
    #         raise ValueError(f"{deformation_axis} is not a valid "
    #                          "deformation axis")

    #     if deformation_name == 'XZZX':
    #         undeformed_dict = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
    #         deformed_dict = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}

    #         if self.qubit_axis(location) == deformation_axis:
    #             deformation = deformed_dict
    #         else:
    #             deformation = undeformed_dict

    #     elif deformation_name == 'XY':
    #         deformation = {'X': 'X', 'Y': 'Z', 'Z': 'Y'}

    #     else:
    #         raise ValueError(f"The deformation {deformation_name}"
    #                          "does not exist")

    #     return deformation
