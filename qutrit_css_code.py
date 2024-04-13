import numpy as np


class QutritCSSCode:
    def __init__(self, Hx, Hz):
        """
        Initialize the CSS code object with two parity check matrices.
        We assume that the CSS condition is already satisfied Hx(Hz)^T = 0

        Parameters:
        - Hx, Hz: Parity check matrices for the CSS code.
        Assuming Hx and Hz are Matrix objects in SageMath, we use SageMath functions
        """
        self.Hx = Hx
        self.Hz = Hz
        # assert np.allclose(Hx @ Hz.T, np.zeros_like(Hx @ Hz.T)), 'CSS condition not satisfied! We need: Hx @ Hz.T == 0'

        self.N = len(self.Hx[0])  # Number of physical qubits = number of columns
        self.K = (
            self.N - np.linalg.matrix_rank(self.Hx) - np.linalg.matrix_rank(self.Hz)
        )  # Number of logical qubits

        # TODO: Include the logicals lx, lz

    def test(self):
        """
        Placeholder method for testing the code.
        Returns True if the code passes the test.
        """
        if len(self.Hx[0]) == len(self.Hz[0]):
            return True
        else:
            print("ncols of Hx not same as ncols of Hz!")
            return False
