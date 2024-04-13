import numpy as np
from panqec.codes import StabilizerCode
from panqec.error_models import BaseErrorModel
from panqec.decoders import BaseDecoder

import z3 as z3


"""
Z3 Decoder for PanQEC. Needs modification of installation of site-packages
Make new folder at decoders/z3_deocer/
Add this file into decoders/z3_decoder/, together with empty __init.py__ file.
Also modify config.py file:
Add at header
from .decoders.z3_decoder._z3decoder import Z3Decoder
Add into CONFIG dict:
    'Z3Decoder':Z3Decoder

"""


class Z3Decoder(BaseDecoder):
    label = "Z3Decoder"
    allowed_codes = None  # all codes allowed

    def __init__(
        self,
        code: StabilizerCode,
        error_model: BaseErrorModel,
        error_rate: float,
        param: int = 0,
        ignore_z: bool = False,
        ignore_x: bool = False,
    ):
        super().__init__(code, error_model, error_rate)
        self._param = param

        self.ignore_z = ignore_z
        self.ignore_x = ignore_x

        # Do not initialize the decoder until we call the decode method.
        # This is required because during analysis, there is no need to
        # initialize the decoder every time.
        self._initialized = False

    @property
    def params(self) -> dict:
        return {
            "param": self._param,
        }

    def get_probabilities(self):
        pi, px, py, pz = self.error_model.probability_distribution(
            self.code, self.error_rate
        )

        return pi, px, py, pz

    @staticmethod
    def binary_xor_to_CNF(A, B, C):
        ##A+B==C turns into
        ## (A v B v not(C)) &
        ## (not(A) v B v C) &
        ## (A v not(B) v C) &
        ## (not(A) v not(B) v not(C)) &
        form1 = A + " v " + B + " v " + "NOT(" + C + ")"
        form2 = "NOT(" + A + ")" + " v " + B + " v " + C
        form3 = A + " v " + "NOT(" + B + ")" + " v " + C
        form4 = "NOT(" + A + ")" + " v " + "NOT(" + B + ")" + " v " + "NOT(" + C + ")"
        return [form1, form2, form3, form4]

    @staticmethod
    def binary_xor_fixed_to_CNF(A, B, val, helper):
        ##We want to implement A+B==val as CNF
        ##A, B are variables, while val is a binary number
        ##helper is name of helper variable, should be unique
        ##A+B==val turns into
        if val == 0:
            form1 = "NOT(" + A + ")" + " v " + B + " v " + helper
            form2 = "NOT(" + A + ")" + " v " + B + " v " + "NOT(" + helper + ")"
            form3 = A + " v " + "NOT(" + B + ")" + " v " + helper
            form4 = A + " v " + "NOT(" + B + ")" + " v " + "NOT(" + helper + ")"
        elif val == 1:
            form1 = A + " v " + B + " v " + helper
            form2 = A + " v " + B + " v " + "NOT(" + helper + ")"
            form3 = "NOT(" + A + ")" + " v " + "NOT(" + B + ")" + " v " + helper
            form4 = (
                "NOT("
                + A
                + ")"
                + " v "
                + "NOT("
                + B
                + ")"
                + " v "
                + "NOT("
                + helper
                + ")"
            )
        else:
            raise NameError("Incorrect value for val", val)

        return [form1, form2, form3, form4]

    @staticmethod
    def binary_equal_to_CNF(A, val, helper1, helper2):
        ##A is variables, while val is a binary number
        ##We want to implement A=val as CNF
        ##helper1, helper2 is name of helper variables, should be unique
        ##A==val turns into
        if val == 0:
            form1 = "NOT(" + A + ")" + " v " + helper1 + " v " + helper2
            form2 = "NOT(" + A + ")" + " v " + "NOT(" + helper1 + ")" + " v " + helper2
            form3 = "NOT(" + A + ")" + " v " + helper1 + " v " + "NOT(" + helper2 + ")"
            form4 = (
                "NOT("
                + A
                + ")"
                + " v "
                + "NOT("
                + helper1
                + ")"
                + " v "
                + "NOT("
                + helper2
                + ")"
            )
        elif val == 1:
            form1 = A + " v " + helper1 + " v " + helper2
            form2 = A + " v " + "NOT(" + helper1 + ")" + " v " + helper2
            form3 = A + " v " + helper1 + " v " + "NOT(" + helper2 + ")"
            form4 = A + " v " + "NOT(" + helper1 + ")" + " v " + "NOT(" + helper2 + ")"
        else:
            raise NameError("Incorrect value for val", val)

        return [form1, form2, form3, form4]

    @staticmethod
    def z3_decoder_ini(parity_check_matrix, error_rate):
        ##initialise decoder

        ##weights for each error due to maximum likelihood
        weights = np.log((1 - error_rate) / error_rate)

        num_errors = np.shape(parity_check_matrix)[1]
        num_detectors = np.shape(parity_check_matrix)[0]

        ##set up z3
        ##soft and hard constraints
        CNF_form_hard_const = (
            []
        )  ##only contains constant CNF terms not involving syndrom
        CNF_form_soft_const = []  ##full soft constraints for CNF

        opt_z3 = z3.Optimize()

        error_vars = [z3.Bool(f"error_{i}") for i in range(num_errors)]

        ##switch variables for normal form
        error_CNF_vars = [f"x_{i}" for i in range(num_errors)]

        # syndrom_CNF_vars=[f"s_{i}" for i in range(num_detectors)]

        helper_vars = [[] for k in range(num_detectors)]

        ##helper variables for CNF form
        helper_CNF_vars = [[] for k in range(num_detectors)]

        ##add soft constraints of likelyhood for each error
        for k in range(num_errors):
            opt_z3.add_soft(z3.Not(error_vars[k]), weight=weights[k])

            CNF_form_soft_const.append("NOT(" + error_CNF_vars[k] + ")")

        errors_trigger_detector_list = []
        for k in range(num_detectors):
            index_matrix = parity_check_matrix[k].toarray().flatten()
            # print(parity_check_matrix[k].toarray())
            errors_trigger_detector = np.arange(num_errors)[index_matrix > 0]
            errors_trigger_detector_list.append(errors_trigger_detector)

        ##hard constraint on syndromes
        ##preconstruction only, syndrom values are added later
        for k in range(num_detectors):
            ##get errors that trigger detector, i.e. switches
            # #print(np.shape(parity_check_matrix))
            # index_matrix=parity_check_matrix[k].toarray().flatten()
            # #print(parity_check_matrix[k].toarray())
            # errors_trigger_detector=np.arange(num_errors)[index_matrix>0]

            errors_trigger_detector = errors_trigger_detector_list[k]

            # helper_vars[k] = [z3.Bool(f"helper_{k}_{i}") for i in range(len(errors_trigger_detector) - 1)]
            ##for a parity constraints involving k switches, we need k-2 helper variables
            helper_vars[k] = [
                z3.Bool(f"helper_{k}_{i}")
                for i in range(len(errors_trigger_detector) - 2)
            ]

            ##helper variables for normal form
            helper_CNF_vars[k] = [
                f"h_{k}_{i}" for i in range(len(errors_trigger_detector) - 2)
            ]

            ##this mapping is adapted from https://arxiv.org/pdf/2303.14237.pdf
            ##but uses one less helper variable which is redundant
            for i in range(1, len(errors_trigger_detector) - 2):
                constraint = (
                    z3.Xor(error_vars[errors_trigger_detector[i]], helper_vars[k][i])
                    == helper_vars[k][i - 1]
                )
                opt_z3.add(z3.simplify(constraint))

                ##do same for CNF
                CNF_form_hard_const.append(
                    Z3Decoder.binary_xor_to_CNF(
                        error_CNF_vars[errors_trigger_detector[i]],
                        helper_CNF_vars[k][i],
                        helper_CNF_vars[k][i - 1],
                    )
                )

            if len(errors_trigger_detector) > 2:
                constraint = (
                    z3.Xor(
                        error_vars[errors_trigger_detector[-2]],
                        error_vars[errors_trigger_detector[-1]],
                    )
                    == helper_vars[k][-1]
                )
                opt_z3.add(z3.simplify(constraint))
                CNF_form_hard_const.append(
                    Z3Decoder.binary_xor_to_CNF(
                        error_CNF_vars[errors_trigger_detector[-2]],
                        error_CNF_vars[errors_trigger_detector[-1]],
                        helper_CNF_vars[k][-1],
                    )
                )

        return (
            opt_z3,
            error_vars,
            helper_vars,
            errors_trigger_detector_list,
            CNF_form_hard_const,
            CNF_form_soft_const,
        )

    @staticmethod
    def flatten(l):
        return [item for sublist in l for item in sublist]

    @staticmethod
    def z3_do_decoding(
        opt_z3, error_vars, helper_vars, syndrome, errors_trigger_detector_list
    ):
        ##run decoder

        ##turn syndrom into boolean variable for z3
        bool_syndrom = [bool(b) for b in syndrome]

        num_errors = len(error_vars)
        num_detectors = len(syndrome)

        # num_errors=np.shape(parity_check_matrix)[1]
        # num_detectors=np.shape(parity_check_matrix)[0]

        opt_z3.push()  ##create safe state of optimizer for syndrom value to add

        CNF_form_hard_flex = []

        # ##switch variables for normal form
        # error_CNF_vars=[f"x_{i}" for i in range(num_errors)]

        ##add syndrom to z3 constraints
        for k in range(num_detectors):
            # index_matrix=parity_check_matrix[k].toarray().flatten()
            # errors_trigger_detector=np.arange(num_errors)[index_matrix>0]
            errors_trigger_detector = errors_trigger_detector_list[k]
            # errors_trigger_detector=np.arange(num_errors)[parity_check_matrix[k]>0]
            if len(errors_trigger_detector) == 0:
                ##do nothing when parity constraint involes 0 switches
                pass
                # print("Detector",k,"cannot ever be triggered")
            elif len(errors_trigger_detector) == 1:
                ##Need to set switch equal to syndrom
                ##when parity constraint involes 1 switch simply set light equal to syndrom
                constraint = error_vars[errors_trigger_detector[0]] == bool_syndrom[k]
                opt_z3.add(z3.simplify(constraint))

                # helper1=f"g_{k}_{0}" ##helper1 for syndrome
                # helper2=f"g_{k}_{1}" ##helper2 for syndrome
                # CNF=Z3Decoder.binary_equal_to_CNF(error_CNF_vars[errors_trigger_detector[0]],bool_syndrom[k],helper1,helper2)

                # CNF_form_hard_flex.append(CNF)
            elif len(errors_trigger_detector) == 2:
                ##for k=2 switch constraint we can directly implement constraint without helper
                constraint = (
                    error_vars[errors_trigger_detector[0]],
                    error_vars[errors_trigger_detector[1]] == bool_syndrom[k],
                )
                opt_z3.add(z3.simplify(constraint))

                # helper=f"g_{k}_{0}" ##helper1 for syndrome
                # CNF=Z3Decoder.binary_xor_fixed_to_CNF(error_CNF_vars[errors_trigger_detector[0]],error_CNF_vars[errors_trigger_detector[1]],bool_syndrom[k],helper)
                # CNF_form_hard_flex.append(CNF)

            else:
                ##implement constraint for syndrom via helper for k>2
                constraint = (
                    z3.Xor(error_vars[errors_trigger_detector[0]], helper_vars[k][0])
                    == bool_syndrom[k]
                )
                opt_z3.add(z3.simplify(constraint))

                # helper=f"g_{k}_{0}" ##helper1 for syndrome
                # CNF=Z3Decoder.binary_xor_fixed_to_CNF(error_CNF_vars[errors_trigger_detector[0]],helper_CNF_vars[k][0],bool_syndrom[k],helper)
                # CNF_form_hard_flex.append(CNF)

        CNF_form_hard_flex = Z3Decoder.flatten(CNF_form_hard_flex)  ##flatten CNF

        # print(opt)

        result = opt_z3.check()

        assert str(result) == "sat", "No solution found"

        # validate the model
        model_z3 = opt_z3.model()

        ##error mechanism (i.e. switches) predicted by decoder
        predicted_error = [1 if model_z3[var] else 0 for var in error_vars]

        opt_z3.pop()  # return to previous constraint condition

        return predicted_error, CNF_form_hard_flex

    def initialize_decoders(self):
        self.is_css = self.code.is_css
        ##self.is_css=False

        pi, px, py, pz = self.get_probabilities()

        probabilities_x = px + py
        probabilities_z = pz + py

        probabilities = np.hstack([probabilities_z, probabilities_x])

        if self.is_css:
            if self.ignore_z == False:
                (
                    self.opt_z3_z,
                    self.error_vars_z,
                    self.helper_vars_z,
                    self.errors_trigger_detector_z,
                    self.CNF_form_hard_const_z,
                    CNF_form_soft_const_z,
                ) = Z3Decoder.z3_decoder_ini(
                    self.code.Hx,
                    error_rate=probabilities_z,
                )

            if self.ignore_x == False:
                (
                    self.opt_z3_x,
                    self.error_vars_x,
                    self.helper_vars_x,
                    self.errors_trigger_detector_x,
                    self.CNF_form_hard_const_x,
                    CNF_form_soft_const_x,
                ) = Z3Decoder.z3_decoder_ini(
                    self.code.Hz,
                    error_rate=probabilities_x,
                )

        else:
            (
                self.opt_z3,
                self.error_vars,
                self.helper_vars,
                self.errors_trigger_detector,
                self.CNF_form_hard_const,
                CNF_form_soft_const,
            ) = Z3Decoder.z3_decoder_ini(
                self.code.stabilizer_matrix,
                error_rate=probabilities,
            )
        self._initialized = True

    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        if not self._initialized:
            self.initialize_decoders()

        n_qubits = self.code.n
        syndrome = np.array(syndrome, dtype=int)

        if self.is_css:
            syndrome_z = self.code.extract_z_syndrome(syndrome)
            syndrome_x = self.code.extract_x_syndrome(syndrome)

        if self.is_css:
            if self.ignore_z == False:
                # Decode Z errors
                z_correction, CNF_form_hard_flex_z = Z3Decoder.z3_do_decoding(
                    self.opt_z3_z,
                    self.error_vars_z,
                    self.helper_vars_z,
                    syndrome_x,
                    self.errors_trigger_detector_z,
                )
            else:
                z_correction = [0 for k in range(n_qubits)]

            if self.ignore_x == False:
                # Decode X errors
                x_correction, CNF_form_hard_flex_x = Z3Decoder.z3_do_decoding(
                    self.opt_z3_x,
                    self.error_vars_x,
                    self.helper_vars_x,
                    syndrome_z,
                    self.errors_trigger_detector_x,
                )
            else:
                x_correction = [0 for k in range(n_qubits)]

            correction = np.concatenate([x_correction, z_correction])
            # print(correction)
        else:
            # Decode all errors
            correction, CNF_form_hard_flex = Z3Decoder.z3_do_decoding(
                self.opt_z3,
                self.error_vars,
                self.helper_vars,
                syndrome,
                self.errors_trigger_detector,
            )

            correction = np.concatenate([correction[n_qubits:], correction[:n_qubits]])
            # print(correction)

        return correction


def test_decoder():
    from panqec.codes import XCubeCode
    from panqec.error_models import PauliErrorModel
    import time

    rng = np.random.default_rng()

    L = 20
    code = XCubeCode(L, L, L)

    error_rate = 0.5
    r_x, r_y, r_z = [0.15, 0.15, 0.7]
    error_model = PauliErrorModel(r_x, r_y, r_z)

    print("Create stabilizer matrix")
    code.stabilizer_matrix

    print("Create Hx and Hz")
    code.Hx
    code.Hz

    print("Create logicals")
    code.logicals_x
    code.logicals_z

    print("Instantiate z3 decoder")
    decoder = Z3Decoder(
        code,
        error_model,
        error_rate,
    )

    # Start timer
    start = time.time()

    n_iter = 1
    accuracy = 0
    for i in range(n_iter):
        print(f"\nRun {code.label} {i}...")
        print("Generate errors")
        error = error_model.generate(code, error_rate, rng=rng)
        print("Calculate syndrome")
        syndrome = code.measure_syndrome(error)
        print("Decode")
        correction = decoder.decode(syndrome)
        print("Get total error")
        total_error = (correction + error) % 2

        codespace = code.in_codespace(total_error)
        success = not code.is_logical_error(total_error) and codespace
        print(success)
        accuracy += success

    accuracy /= n_iter
    print("Average time per iteration", (time.time() - start) / n_iter)
    print("Logical error rate", 1 - accuracy)


if __name__ == "__main__":
    test_decoder()
