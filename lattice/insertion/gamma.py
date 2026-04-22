from functools import lru_cache

from ..backend import get_backend


class _Constant:
    @staticmethod
    @lru_cache(1)
    def zero():
        backend = get_backend()
        return backend.zeros((4, 4))

    @staticmethod
    @lru_cache(1)
    def one():
        backend = get_backend()
        return backend.identity(4)

    @staticmethod
    @lru_cache(1)
    def gamma_0():
        backend = get_backend()
        return backend.array(
            [
                [0, 0, 0, 1j],
                [0, 0, 1j, 0],
                [0, -1j, 0, 0],
                [-1j, 0, 0, 0],
            ]
        )

    @staticmethod
    @lru_cache(1)
    def gamma_1():
        backend = get_backend()
        return backend.array(
            [
                [0, 0, 0, -1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
            ]
        )

    @staticmethod
    @lru_cache(1)
    def gamma_2():
        backend = get_backend()
        return backend.array(
            [
                [0, 0, 1j, 0],
                [0, 0, 0, -1j],
                [-1j, 0, 0, 0],
                [0, 1j, 0, 0],
            ]
        )

    @staticmethod
    @lru_cache(1)
    def gamma_3():
        backend = get_backend()
        return backend.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )


def output(n: int):
    assert isinstance(n, int) and 0 <= n <= 15
    if n == 0:
        return ""
    elif n == 15:
        return "γ5"
    elif n == 7:
        return "γ5γ4"
    elif n == 8:
        return "γ4"
    elif n in [14, 13, 11]:
        return f"γ{[14, 13, 11].index(n)+1}γ5"
    elif n in [1, 2, 4]:
        return f"γ{[1, 2, 4].index(n)+1}"
    elif n in [9, 10, 12]:
        return f"γ{[9, 10, 12].index(n)+1}γ4"
    elif n in [6, 5, 3]:
        return f"γ{[6, 5, 3].index(n)+1}γ5γ4"


def gamma(n: int):
    assert isinstance(n, int) and 0 <= n <= 15
    backend = get_backend()
    return backend.asarray(
        (_Constant.gamma_0() if n & 0b0001 else _Constant.one())
        @ (_Constant.gamma_1() if n & 0b0010 else _Constant.one())
        @ (_Constant.gamma_2() if n & 0b0100 else _Constant.one())
        @ (_Constant.gamma_3() if n & 0b1000 else _Constant.one())
    )


_naming_scheme = {
    R"$a_0$": [0],  # g0 # 0++ # 1
    R"$\pi$": [15],  # g5 # 0-+ # g1g2g3g4
    R"$\pi(2)$": [7],  # g5g4 # 0-+ # g1g2g3
    R"$b_0$": [8],  # g4 # 0+- # g4
    R"$a_1$": [14, 13, 11],  # gig5 # 1++ # g2g3g4 -g1g3g4 g1g2g4
    R"$\rho$": [1, 2, 4],  # gi # 1-- # g1 g2 g3
    R"$\rho(2)$": [9, 10, 12],  # gig4 # 1-- # g1g4 g2g4 g3g4
    R"$b_1$": [6, 5, 3],  # gig5g4 # 1+- # g2g3 -g1g3 g2g3
}

_naming_group = {
    R"$a_0$": "A_1",
    R"$\pi$": "A_1",
    R"$\pi(2)$": "A_1",
    R"$b_0$": "A_1",
    R"$a_1$": "T_1",
    R"$\rho$": "T_1",
    R"$\rho(2)$": "T_1",
    R"$b_1$": "T_1",
}

_naming_hermiticity = {
    R"$a_0$": "+",
    R"$\pi$": "-",
    R"$\pi(2)$": "+",
    R"$b_0$": "+",
    R"$a_1$": "-",
    R"$\rho$": "-",
    R"$\rho(2)$": "+",
    R"$b_1$": "-",
}

_naming_parity = {
    R"$a_0$": "+",
    R"$\pi$": "-",
    R"$\pi(2)$": "-",
    R"$b_0$": "+",
    R"$a_1$": "+",
    R"$\rho$": "-",
    R"$\rho(2)$": "-",
    R"$b_1$": "+",
}

_naming_charge_conjugation = {
    R"$a_0$": "+",
    R"$\pi$": "+",
    R"$\pi(2)$": "+",
    R"$b_0$": "-",
    R"$a_1$": "+",
    R"$\rho$": "-",
    R"$\rho(2)$": "-",
    R"$b_1$": "-",
}

_naming_time_reversal = {
    R"$a_0$": "+",
    R"$\pi$": "+",
    R"$\pi(2)$": "+",
    R"$b_0$": "+",
    R"$a_1$": "+",
    R"$\rho$": "+",
    R"$\rho(2)$": "+",
    R"$b_1$": "+",
}


def scheme(name: str):
    assert name in _naming_scheme
    return _naming_scheme[name]


def irrep(name: str):
    assert name in _naming_scheme
    return _naming_group[name]


def parity(name: str):
    assert name in _naming_scheme
    return 1 if _naming_parity[name] == "+" else -1


def charge_conjugation(name: str):
    assert name in _naming_scheme
    return 1 if _naming_charge_conjugation[name] == "+" else -1


def hermiticity(name: str):
    assert name in _naming_scheme
    return 1 if _naming_hermiticity[name] == "+" else -1


class GammaName:
    A0 = R"$a_0$"
    B0 = R"$b_0$"
    PI = R"$\pi$"
    PI_2 = R"$\pi(2)$"
    RHO = R"$\rho$"
    RHO_2 = R"$\rho(2)$"
    A1 = R"$a_1$"
    B1 = R"$b_1$"


from lattice.symmetry.gen_hardcoded_rep import *
from lattice.symmetry.hardcoded_rep import group_element


def genGammaTransformDict():
    gamma_transform_dict = {}
    for key in group_element:
        # if key.startswith("invr") or key.startswith("r"):
        #     continue
        gamma_transform_dict[key] = [None] * 16
    gamma_transform_dict["conj"] = [None] * 16
    for i in range(16):
        for name in _naming_scheme:
            if i in scheme(name):
                idx = scheme(name).index(i)
                break
        irrep_name = _naming_group[name]
        p = parity(name)
        c = charge_conjugation(name)
        h = hermiticity(name)
        group_matrix = genLittleGroupIrrep([0, 0, 0], irrep_name, p)
        for key in group_element:
            # if key.startswith("invr") or key.startswith("r"):
            #     continue
            matrix = group_matrix[key]
            result = matrix[:, idx]
            for j in range(len(result)):
                if result[j] == 1:
                    gamma_transform_dict[key][i] = _naming_scheme[name][j]
                elif result[j] == -1:
                    gamma_transform_dict[key][i] = _naming_scheme[name][j] + 16
        if c == 1:
            gamma_transform_dict["conj"][i] = i
        elif c == -1:
            gamma_transform_dict["conj"][i] = i + 16

    return gamma_transform_dict


_gamma_transform_dict = {
    "iden": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "c4x": [0, 1, 4, 21, 18, 3, 6, 7, 8, 9, 12, 29, 26, 11, 14, 15],
    "c2x": [0, 1, 18, 19, 20, 21, 6, 7, 8, 9, 26, 27, 28, 29, 14, 15],
    "c4x^-1": [0, 1, 20, 5, 2, 19, 6, 7, 8, 9, 28, 13, 10, 27, 14, 15],
    "c4y": [0, 20, 2, 6, 1, 5, 19, 7, 8, 28, 10, 14, 9, 13, 27, 15],
    "c2y": [0, 17, 2, 19, 20, 5, 22, 7, 8, 25, 10, 27, 28, 13, 30, 15],
    "c4y^-1": [0, 4, 2, 22, 17, 5, 3, 7, 8, 12, 10, 30, 25, 13, 11, 15],
    "c4z": [0, 2, 17, 3, 4, 22, 5, 7, 8, 10, 25, 11, 12, 30, 13, 15],
    "c2z": [0, 17, 18, 3, 4, 21, 22, 7, 8, 25, 26, 11, 12, 29, 30, 15],
    "c4z^-1": [0, 18, 1, 3, 4, 6, 21, 7, 8, 26, 9, 11, 12, 14, 29, 15],
    "c3delta": [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15],
    "c3delta^-1": [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15],
    "c3gamma": [0, 18, 4, 22, 17, 3, 21, 7, 8, 26, 12, 30, 25, 11, 29, 15],
    "c3gamma^-1": [0, 20, 17, 5, 2, 22, 19, 7, 8, 28, 25, 13, 10, 30, 27, 15],
    "c3beta": [0, 18, 20, 6, 1, 19, 21, 7, 8, 26, 28, 14, 9, 27, 29, 15],
    "c3beta^-1": [0, 4, 17, 21, 18, 22, 3, 7, 8, 12, 25, 29, 26, 30, 11, 15],
    "c3alpha": [0, 2, 20, 22, 17, 19, 5, 7, 8, 10, 28, 30, 25, 27, 13, 15],
    "c3alpha^1": [0, 20, 1, 21, 18, 6, 19, 7, 8, 28, 9, 29, 26, 14, 27, 15],
    "c2e": [0, 17, 4, 5, 2, 3, 22, 7, 8, 25, 12, 13, 10, 11, 30, 15],
    "c2f": [0, 17, 20, 21, 18, 19, 22, 7, 8, 25, 28, 29, 26, 27, 30, 15],
    "c2c": [0, 4, 18, 6, 1, 21, 3, 7, 8, 12, 26, 14, 9, 29, 11, 15],
    "c2d": [0, 20, 18, 22, 17, 21, 19, 7, 8, 28, 26, 30, 25, 29, 27, 15],
    "c2a": [0, 2, 1, 19, 20, 6, 5, 7, 8, 10, 9, 27, 28, 14, 13, 15],
    "c2b": [0, 18, 17, 19, 20, 22, 21, 7, 8, 26, 25, 27, 28, 30, 29, 15],
    "riden": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "rc4x": [0, 1, 4, 21, 18, 3, 6, 7, 8, 9, 12, 29, 26, 11, 14, 15],
    "rc2x": [0, 1, 18, 19, 20, 21, 6, 7, 8, 9, 26, 27, 28, 29, 14, 15],
    "rc4x^-1": [0, 1, 20, 5, 2, 19, 6, 7, 8, 9, 28, 13, 10, 27, 14, 15],
    "rc4y": [0, 20, 2, 6, 1, 5, 19, 7, 8, 28, 10, 14, 9, 13, 27, 15],
    "rc2y": [0, 17, 2, 19, 20, 5, 22, 7, 8, 25, 10, 27, 28, 13, 30, 15],
    "rc4y^-1": [0, 4, 2, 22, 17, 5, 3, 7, 8, 12, 10, 30, 25, 13, 11, 15],
    "rc4z": [0, 2, 17, 3, 4, 22, 5, 7, 8, 10, 25, 11, 12, 30, 13, 15],
    "rc2z": [0, 17, 18, 3, 4, 21, 22, 7, 8, 25, 26, 11, 12, 29, 30, 15],
    "rc4z^-1": [0, 18, 1, 3, 4, 6, 21, 7, 8, 26, 9, 11, 12, 14, 29, 15],
    "rc3delta": [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15],
    "rc3delta^-1": [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15],
    "rc3gamma": [0, 18, 4, 22, 17, 3, 21, 7, 8, 26, 12, 30, 25, 11, 29, 15],
    "rc3gamma^-1": [0, 20, 17, 5, 2, 22, 19, 7, 8, 28, 25, 13, 10, 30, 27, 15],
    "rc3beta": [0, 18, 20, 6, 1, 19, 21, 7, 8, 26, 28, 14, 9, 27, 29, 15],
    "rc3beta^-1": [0, 4, 17, 21, 18, 22, 3, 7, 8, 12, 25, 29, 26, 30, 11, 15],
    "rc3alpha": [0, 2, 20, 22, 17, 19, 5, 7, 8, 10, 28, 30, 25, 27, 13, 15],
    "rc3alpha^1": [0, 20, 1, 21, 18, 6, 19, 7, 8, 28, 9, 29, 26, 14, 27, 15],
    "rc2e": [0, 17, 4, 5, 2, 3, 22, 7, 8, 25, 12, 13, 10, 11, 30, 15],
    "rc2f": [0, 17, 20, 21, 18, 19, 22, 7, 8, 25, 28, 29, 26, 27, 30, 15],
    "rc2c": [0, 4, 18, 6, 1, 21, 3, 7, 8, 12, 26, 14, 9, 29, 11, 15],
    "rc2d": [0, 20, 18, 22, 17, 21, 19, 7, 8, 28, 26, 30, 25, 29, 27, 15],
    "rc2a": [0, 2, 1, 19, 20, 6, 5, 7, 8, 10, 9, 27, 28, 14, 13, 15],
    "rc2b": [0, 18, 17, 19, 20, 22, 21, 7, 8, 26, 25, 27, 28, 30, 29, 15],
    "inviden": [0, 17, 18, 3, 20, 5, 6, 23, 8, 25, 26, 11, 28, 13, 14, 31],
    "invc4x": [0, 17, 20, 21, 2, 3, 6, 23, 8, 25, 28, 29, 10, 11, 14, 31],
    "invc2x": [0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31],
    "invc4x^-1": [0, 17, 4, 5, 18, 19, 6, 23, 8, 25, 12, 13, 26, 27, 14, 31],
    "invc4y": [0, 4, 18, 6, 17, 5, 19, 23, 8, 12, 26, 14, 25, 13, 27, 31],
    "invc2y": [0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31],
    "invc4y^-1": [0, 20, 18, 22, 1, 5, 3, 23, 8, 28, 26, 30, 9, 13, 11, 31],
    "invc4z": [0, 18, 1, 3, 20, 22, 5, 23, 8, 26, 9, 11, 28, 30, 13, 31],
    "invc2z": [0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31],
    "invc4z^-1": [0, 2, 17, 3, 20, 6, 21, 23, 8, 10, 25, 11, 28, 14, 29, 31],
    "invc3delta": [0, 18, 20, 6, 17, 3, 5, 23, 8, 26, 28, 14, 25, 11, 13, 31],
    "invc3delta^-1": [0, 20, 17, 5, 18, 6, 3, 23, 8, 28, 25, 13, 26, 14, 11, 31],
    "invc3gamma": [0, 2, 20, 22, 1, 3, 21, 23, 8, 10, 28, 30, 9, 11, 29, 31],
    "invc3gamma^-1": [0, 4, 1, 5, 18, 22, 19, 23, 8, 12, 9, 13, 26, 30, 27, 31],
    "invc3beta": [0, 2, 4, 6, 17, 19, 21, 23, 8, 10, 12, 14, 25, 27, 29, 31],
    "invc3beta^-1": [0, 20, 1, 21, 2, 22, 3, 23, 8, 28, 9, 29, 10, 30, 11, 31],
    "invc3alpha": [0, 18, 4, 22, 1, 19, 5, 23, 8, 26, 12, 30, 9, 27, 13, 31],
    "invc3alpha^1": [0, 4, 17, 21, 2, 6, 19, 23, 8, 12, 25, 29, 10, 14, 27, 31],
    "invc2e": [0, 1, 20, 5, 18, 3, 22, 23, 8, 9, 28, 13, 26, 11, 30, 31],
    "invc2f": [0, 1, 4, 21, 2, 19, 22, 23, 8, 9, 12, 29, 10, 27, 30, 31],
    "invc2c": [0, 20, 2, 6, 17, 21, 3, 23, 8, 28, 10, 14, 25, 29, 11, 31],
    "invc2d": [0, 4, 2, 22, 1, 21, 19, 23, 8, 12, 10, 30, 9, 29, 27, 31],
    "invc2a": [0, 18, 17, 19, 4, 6, 5, 23, 8, 26, 25, 27, 12, 14, 13, 31],
    "invc2b": [0, 2, 1, 19, 4, 22, 21, 23, 8, 10, 9, 27, 12, 30, 29, 31],
    "invriden": [0, 17, 18, 3, 20, 5, 6, 23, 8, 25, 26, 11, 28, 13, 14, 31],
    "invrc4x": [0, 17, 20, 21, 2, 3, 6, 23, 8, 25, 28, 29, 10, 11, 14, 31],
    "invrc2x": [0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31],
    "invrc4x^-1": [0, 17, 4, 5, 18, 19, 6, 23, 8, 25, 12, 13, 26, 27, 14, 31],
    "invrc4y": [0, 4, 18, 6, 17, 5, 19, 23, 8, 12, 26, 14, 25, 13, 27, 31],
    "invrc2y": [0, 1, 18, 19, 4, 5, 22, 23, 8, 9, 26, 27, 12, 13, 30, 31],
    "invrc4y^-1": [0, 20, 18, 22, 1, 5, 3, 23, 8, 28, 26, 30, 9, 13, 11, 31],
    "invrc4z": [0, 18, 1, 3, 20, 22, 5, 23, 8, 26, 9, 11, 28, 30, 13, 31],
    "invrc2z": [0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31],
    "invrc4z^-1": [0, 2, 17, 3, 20, 6, 21, 23, 8, 10, 25, 11, 28, 14, 29, 31],
    "invrc3delta": [0, 18, 20, 6, 17, 3, 5, 23, 8, 26, 28, 14, 25, 11, 13, 31],
    "invrc3delta^-1": [0, 20, 17, 5, 18, 6, 3, 23, 8, 28, 25, 13, 26, 14, 11, 31],
    "invrc3gamma": [0, 2, 20, 22, 1, 3, 21, 23, 8, 10, 28, 30, 9, 11, 29, 31],
    "invrc3gamma^-1": [0, 4, 1, 5, 18, 22, 19, 23, 8, 12, 9, 13, 26, 30, 27, 31],
    "invrc3beta": [0, 2, 4, 6, 17, 19, 21, 23, 8, 10, 12, 14, 25, 27, 29, 31],
    "invrc3beta^-1": [0, 20, 1, 21, 2, 22, 3, 23, 8, 28, 9, 29, 10, 30, 11, 31],
    "invrc3alpha": [0, 18, 4, 22, 1, 19, 5, 23, 8, 26, 12, 30, 9, 27, 13, 31],
    "invrc3alpha^1": [0, 4, 17, 21, 2, 6, 19, 23, 8, 12, 25, 29, 10, 14, 27, 31],
    "invrc2e": [0, 1, 20, 5, 18, 3, 22, 23, 8, 9, 28, 13, 26, 11, 30, 31],
    "invrc2f": [0, 1, 4, 21, 2, 19, 22, 23, 8, 9, 12, 29, 10, 27, 30, 31],
    "invrc2c": [0, 20, 2, 6, 17, 21, 3, 23, 8, 28, 10, 14, 25, 29, 11, 31],
    "invrc2d": [0, 4, 2, 22, 1, 21, 19, 23, 8, 12, 10, 30, 9, 29, 27, 31],
    "invrc2a": [0, 18, 17, 19, 4, 6, 5, 23, 8, 26, 25, 27, 12, 14, 13, 31],
    "invrc2b": [0, 2, 1, 19, 4, 22, 21, 23, 8, 10, 9, 27, 12, 30, 29, 31],
    "conj": [0, 17, 18, 19, 20, 21, 22, 7, 24, 25, 26, 11, 28, 13, 14, 15],
}


def gamma_transform(name: str, gamma_idx: int):
    return _gamma_transform_dict[name][gamma_idx]
