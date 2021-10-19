from typing import Dict, List, Any
from itertools import product

import numpy as np


def all_combinations(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    return list(map(
        lambda x: {
            list(params.keys())[i]: t_ for i, t_ in enumerate(x)
        },
        product(
            *(params.values())
        )
    ))


def decimal_adjustment(decimals_0: int, decimals_1: int) -> int:
    return 10 ** (decimals_0 - decimals_1)


def tick_spacing(fee: float) -> int:
    return int(fee * 2 * 10000)


def tick_to_price(tick: int) -> float:
    return 1.0001 ** tick


def tick_from_price_1_human(
    price_1: float,
    decimal_adjustment: int,
    tick_spacing: int
) -> int:

    price_1_decimal_adjusted = price_1 / decimal_adjustment

    tick = np.floor(
        np.log(price_1_decimal_adjusted) /
        np.log(1.0001)
    )

    return int(tick / tick_spacing) * tick_spacing


def L_x(x: float, P: float, P_up: float) -> float:
    P_sq = np.sqrt(P)
    P_up_sq = np.sqrt(P_up)
    return x * P_sq * P_up_sq / (P_up_sq - P_sq)


def L_y(y: float, P: float, P_down: float) -> float:
    P_sq = np.sqrt(P)
    P_down_sq = np.sqrt(P_down)
    return y / (P_sq - P_down_sq)


def L_x_X96(P_up_X96, P_down_X96, x: float, decimals_0: int):

    if P_down_X96 > P_up_X96:
        P_down_X96, P_up_X96 = P_up_X96, P_down_X96

    return int(
        x / (
            (2**96 * (P_up_X96 - P_down_X96) / P_up_X96 / P_down_X96) /
            10**decimals_0
        )
    )


def L_y_X96(P_up_X96, P_down_X96, y: float, decimals_1: int):

    if P_down_X96 > P_up_X96:
        P_down_X96, P_up_X96 = P_up_X96, P_down_X96

    return int(
        y / (
            (P_up_X96 - P_down_X96) / 2**96 /
            10**decimals_1
        )
    )


def L_position_X96(
    tick: int,
    tick_up: int,
    tick_down: int,
    x: float,
    y: float,
    decimals_0: int,
    decimals_1: int
) -> int:

    P_X96 = int(
        1.0001**(tick/2) * (2**96)
    )
    P_up_X96 = int(
        1.0001**(tick_up/2) * (2**96)
    )
    P_down_X96 = int(
        1.0001**(tick_down/2) * (2**96)
    )

    if P_down_X96 > P_up_X96:
        P_down_X96, P_up_X96 = P_up_X96, P_down_X96

    # All assets in X
    if P_X96 <= P_down_X96:
        return L_x_X96(P_up_X96, P_down_X96, x, decimals_0)

    # In range
    if P_up_X96 > P_X96 > P_down_X96:
        return min(
            L_x_X96(P_X96, P_down_X96, x, decimals_0),
            L_y_X96(P_up_X96, P_X96, y, decimals_1)
        )

    # All assets in Y
    else:
        return L_y_X96(P_up_X96, P_down_X96, y, decimals_1)


def L_position(
    x: float,
    y: float,
    P: float,
    P_up: float,
    P_down: float
) -> float:

    # All assets in X
    if P <= P_down:
        return L_x(x, P_down, P_up)

    # In range
    if P_up > P > P_down:
        return min(
            L_x(x, P, P_up),
            L_y(y, P, P_down)
        )

    # All assets in Y
    if P >= P_up:
        return L_y(y, P_up, P_down)


def calculate_x(L, P, P_up, P_down) -> float:
    P_sq = np.sqrt(P)
    P_down_sq = np.sqrt(P_down)
    P_up_sq = np.sqrt(P_up)
    P_sq = max(min(P_sq, P_up_sq), P_down_sq)
    return L * (P_up_sq - P_sq) / (P_sq * P_up_sq)


def calculate_y(L, P, P_up, P_down) -> float:
    P_sq = np.sqrt(P)
    P_down_sq = np.sqrt(P_down)
    P_up_sq = np.sqrt(P_up)
    P_sq = max(min(P_sq, P_up_sq), P_down_sq)
    return L * (P_sq - P_down_sq)
