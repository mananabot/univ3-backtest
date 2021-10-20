from __future__ import annotations
from abc import abstractmethod, ABC
from typing import List, Any, Optional, Tuple

import pandas as pd

from univ3backtest.utils import (
    all_combinations,
    L_x,
    L_position,
    L_position_X96,
    calculate_x, calculate_y,
    tick_from_price_1_human,
    tick_spacing, decimal_adjustment
)


class StrategyState:
    """
    Monitors state of a strategy at a given timestamp

    Does all the calculation on swap datapoint

    So far works only with Credmark data
    """

    def __init__(
        self,
        upper_range_price: Optional[float] = None,
        lower_range_price: Optional[float] = None,
        amount_x: Optional[int] = None,
        amount_y: Optional[int] = None,
        position_liquidity: Optional[float] = None,
        position_liquidity_X96: Optional[int] = None
    ):
        self.upper_range_price = upper_range_price
        self.lower_range_price = lower_range_price

        self.amount_x = amount_x
        self.amount_y = amount_y

        self.position_liquidity = position_liquidity
        self.position_liquidity_X96 = position_liquidity_X96

        self.timestamp: pd.Timestamp = None
        self.price_1 = None

        self.amount_x_initial = None
        self.amount_y_initial = None

        self.upper_range_tick = None
        self.lower_range_tick = None

        self.accrued_fees_0 = 0.0
        self.accrued_fees_1 = 0.0

        self.accrued_fees_total_0 = 0.0
        self.accrued_fees_total_1 = 0.0

        self.accrued_fees_1_relative = 0.0

        self.impermanent_loss_relative = 0.0
        self.impermanent_loss_absolute = 0.0

        self.pnl_absolute = 0.0
        self.pnl_relative = 0.0

        self.upper_range_price_last: float = None
        self.lower_range_price_last: float = None

        self.rebalanced = False

    def update(
        self,
        timestamp: pd.Timestamp,
        tick: int,
        price_1: float,
        amount_0: float,
        amount_1: float,
        liquidity: int,
        decimals_0: int,
        decimals_1: int,
        fee: float,
        output_ranges: OutputRangesCredmark,
        amount_x_initial: Optional[int] = None
    ) -> None:


        # Set timestamp and price of interest
        self.timestamp = timestamp
        self.price_1 = price_1

        self.rebalanced = False

        # There are no ranges yet
        if not output_ranges.date:
            return

        # Update price ranges
        self.upper_range_price = output_ranges.positionPriceUpper[-1]
        self.lower_range_price = output_ranges.positionPriceLower[-1]

        self.upper_range_tick, self.lower_range_tick = self.get_tick_ranges(
            decimals_0,
            decimals_1,
            fee
        )

        # Rebalance on price range change (clain fees and reset amounts)
        # TODO: implement rebalance swap fees and gas fees
        if (self.upper_range_price_last is not None and
            self.lower_range_price_last is not None) and \
           (self.upper_range_price != self.upper_range_price_last or
            self.lower_range_price != self.lower_range_price_last):

            # Claim & reset accrued fees
            self.amount_x += self.accrued_fees_0
            self.amount_y += self.accrued_fees_1

            self.accrued_fees_total_0 += self.accrued_fees_0
            self.accrued_fees_total_1 += self.accrued_fees_1

            self.accrued_fees_0 = 0.0
            self.accrued_fees_1 = 0.0

            # Reset amounts reinvesting claimed fees
            amount_x_rebalance = self.amount_x + self.amount_y / price_1

            Lx = L_x(
                x=amount_x_rebalance,
                P=price_1,
                P_up=self.upper_range_price
            )

            amount_y_rebalance = calculate_y(
                L=Lx,
                P=price_1,
                P_up=self.upper_range_price,
                P_down=self.lower_range_price
            )

            rebalance_ratio = amount_x_rebalance / (
                amount_y_rebalance / price_1 + amount_x_rebalance
            )

            self.amount_x = amount_x_rebalance * rebalance_ratio
            self.amount_y = amount_x_rebalance * (1 - rebalance_ratio) * price_1

            self.rebalanced = True

        self.upper_range_price_last = self.upper_range_price
        self.lower_range_price_last = self.lower_range_price

        # Set initial amount and position liquidity
        if self.amount_x is None and self.amount_y is None:

            assert amount_x_initial is not None, \
                "'amount_x_initial' must be provided"

            self.amount_x = amount_x_initial

            # Calculate liquidity for x
            Lx = L_x(
                x=amount_x_initial,
                P=price_1,
                P_up=self.upper_range_price
            )

            # Calculate y amount and set initial amounts
            self.amount_y = calculate_y(
                L=Lx,
                P=price_1,
                P_up=self.upper_range_price,
                P_down=self.lower_range_price
            )

            self.amount_x_initial = self.amount_x
            self.amount_y_initial = self.amount_y

            # Update liquidity
            self.position_liquidity = L_position(
                x=self.amount_x,
                y=self.amount_y,
                P=price_1,
                P_up=self.upper_range_price,
                P_down=self.lower_range_price
            )

            self.position_liquidity_X96 = L_position_X96(
                tick=tick,
                tick_up=self.upper_range_tick,
                tick_down=self.lower_range_tick,
                x=self.amount_x,
                y=self.amount_y,
                decimals_0=decimals_0,
                decimals_1=decimals_1
            )

            return

        # Update amounts and position liquidity after a price change
        if not self.rebalanced:
            self.amount_x = calculate_x(
                L=self.position_liquidity,
                P=price_1,
                P_up=self.upper_range_price,
                P_down=self.lower_range_price
            )

            self.amount_y = calculate_y(
                L=self.position_liquidity,
                P=price_1,
                P_up=self.upper_range_price,
                P_down=self.lower_range_price
            )

        # Accrue fees
        self.accrue_fees(tick, price_1, amount_0, amount_1, liquidity, fee)

        # Calculate impermanent loss
        self.update_impermanent_loss(price_1)

        # Update liquidity
        self.position_liquidity = L_position(
            x=self.amount_x,
            y=self.amount_y,
            P=price_1,
            P_up=self.upper_range_price,
            P_down=self.lower_range_price
        )

        self.position_liquidity_X96 = L_position_X96(
            tick=tick,
            tick_up=self.upper_range_tick,
            tick_down=self.lower_range_tick,
            x=self.amount_x,
            y=self.amount_y,
            decimals_0=decimals_0,
            decimals_1=decimals_1
        )

        # Update accred fees and PnL
        accrued_fees_1_current = \
            self.accrued_fees_1 + \
            self.accrued_fees_total_1

        self.accrued_fees_1_relative = \
            accrued_fees_1_current / self.amount_y_initial

        self.pnl_absolute = \
            accrued_fees_1_current + \
            self.impermanent_loss_absolute

        self.pnl_relative = \
            self.accrued_fees_1_relative + \
            self.impermanent_loss_relative

    def update_impermanent_loss(self, price_1: float) -> None:
        """
        Impermanent loss calculated on asset Y
        """
        y_portfolio_value_hold = \
            self.amount_x_initial * price_1 + self.amount_y_initial

        y_portfolio_value_lp = \
            self.amount_x * price_1 + self.amount_y - \
            (self.accrued_fees_total_0 * price_1 + self.accrued_fees_total_1)

        self.impermanent_loss_relative = \
            y_portfolio_value_lp / y_portfolio_value_hold - 1

        self.impermanent_loss_absolute = \
            y_portfolio_value_lp - y_portfolio_value_hold

    def accrue_fees(
        self,
        tick: int,
        price_1: float,
        amount_0: float,
        amount_1: float,
        liquidity: float,
        fee: float
    ) -> None:

        in_range = self.lower_range_tick <= tick <= self.upper_range_tick

        pool_liquidity = liquidity + self.position_liquidity_X96

        fee_fraction = self.position_liquidity_X96 / pool_liquidity

        if amount_0 > 0:
            self.accrued_fees_0 += in_range * fee * fee_fraction * amount_0
        else:
            self.accrued_fees_1 += in_range * fee * fee_fraction * amount_1

    def update_position_liquidity(self, tick: int) -> None:
        pass

    def update_accrued_fees(self, swaps: Any) -> None:
        pass

    def get_tick_ranges(
        self,
        decimals_0: int,
        decimals_1: int,
        fee: float
    ) -> Tuple[int, int]:

        adjustment = decimal_adjustment(decimals_0, decimals_1)
        spacing = tick_spacing(fee)

        tick_upper = tick_from_price_1_human(
            self.upper_range_price,
            adjustment,
            spacing
        )
        tick_lower = tick_from_price_1_human(
            self.lower_range_price,
            adjustment,
            spacing
        )

        return tick_upper, tick_lower

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class OutputRangesBase(ABC):
    ...


class OutputRangesCredmark(OutputRangesBase):

    def __init__(self, risk: str):
        self.risk = risk

        self.date = []
        self.positionPriceLower = []
        self.positionPriceUpper = []

    def append(
        self,
        date: pd.Timestamp,
        positionPriceLower: float,
        positionPriceUpper: float
    ) -> OutputRangesCredmark:
        self.date.append(date)
        self.positionPriceLower.append(positionPriceLower)
        self.positionPriceUpper.append(positionPriceUpper)
        return self

    @property
    def empty(self) -> bool:
        return not self.date

    def price_in_range(self, price: float) -> bool:
        if self.empty:
            return False

        return price >= self.positionPriceLower[-1] \
            and price <= self.positionPriceUpper[-1]

    def as_dict(self) -> dict:
        return {
            'date': [d.strftime('%Y-%m-%d %H:%M:%S:%f') for d in self.date],
            'positionPriceLower': self.positionPriceLower,
            'positionPriceUpper': self.positionPriceUpper
        }


class StrategyBase(ABC):

    def __init__(self, params: dict):
        self.params = params
        self.state = StrategyState()
        self.output_ranges: OutputRangesBase = None

    @abstractmethod
    def update(self, input: Any) -> OutputRangesBase:
        """ Implement strategy logic here """
        ...

    @classmethod
    def from_params(cls, params: dict, **kwargs) -> List[StrategyBase]:
        return [
            cls(params=param_combination, **kwargs)
            for param_combination in all_combinations(params)
        ]

    def update_state(self, *args, **kwargs) -> None:
        self.state.update(*args, **kwargs)
