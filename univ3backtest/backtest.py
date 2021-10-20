from __future__ import annotations
from abc import abstractmethod, ABC
from typing import List, Any, Union, Optional

import pandas as pd
import matplotlib.pyplot as plt

from univ3backtest.strategy import StrategyBase, OutputRangesCredmark


class BacktestBase(ABC):

    def __init__(
        self,
        data_filepath: Union[str, List[str]],
        strategies: List[StrategyBase],
        amount_x_initial: int,
        decimals_0: int,
        decimals_1: int,
        fee: float
    ):
        self.strategies = strategies
        self.data = self._parse_data(data_filepath)

        self.amount_x_initial = amount_x_initial
        self.decimals_0 = decimals_0
        self.decimals_1 = decimals_1
        self.fee = fee

    def run(self) -> BacktestBase:
        self._run_main_loop()
        return self

    @abstractmethod
    def _parse_data(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def _run_main_loop(self) -> None:
        ...

    @abstractmethod
    def get_result(self) -> Any:
        ...


class CompetitionBacktest(BacktestBase):
    """
    If 'lookback' is None, the backtest passes only 1 row
    to strategy at a time.
    """

    def __init__(
        self,
        data_filepath: Union[str, List[str]],
        strategies: List[StrategyBase],
        amount_x_initial: int,
        decimals_0: int,
        decimals_1: int,
        fee: float,
        lookback: Optional[int] = None
    ):
        self.lookback = lookback
        self._result: List[List[dict]] = [[] for s in strategies]

        super().__init__(
            data_filepath,
            strategies,
            amount_x_initial,
            decimals_0,
            decimals_1,
            fee
        )

    def _parse_data(self, swap_filepath: str) -> pd.DataFrame:
        return pd.read_csv(
            swap_filepath,
            parse_dates=['datetime'],
            index_col='datetime'
        ).iloc[:, 1:]

    def _run_main_loop(self) -> None:
        if self.lookback is None or self.lookback == 0:
            self._run_main_loop_one()
        else:
            self._run_main_loop_lookback()

    def _run_main_loop_lookback(self) -> None:

        data_len = self.data.shape[0]

        for i in range(self.lookback, data_len):

            input_df = self.data.iloc[i-self.lookback:i, :]

            for strategy, result in zip(self.strategies, self._result):

                # Update strategy
                strategy.update(input_df)

                # Update strategy state
                strategy.update_state(
                    timestamp=input_df.index[-1],
                    tick=input_df.tick[-1],
                    price_1=input_df.token1Price[-1],
                    amount_0=input_df.amount0[-1],
                    amount_1=input_df.amount1[-1],
                    liquidity=input_df.liquidity[-1],
                    decimals_0=self.decimals_0,
                    decimals_1=self.decimals_1,
                    fee=self.fee,
                    output_ranges=strategy.output_ranges,
                    amount_x_initial=self.amount_x_initial
                )

                # Save last state
                state = strategy.state.as_dict()
                result.append(state)

    def _run_main_loop_one(self) -> None:

        data_records = self.data.reset_index().to_dict('records')

        for input_row in data_records:
            for strategy, result in zip(self.strategies, self._result):

                # Update strategy
                strategy.update(input_row)

                # Update strategy state
                strategy.update_state(
                    timestamp=input_row['datetime'],
                    tick=input_row['tick'],
                    price_1=input_row['token1Price'],
                    amount_0=input_row['amount0'],
                    amount_1=input_row['amount1'],
                    liquidity=input_row['liquidity'],
                    decimals_0=self.decimals_0,
                    decimals_1=self.decimals_1,
                    fee=self.fee,
                    output_ranges=strategy.output_ranges,
                    amount_x_initial=self.amount_x_initial
                )

                # Save last state
                state = strategy.state.as_dict()
                result.append(state)

    @staticmethod
    def simulate(
        price_df: pd.DataFrame,
        strategy: StrategyBase,
        lookback: int
    ) -> dict:
        """
        Simulate a strategy on price (test) data
        """
        # TODO make it work for lookback=0 as well (pass as dict)

        data_len = price_df.shape[0]

        for i in range(lookback, data_len):

            input_df = price_df.iloc[i-lookback:i, :]

            strategy.update(input_df)

        return strategy.output_ranges.as_dict()

    @staticmethod
    def plot_simulation(
        title: str,
        price_df: pd.DataFrame,
        output_ranges: dict
    ) -> None:

        output_ranges_df = pd.DataFrame(output_ranges)

        output_ranges_df.date = pd.to_datetime(
            output_ranges_df.date,
            format='%Y-%m-%d %H:%M:%S:%f'
        )

        output_ranges_df.set_index('date', inplace=True)

        df_all = pd.concat([price_df, output_ranges_df]) \
            .sort_index() \
            .fillna(method='pad') \
            .dropna()

        plt.title(title)
        plt.plot(df_all.token1Price, label='Price')
        plt.plot(df_all.positionPriceLower, label='Lower range')
        plt.plot(df_all.positionPriceUpper, label='Upper range')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_backtest_result(
        result_df: pd.DataFrame,
        params: Optional[dict] = None,
        pool: Optional[str] = None
    ) -> None:

        def set_title(
            pool: Union[str, None],
            params: Union[dict, None]
        ) -> None:

            title = f"{pool} pool\n" if pool is not None else ""

            if params is not None:

                prms = [
                    f" {k} = {v} |"
                    for k, v in params.items()
                ]

                prms[-1] = prms[-1][:-2]

                title += "Params:\n" + "".join(prms)

            plt.title(title)

        # Price and ranges
        plt.plot(
            result_df.timestamp,
            result_df.price_1,
            label='Asset Y price')
        plt.plot(
            result_df.timestamp,
            result_df.upper_range_price_last,
            label="Upper range"
        )
        plt.plot(
            result_df.timestamp,
            result_df.lower_range_price_last,
            label="Lower range"
        )
        plt.ylabel("Price")
        plt.legend()

        set_title(pool, params)

        plt.show()

        # Impermanent loss, PnL
        # TODO calc and print APR to plot
        plt.plot(
            result_df.timestamp,
            (result_df.impermanent_loss_relative * 100).round(2),
            label="Impermanent loss [%]"
        )
        plt.plot(
            result_df.timestamp,
            (result_df.accrued_fees_1_relative * 100).round(2),
            label="Accrued fees [%]"
        )
        plt.plot(
            result_df.timestamp,
            (result_df.pnl_relative * 100).round(2),
            label="PnL [%]"
        )

        set_title(pool, params)

        plt.ylabel("%")
        plt.legend()
        plt.show()

    def get_result(self) -> List[pd.DataFrame]:
        return [pd.DataFrame(res) for res in self._result]

    def get_ranges(self) -> List[OutputRangesCredmark]:
        return [
            strategy.output_ranges.as_dict()
            for strategy in self.strategies
        ]
