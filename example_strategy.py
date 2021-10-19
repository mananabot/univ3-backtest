from univ3backtest.backtest import CompetitionBacktest
from univ3backtest.strategy import StrategyBase, OutputRangesCredmark

import json

import pandas as pd
import numpy as np


class SampleStrategyCredmark(StrategyBase):
    """
    Sets fixed pct range on both sides, rebalances on price cross
    """

    def __init__(self, params: dict, risk: str):
        super().__init__(params)
        self.output_ranges = OutputRangesCredmark(risk)

    def update(self, input_df: pd.DataFrame) -> None:
        """
        This is the only method that must be implemented.

        It receives input_df with latest 'lookback' datapoints
        and decides to append new range to OutputRangesCredmark

        The strategy updates ranges only if they are not set or on price cross
        """
        price = input_df.iloc[-1, :].token1Price

        range_empty = self.output_ranges.empty
        in_range = self.output_ranges.price_in_range(price)

        if range_empty or not in_range:
            timestamp = input_df.index[-1]
            self.set_new_ranges(timestamp, price)

    def set_new_ranges(self, timestamp: pd.Timestamp, price: float) -> None:

        pct_range_abs = self.params['pct_range'] * price

        price_range_upper = max(0, price + pct_range_abs)
        price_range_lower = max(0, price - pct_range_abs)

        self.output_ranges.append(
            timestamp,
            price_range_lower,
            price_range_upper
        )


if __name__ == "__main__":
    """
    Example backtest run, params selection and submission generation
    """

    ##
    # 1. Choose best strategy params in a backtest
    ##

    swap_filepaths = [
        "competition_data/train/train1Swap.csv",  # CMK/USDC
        "competition_data/train/train2Swap.csv",  # USDC/USDT
        "competition_data/train/train3Swap.csv"   # WBTC/WETH
    ]

    pools = [
        "CMK/USDC",
        "USDC/USDT",
        "WBTC/WETH"
    ]

    decimals = [
        [18, 6],  # CMK/USDC
        [6, 6],   # USDC/USDT
        [8, 18]   # WBTC/WETH
    ]

    fees = [
        0.01,    # CMK/USDC
        0.0005,  # USDC/USDT
        0.003    # WBTC/WETH
    ]

    amounts_initial_x = [
        300000,  # CMK
        100000,  # USDC
        2        # WBTC
    ]

    # Params for SampleStrategy that will be iterated over
    strategy_params = {
        "low": {
            'pct_range': [0.2, 0.3]
        },
        "medium": {
            'pct_range': [0.1, 0.15]
        },
        "high": {
            'pct_range': [0.025, 0.05]
        }
    }

    best_param_combinations = {
        "low": [],
        "medium": [],
        "high": []
    }

    plot_backtest_results = True

    pool_params = zip(swap_filepaths, pools, decimals, fees, amounts_initial_x)

    for filepath, pool, decimal, fee, amount_x_initial in pool_params:

        # Init 1 instance of SampleStrategy per 1 param combination
        strategies = {
            k: SampleStrategyCredmark.from_params(
                params=strategy_params[k],
                risk=k
            )
            for k, v in strategy_params.items()
        }

        # Number of last datapoints for strategy input,
        # we actually don't need any lookback for our example strategy
        lookback = 1

        # Run backtest of every SampleStrategy param combination
        # for each training pool
        backtest_results = {}
        backtest_ranges = {}

        for risk in strategies.keys():

            backtest = CompetitionBacktest(
                data_filepath=filepath,
                strategies=strategies[risk],
                amount_x_initial=amount_x_initial,
                decimals_0=decimal[0],
                decimals_1=decimal[1],
                fee=fee,
                lookback=lookback
            )

            result = backtest.run().get_result()
            ranges = backtest.get_ranges()

            backtest_results[risk] = result
            backtest_ranges[risk] = ranges

            param_combinations = {
                "low": [],
                "medium": [],
                "high": []
            }

            i = 0
            for res in result:

                # Take out the param combination from strategy
                # TODO refactor
                params = backtest.strategies[i].params

                params_for_strategy = params.copy()
                params_for_strategy['risk'] = risk

                param_combinations[risk].append(params)

                if plot_backtest_results:
                    CompetitionBacktest.plot_backtest_result(
                        result_df=res,
                        params=params_for_strategy,
                        pool=pool
                    )

                i += 1

            # Choose best param combination for risk level by highest PnL
            best_pnl_idx = np.argmax(
                [res.pnl_relative.iloc[-1] for res in result]
            )

            best_param_combination = param_combinations[risk][best_pnl_idx]
            best_param_combinations[risk].append(best_param_combination)

            del backtest

    # Arbitrarily choose best params for submission
    # (best params from last backtest)
    best_param_combinations = {
        k: v[-1]
        for k, v in best_param_combinations.items()
    }

    print(f"Chosen best params for submission: {best_param_combinations}")

    ##
    # 2. Produce submission on test data
    ##
    submission = {
        "test1": {
            "positions": {
                "low": {},
                "medium": {},
                "high": {}
            }
        },
        "test2": {
            "positions": {
                "low": {},
                "medium": {},
                "high": {}
            }
        }
    }

    test_filepaths = [
        "competition_data/test/test1Price.csv",
        "competition_data/test/test2Price.csv"
    ]

    for filepath in test_filepaths:

        price_df = pd.read_csv(
            filepath,
            parse_dates=['datetime'],
            index_col='datetime'
        ).iloc[:, 1:]

        for risk in ["low", "medium", "high"]:

            strategy_best_params = best_param_combinations[risk]

            strategy = SampleStrategyCredmark(
                params=strategy_best_params,
                risk=risk
            )

            output_ranges = CompetitionBacktest.simulate(
                price_df=price_df,
                strategy=strategy,
                lookback=1
            )

            sub_name = filepath.split("/")[-1].split(".")[0][:5]

            submission[sub_name]['positions'][risk] = output_ranges

            title = f"Submission ranges for '{sub_name}', {risk} risk"

            CompetitionBacktest.plot_simulation(
                title=title,
                price_df=price_df,
                output_ranges=output_ranges
            )

# Save sumbission to json
with open('submission.json', 'w') as fp:
    json.dump(submission, fp, sort_keys=True, indent=4)
