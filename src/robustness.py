
import pandas as pd

from src.backtest import run_momentum_backtest
from src.features import build_features


def add_dynamic_leverage(
    data,
    target_vol=0.02,
    max_leverage=4.0,
    vol_window=14,
):
    """
    Add paper-style volatility-based position sizing.

    Daily leverage =
        min(max_leverage, target_vol / recent daily volatility)

    The volatility estimate uses information available before
    the current trading day.
    """

    df = data.copy()

    daily = (
        df.groupby("date")
          .agg(
              close=("close", "last")
          )
          .reset_index()
    )

    daily["spy_return"] = (
        daily["close"]
        .pct_change()
    )

    daily["vol_14"] = (
        daily["spy_return"]
        .rolling(
            window=vol_window,
            min_periods=vol_window
        )
        .std(ddof=1)
        .shift(1)
    )

    daily["dynamic_leverage"] = (
        target_vol
        / daily["vol_14"]
    )

    daily["dynamic_leverage"] = (
        daily["dynamic_leverage"]
        .clip(
            upper=max_leverage
        )
    )

    df = df.merge(
        daily[
            [
                "date",
                "vol_14",
                "dynamic_leverage"
            ]
        ],
        on="date",
        how="left"
    )

    return df


def evaluate_parameter_set(
    raw_data,
    performance_function,
    train_end,
    lookback=14,
    vol_multiplier=1.0,
    initial_capital=100_000,
    commission_per_share=0.0035,
    slippage_per_share=0.001,
    target_vol=0.02,
    max_leverage=4.0,
):
    """
    Evaluate one parameter specification on the train sample.
    """

    features = build_features(
        raw_data,
        lookback=lookback,
        vol_multiplier=vol_multiplier
    )

    features = add_dynamic_leverage(
        features,
        target_vol=target_vol,
        max_leverage=max_leverage
    )

    train = features.loc[
        features["date"] <= train_end
    ].copy()

    daily, trades = run_momentum_backtest(
        train,
        initial_capital=initial_capital,
        commission_per_share=commission_per_share,
        slippage_per_share=slippage_per_share,
        stop_mode="band_vwap",
        sizing_mode="dynamic"
    )

    metrics = performance_function(
        daily["return"]
    )

    metrics["Trades"] = len(trades)

    metrics["Average Trades / Day"] = (
        len(trades) / len(daily)
        if len(daily) > 0
        else 0
    )

    return metrics
