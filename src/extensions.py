
import numpy as np
import pandas as pd

from src.features import build_features


def build_adaptive_vm_features(
    data,
    lookback=14,
    vol_window=14,
    regime_reference_window=60,
    low_ratio=0.80,
    high_ratio=1.20,
    low_vm=0.75,
    normal_vm=1.00,
    high_vm=1.25,
    target_vol=0.02,
    max_leverage=4.0,
):
    """
    Original extension:
    volatility-regime adaptive Noise Area.

    The Noise Area multiplier changes according to recent
    daily volatility relative to its trailing historical median.

    All volatility measures use information available before
    the current trading session.
    """

    # Start from paper-style features
    df = build_features(
        data,
        lookback=lookback,
        vol_multiplier=1.0
    )

    # DAILY VOLATILITY

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

    # Available before day t
    daily["vol_14"] = (
        daily["spy_return"]
        .rolling(
            window=vol_window,
            min_periods=vol_window
        )
        .std(ddof=1)
        .shift(1)
    )

    # Historical reference level
    daily["vol_reference"] = (
        daily["vol_14"]
        .rolling(
            window=regime_reference_window,
            min_periods=regime_reference_window
        )
        .median()
    )

    daily["vol_ratio"] = (
        daily["vol_14"]
        / daily["vol_reference"]
    )

    # VOLATILITY REGIME

    daily["vol_regime"] = "Normal"

    daily.loc[
        daily["vol_ratio"] < low_ratio,
        "vol_regime"
    ] = "Low"

    daily.loc[
        daily["vol_ratio"] > high_ratio,
        "vol_regime"
    ] = "High"

    # ADAPTIVE VOLATILITY MULTIPLIER

    daily["adaptive_vm"] = normal_vm

    daily.loc[
        daily["vol_regime"] == "Low",
        "adaptive_vm"
    ] = low_vm

    daily.loc[
        daily["vol_regime"] == "High",
        "adaptive_vm"
    ] = high_vm

    # DYNAMIC POSITION SIZING

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

    # MERGE DAILY VARIABLES

    df = df.merge(
        daily[
            [
                "date",
                "vol_14",
                "vol_reference",
                "vol_ratio",
                "vol_regime",
                "adaptive_vm",
                "dynamic_leverage",
            ]
        ],
        on="date",
        how="left"
    )

    # SAVE PAPER BANDS FOR DIAGNOSTICS

    df["paper_upper_band"] = (
        df["upper_band"]
    )

    df["paper_lower_band"] = (
        df["lower_band"]
    )

    # REBUILD NOISE AREA USING ADAPTIVE VM

    upper_reference = np.maximum(
        df["day_open"],
        df["prev_close"]
    )

    lower_reference = np.minimum(
        df["day_open"],
        df["prev_close"]
    )

    df["upper_band"] = (
        upper_reference
        * (
            1
            + df["adaptive_vm"]
            * df["sigma_14"]
        )
    )

    df["lower_band"] = (
        lower_reference
        * (
            1
            - df["adaptive_vm"]
            * df["sigma_14"]
        )
    )

    # REBUILD RAW SIGNAL USING ADAPTIVE BANDS

    df["raw_signal"] = 0

    df.loc[
        df["decision_time"]
        & (
            df["open"]
            > df["upper_band"]
        ),
        "raw_signal"
    ] = 1

    df.loc[
        df["decision_time"]
        & (
            df["open"]
            < df["lower_band"]
        ),
        "raw_signal"
    ] = -1

    return df


def add_confirmation_signal(data):
    """
    Require two consecutive semi-hourly momentum signals
    in the same direction before allowing a new entry.

    confirmed_signal:
         1 = confirmed long
        -1 = confirmed short
         0 = no confirmed entry
    """

    df = data.copy()

    df["confirmed_signal"] = 0

    # Work only with actual strategy decision timestamps
    decision_mask = df["decision_time"]

    decisions = (
        df.loc[
            decision_mask,
            [
                "date",
                "datetime",
                "raw_signal"
            ]
        ]
        .copy()
    )

    # Previous semi-hourly signal within the SAME trading day
    decisions["previous_signal"] = (
        decisions.groupby("date")["raw_signal"]
                 .shift(1)
    )

    decisions["confirmed_signal"] = 0

    # Two consecutive bullish breakouts
    decisions.loc[
        (decisions["raw_signal"] == 1)
        &
        (decisions["previous_signal"] == 1),
        "confirmed_signal"
    ] = 1

    # Two consecutive bearish breakouts
    decisions.loc[
        (decisions["raw_signal"] == -1)
        &
        (decisions["previous_signal"] == -1),
        "confirmed_signal"
    ] = -1

    df.loc[
        decisions.index,
        "confirmed_signal"
    ] = decisions["confirmed_signal"]

    return df
