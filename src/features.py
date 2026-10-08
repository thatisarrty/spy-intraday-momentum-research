
import numpy as np
import pandas as pd


def prepare_datetime(data):
    """
    Standardize the timestamp column and convert timestamps
    to timezone-naive New York market time.
    """

    df = data.copy()

    # Standardize column name

    if "datetime" not in df.columns:

        if "timestamp" in df.columns:
            df = df.rename(
                columns={"timestamp": "datetime"}
            )

        else:
            raise ValueError(
                "Neither 'datetime' nor 'timestamp' column found."
            )

    # Parse timestamps

    if not pd.api.types.is_datetime64_any_dtype(
        df["datetime"]
    ):

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            utc=True
        )

        df["datetime"] = (
            df["datetime"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

    elif getattr(
        df["datetime"].dt,
        "tz",
        None
    ) is not None:

        df["datetime"] = (
            df["datetime"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

    df = (
        df.sort_values("datetime")
          .reset_index(drop=True)
    )

    return df


def build_features(
    data,
    lookback=14,
    vol_multiplier=1.0
):
    """
    Construct the features required for the SPY intraday
    momentum strategy.

    Includes:
    - date/time variables
    - daily open
    - previous close
    - absolute movement from daily open
    - time-of-day rolling average movement
    - upper/lower Noise Area boundaries
    - cumulative session VWAP
    - lagged session VWAP
    - semi-hourly decision timestamps
    - raw momentum signal
    """

    df = prepare_datetime(data)

    # TIME VARIABLES

    df["date"] = (
        df["datetime"]
        .dt.normalize()
    )

    df["time"] = (
        df["datetime"]
        .dt.time
    )

    df["hour"] = (
        df["datetime"]
        .dt.hour
    )

    df["minute"] = (
        df["datetime"]
        .dt.minute
    )

    df["minute_of_day"] = (
        df["hour"] * 60
        + df["minute"]
    )

    # DAILY OPEN

    df["day_open"] = (
        df.groupby("date")["open"]
          .transform("first")
    )

    # PREVIOUS TRADING DAY CLOSE

    daily_close = (
        df.groupby("date")["close"]
          .last()
    )

    prev_close = (
        daily_close.shift(1)
    )

    df["prev_close"] = (
        df["date"]
        .map(prev_close)
    )

    # ABSOLUTE MOVE FROM OPEN

    df["abs_move_from_open"] = np.abs(
        df["close"]
        / df["day_open"]
        - 1
    )

    # TIME-OF-DAY HISTORICAL MOVE

    df["sigma_14"] = (
        df.groupby(
            "minute_of_day"
        )["abs_move_from_open"]
        .transform(
            lambda x:
                x.shift(1)
                 .rolling(
                     window=lookback,
                     min_periods=lookback
                 )
                 .mean()
        )
    )

    # NOISE AREA

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
            + vol_multiplier
            * df["sigma_14"]
        )
    )

    df["lower_band"] = (
        lower_reference
        * (
            1
            - vol_multiplier
            * df["sigma_14"]
        )
    )

    # CUMULATIVE SESSION VWAP

    required_vwap_columns = [
        "vwap",
        "volume"
    ]

    missing = [
        col
        for col in required_vwap_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing VWAP columns: {missing}"
        )

    df["vwap_x_volume"] = (
        df["vwap"]
        * df["volume"]
    )

    df["cum_vwap_volume"] = (
        df.groupby("date")[
            "vwap_x_volume"
        ]
        .cumsum()
    )

    df["cum_volume"] = (
        df.groupby("date")["volume"]
          .cumsum()
    )

    df["session_vwap"] = (
        df["cum_vwap_volume"]
        / df["cum_volume"]
    )

    # Only information available before current minute
    df["session_vwap_prev"] = (
        df.groupby("date")[
            "session_vwap"
        ]
        .shift(1)
    )

    # DECISION TIMES

    df["decision_time"] = (
        df["minute"].isin([0, 30])
        &
        (df["minute_of_day"] >= 600)
        &
        (df["minute_of_day"] <= 930)
    )

    # RAW SIGNAL

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
