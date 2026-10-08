
import numpy as np
import pandas as pd


def run_momentum_backtest(
    data,
    initial_capital=100_000,
    commission_per_share=0.0035,
    slippage_per_share=0.001,
    stop_mode="opposite_band",
    sizing_mode="fixed",
    entry_signal_col=None,
):
    """
    Backtest the SPY intraday momentum strategy.

    Parameters
    ----------
    data : pd.DataFrame
        Feature-engineered intraday SPY data.

    initial_capital : float
        Starting portfolio value.

    commission_per_share : float
        Commission charged per share per transaction.

    slippage_per_share : float
        Adverse execution slippage per share.

    stop_mode : str
        "opposite_band"
        "band_vwap"

    sizing_mode : str
        "fixed"
        "dynamic"

    entry_signal_col : str or None
        None:
            Use original paper breakout conditions.

        Example "confirmed_signal":
            Only enter when that column equals +1 or -1.
    """

    # VALIDATION

    if entry_signal_col is not None:

        if entry_signal_col not in data.columns:

            raise ValueError(
                f"Entry signal column "
                f"'{entry_signal_col}' not found."
            )

    # INITIAL PORTFOLIO

    equity = initial_capital

    daily_results = []
    trades = []

    # LOOP THROUGH TRADING DAYS

    for date, day in data.groupby(
        "date",
        sort=True
    ):

        day = (
            day.sort_values("datetime")
               .copy()
        )

        # Skip initial Noise Area warm-up days
        if day["upper_band"].isna().all():
            continue

        start_equity = equity

        day_open = (
            day["open"].iloc[0]
        )

        # POSITION SIZING

        if sizing_mode == "fixed":

            leverage = 1.0

        elif sizing_mode == "dynamic":

            if "dynamic_leverage" not in day.columns:

                raise ValueError(
                    "dynamic_leverage column missing."
                )

            leverage = (
                day["dynamic_leverage"]
                .iloc[0]
            )

            if pd.isna(leverage):
                continue

        else:

            raise ValueError(
                f"Unknown sizing_mode: {sizing_mode}"
            )

        shares = int(
            np.floor(
                start_equity
                * leverage
                / day_open
            )
        )

        if shares <= 0:
            continue

        # INITIAL POSITION STATE

        position = 0

        entry_price = None
        entry_time = None

        day_pnl = 0.0

        decision_rows = (
            day.loc[
                day["decision_time"]
            ]
        )

        # TRADE RECORDING HELPER

        def record_trade(
            side,
            exit_price,
            exit_time,
            exit_reason
        ):

            if side == "LONG":

                gross_pnl = (
                    shares
                    * (
                        exit_price
                        - entry_price
                    )
                )

            elif side == "SHORT":

                gross_pnl = (
                    shares
                    * (
                        entry_price
                        - exit_price
                    )
                )

            else:

                raise ValueError(
                    f"Unknown side: {side}"
                )

            commission = (
                2
                * shares
                * commission_per_share
            )

            net_pnl = (
                gross_pnl
                - commission
            )

            trades.append({
                "date": date,
                "side": side,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "leverage": leverage,
                "gross_pnl": gross_pnl,
                "commission": commission,
                "net_pnl": net_pnl,
                "exit_reason": exit_reason,
            })

            return net_pnl

        # INTRADAY DECISION LOOP

        for _, row in decision_rows.iterrows():

            if pd.isna(
                row["upper_band"]
            ):
                continue

            price = row["open"]

            upper = row["upper_band"]
            lower = row["lower_band"]

            vwap = (
                row["session_vwap_prev"]
            )

            # ENTRY CONDITIONS

            if entry_signal_col is None:

                long_entry = (
                    price > upper
                )

                short_entry = (
                    price < lower
                )

            else:

                signal = (
                    row[entry_signal_col]
                )

                long_entry = (
                    signal == 1
                )

                short_entry = (
                    signal == -1
                )

            # FLAT

            if position == 0:

                if long_entry:

                    entry_price = (
                        price
                        + slippage_per_share
                    )

                    entry_time = (
                        row["datetime"]
                    )

                    position = 1

                elif short_entry:

                    entry_price = (
                        price
                        - slippage_per_share
                    )

                    entry_time = (
                        row["datetime"]
                    )

                    position = -1

            # LONG POSITION

            elif position == 1:

                if stop_mode == "opposite_band":

                    stop = lower

                elif stop_mode == "band_vwap":

                    if pd.isna(vwap):
                        stop = upper

                    else:
                        stop = max(
                            upper,
                            vwap
                        )

                else:

                    raise ValueError(
                        f"Unknown stop_mode: "
                        f"{stop_mode}"
                    )

                # LONG EXIT

                if price < stop:

                    exit_price = (
                        price
                        - slippage_per_share
                    )

                    day_pnl += record_trade(
                        side="LONG",
                        exit_price=exit_price,
                        exit_time=row["datetime"],
                        exit_reason=stop_mode
                    )

                    position = 0

                    entry_price = None
                    entry_time = None

                    # OPTIONAL SHORT REVERSAL

                    if short_entry:

                        entry_price = (
                            price
                            - slippage_per_share
                        )

                        entry_time = (
                            row["datetime"]
                        )

                        position = -1

            # SHORT POSITION

            elif position == -1:

                if stop_mode == "opposite_band":

                    stop = upper

                elif stop_mode == "band_vwap":

                    if pd.isna(vwap):
                        stop = lower

                    else:
                        stop = min(
                            lower,
                            vwap
                        )

                else:

                    raise ValueError(
                        f"Unknown stop_mode: "
                        f"{stop_mode}"
                    )

                # SHORT EXIT

                if price > stop:

                    exit_price = (
                        price
                        + slippage_per_share
                    )

                    day_pnl += record_trade(
                        side="SHORT",
                        exit_price=exit_price,
                        exit_time=row["datetime"],
                        exit_reason=stop_mode
                    )

                    position = 0

                    entry_price = None
                    entry_time = None

                    # OPTIONAL LONG REVERSAL

                    if long_entry:

                        entry_price = (
                            price
                            + slippage_per_share
                        )

                        entry_time = (
                            row["datetime"]
                        )

                        position = 1

        # END-OF-DAY EXIT

        close_price = (
            day["close"].iloc[-1]
        )

        close_time = (
            day["datetime"].iloc[-1]
        )

        if position == 1:

            exit_price = (
                close_price
                - slippage_per_share
            )

            day_pnl += record_trade(
                side="LONG",
                exit_price=exit_price,
                exit_time=close_time,
                exit_reason="market_close"
            )

        elif position == -1:

            exit_price = (
                close_price
                + slippage_per_share
            )

            day_pnl += record_trade(
                side="SHORT",
                exit_price=exit_price,
                exit_time=close_time,
                exit_reason="market_close"
            )

        # DAILY PORTFOLIO UPDATE

        equity = (
            start_equity
            + day_pnl
        )

        daily_return = (
            equity
            / start_equity
            - 1
        )

        daily_results.append({
            "date": date,
            "start_equity": start_equity,
            "end_equity": equity,
            "pnl": day_pnl,
            "return": daily_return,
            "shares": shares,
            "leverage": leverage,
        })

    return (
        pd.DataFrame(daily_results),
        pd.DataFrame(trades)
    )


def run_baseline_backtest(
    data,
    initial_capital=100_000,
    commission_per_share=0.0035,
    slippage_per_share=0.001,
):
    """
    Convenience wrapper for the original fixed-size,
    opposite-band strategy.
    """

    return run_momentum_backtest(
        data=data,
        initial_capital=initial_capital,
        commission_per_share=commission_per_share,
        slippage_per_share=slippage_per_share,
        stop_mode="opposite_band",
        sizing_mode="fixed",
        entry_signal_col=None,
    )
