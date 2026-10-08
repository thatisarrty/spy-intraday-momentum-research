
import numpy as np
import pandas as pd


def performance_metrics(returns):
    """
    Calculate portfolio-level performance metrics
    from a series of daily returns.

    Notes
    -----
    Sharpe ratio assumes a zero risk-free rate.

    Positive Day Ratio is the percentage of daily
    observations with a return greater than zero.
    It is NOT the trade win rate.
    """

    returns = (
        pd.Series(returns)
        .dropna()
        .astype(float)
    )

    n = len(returns)

    if n == 0:
        return {
            "Observations": 0,
            "Total Return": np.nan,
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan,
            "Positive Day Ratio": np.nan,
        }

    growth = (
        1
        + returns
    ).cumprod()

    total_return = (
        growth.iloc[-1]
        - 1
    )

    annualized_return = (
        (1 + total_return)
        ** (252 / n)
        - 1
    )

    annualized_volatility = (
        returns.std(ddof=1)
        * np.sqrt(252)
    )

    if returns.std(ddof=1) > 0:

        sharpe = (
            returns.mean()
            / returns.std(ddof=1)
            * np.sqrt(252)
        )

    else:

        sharpe = np.nan

    running_max = (
        growth.cummax()
    )

    drawdown = (
        growth
        / running_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    positive_day_ratio = (
        returns > 0
    ).mean()

    return {
        "Observations":
            n,

        "Total Return":
            total_return,

        "Annualized Return":
            annualized_return,

        "Annualized Volatility":
            annualized_volatility,

        "Sharpe Ratio":
            sharpe,

        "Max Drawdown":
            max_drawdown,

        "Positive Day Ratio":
            positive_day_ratio,
    }


def trade_metrics(trades):
    """
    Calculate trade-level statistics.
    """

    if trades is None or len(trades) == 0:

        return {
            "Trades": 0,
            "Trade Win Rate": np.nan,
            "Average Trade PnL": np.nan,
            "Median Trade PnL": np.nan,
            "Best Trade": np.nan,
            "Worst Trade": np.nan,
            "Profit Factor": np.nan,
        }

    pnl = (
        trades["net_pnl"]
        .dropna()
    )

    winners = (
        pnl[pnl > 0]
    )

    losers = (
        pnl[pnl < 0]
    )

    gross_profit = (
        winners.sum()
    )

    gross_loss = (
        abs(
            losers.sum()
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.nan
    )

    return {
        "Trades":
            len(pnl),

        "Trade Win Rate":
            (pnl > 0).mean(),

        "Average Trade PnL":
            pnl.mean(),

        "Median Trade PnL":
            pnl.median(),

        "Best Trade":
            pnl.max(),

        "Worst Trade":
            pnl.min(),

        "Profit Factor":
            profit_factor,
    }
