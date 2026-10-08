
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import (
    CLEAN_FILE,
    RESULTS_DIR,
    INITIAL_CAPITAL,
    COMMISSION_PER_SHARE,
    SLIPPAGE_PER_SHARE,
    TARGET_DAILY_VOL,
    MAX_LEVERAGE,
    TRAIN_END,
)

from src.features import (
    build_features
)

from src.robustness import (
    add_dynamic_leverage
)

from src.extensions import (
    add_confirmation_signal
)

from src.backtest import (
    run_momentum_backtest
)

from src.metrics import (
    performance_metrics,
    trade_metrics
)


FIGURES_DIR = Path("output/figures")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def yearly_returns(daily):
    """
    Compound daily strategy returns by calendar year.
    """

    x = daily.copy()

    x["date"] = pd.to_datetime(
        x["date"]
    )

    x["year"] = (
        x["date"]
        .dt.year
    )

    return (
        x.groupby("year")["return"]
        .apply(
            lambda r:
                (1 + r).prod() - 1
        )
    )


def build_drawdown(daily):
    """
    Construct cumulative growth and drawdown series.
    """

    x = daily.copy()

    x["date"] = pd.to_datetime(
        x["date"]
    )

    x["growth"] = (
        1
        + x["return"]
    ).cumprod()

    x["running_max"] = (
        x["growth"]
        .cummax()
    )

    x["drawdown"] = (
        x["growth"]
        / x["running_max"]
        - 1
    )

    return x


def main():

    # 1. LOAD CLEAN DATA

    print("Loading clean SPY data...")

    raw = pd.read_csv(
        CLEAN_FILE
    )

    # 2. PAPER STRATEGY FEATURES

    print("Building Paper Dynamic features...")

    paper_features = build_features(
        raw,
        lookback=14,
        vol_multiplier=1.0
    )

    paper_features = add_dynamic_leverage(
        paper_features,
        target_vol=TARGET_DAILY_VOL,
        max_leverage=MAX_LEVERAGE
    )

    # 3. ORIGINAL STRATEGY V2

    print("Building Confirmed Breakout v2 features...")

    confirmed_features = (
        add_confirmation_signal(
            paper_features
        )
    )

    # 4. TRAIN / TEST SPLIT

    paper_train = (
        paper_features.loc[
            paper_features["date"]
            <= TRAIN_END
        ]
        .copy()
    )

    paper_test = (
        paper_features.loc[
            paper_features["date"]
            > TRAIN_END
        ]
        .copy()
    )

    confirmed_train = (
        confirmed_features.loc[
            confirmed_features["date"]
            <= TRAIN_END
        ]
        .copy()
    )

    confirmed_test = (
        confirmed_features.loc[
            confirmed_features["date"]
            > TRAIN_END
        ]
        .copy()
    )

    # 5. PAPER DYNAMIC BACKTESTS

    print("Running Paper Dynamic train...")

    paper_train_daily, paper_train_trades = (
        run_momentum_backtest(
            paper_train,
            initial_capital=INITIAL_CAPITAL,
            commission_per_share=COMMISSION_PER_SHARE,
            slippage_per_share=SLIPPAGE_PER_SHARE,
            stop_mode="band_vwap",
            sizing_mode="dynamic",
            entry_signal_col=None,
        )
    )

    print("Running Paper Dynamic test...")

    paper_test_daily, paper_test_trades = (
        run_momentum_backtest(
            paper_test,
            initial_capital=INITIAL_CAPITAL,
            commission_per_share=COMMISSION_PER_SHARE,
            slippage_per_share=SLIPPAGE_PER_SHARE,
            stop_mode="band_vwap",
            sizing_mode="dynamic",
            entry_signal_col=None,
        )
    )

    # 6. CONFIRMED BREAKOUT V2 BACKTESTS

    print("Running Confirmed Breakout v2 train...")

    confirmed_train_daily, confirmed_train_trades = (
        run_momentum_backtest(
            confirmed_train,
            initial_capital=INITIAL_CAPITAL,
            commission_per_share=COMMISSION_PER_SHARE,
            slippage_per_share=SLIPPAGE_PER_SHARE,
            stop_mode="band_vwap",
            sizing_mode="dynamic",
            entry_signal_col="confirmed_signal",
        )
    )

    print("Running Confirmed Breakout v2 test...")

    confirmed_test_daily, confirmed_test_trades = (
        run_momentum_backtest(
            confirmed_test,
            initial_capital=INITIAL_CAPITAL,
            commission_per_share=COMMISSION_PER_SHARE,
            slippage_per_share=SLIPPAGE_PER_SHARE,
            stop_mode="band_vwap",
            sizing_mode="dynamic",
            entry_signal_col="confirmed_signal",
        )
    )

    # 7. SPY BUY & HOLD

    spy_daily = (
        paper_features
        .groupby("date")
        .agg(
            close=("close", "last")
        )
        .reset_index()
    )

    spy_daily["return"] = (
        spy_daily["close"]
        .pct_change()
    )

    spy_train = (
        spy_daily.loc[
            spy_daily["date"]
            <= TRAIN_END
        ]
        .dropna()
        .copy()
    )

    spy_test = (
        spy_daily.loc[
            spy_daily["date"]
            > TRAIN_END
        ]
        .dropna()
        .copy()
    )

    # 8. FINAL TRAIN / TEST PERFORMANCE TABLE

    rows = []

    strategies = {
        ("Paper Dynamic", "Train 2018-2022"):
            (
                paper_train_daily,
                paper_train_trades
            ),

        ("Confirmed Breakout v2", "Train 2018-2022"):
            (
                confirmed_train_daily,
                confirmed_train_trades
            ),

        ("SPY Buy & Hold", "Train 2018-2022"):
            (
                spy_train,
                None
            ),

        ("Paper Dynamic", "Test 2023-Apr 2024"):
            (
                paper_test_daily,
                paper_test_trades
            ),

        ("Confirmed Breakout v2", "Test 2023-Apr 2024"):
            (
                confirmed_test_daily,
                confirmed_test_trades
            ),

        ("SPY Buy & Hold", "Test 2023-Apr 2024"):
            (
                spy_test,
                None
            ),
    }

    for (
        strategy,
        sample
    ), (
        daily,
        trades
    ) in strategies.items():

        metrics = performance_metrics(
            daily["return"]
        )

        rows.append({
            "Strategy":
                strategy,

            "Sample":
                sample,

            **metrics,

            "Trades":
                (
                    len(trades)
                    if trades is not None
                    else np.nan
                )
        })

    final_comparison = (
        pd.DataFrame(rows)
    )

    final_comparison.to_csv(
        RESULTS_DIR
        / "final_train_test_comparison.csv",
        index=False
    )

    # 9. TRADE-LEVEL TEST COMPARISON

    trade_comparison = pd.DataFrame({
        "Paper Dynamic":
            trade_metrics(
                paper_test_trades
            ),

        "Confirmed Breakout v2":
            trade_metrics(
                confirmed_test_trades
            ),
    }).T

    trade_comparison.to_csv(
        RESULTS_DIR
        / "final_oos_trade_comparison.csv"
    )

    # 10. SAVE FINAL DAILY / TRADE DATA

    paper_train_daily.to_csv(
        RESULTS_DIR
        / "paper_dynamic_train_daily.csv",
        index=False
    )

    paper_train_trades.to_csv(
        RESULTS_DIR
        / "paper_dynamic_train_trades.csv",
        index=False
    )

    paper_test_daily.to_csv(
        RESULTS_DIR
        / "paper_dynamic_oos_daily.csv",
        index=False
    )

    paper_test_trades.to_csv(
        RESULTS_DIR
        / "paper_dynamic_oos_trades.csv",
        index=False
    )

    confirmed_train_daily.to_csv(
        RESULTS_DIR
        / "confirmed_breakout_v2_train_daily.csv",
        index=False
    )

    confirmed_train_trades.to_csv(
        RESULTS_DIR
        / "confirmed_breakout_v2_train_trades.csv",
        index=False
    )

    confirmed_test_daily.to_csv(
        RESULTS_DIR
        / "confirmed_breakout_v2_oos_daily.csv",
        index=False
    )

    confirmed_test_trades.to_csv(
        RESULTS_DIR
        / "confirmed_breakout_v2_oos_trades.csv",
        index=False
    )

    # 11. YEARLY RETURNS

    yearly = pd.DataFrame({
        "Paper Dynamic":
            yearly_returns(
                pd.concat(
                    [
                        paper_train_daily,
                        paper_test_daily
                    ],
                    ignore_index=True
                )
            ),

        "Confirmed Breakout v2":
            yearly_returns(
                pd.concat(
                    [
                        confirmed_train_daily,
                        confirmed_test_daily
                    ],
                    ignore_index=True
                )
            ),

        "SPY Buy & Hold":
            yearly_returns(
                spy_daily.dropna()
            ),
    })

    yearly.to_csv(
        RESULTS_DIR
        / "yearly_strategy_comparison.csv"
    )

    # 12. OOS EQUITY CURVE

    paper_test_plot = (
        paper_test_daily.copy()
    )

    confirmed_test_plot = (
        confirmed_test_daily.copy()
    )

    spy_test_plot = (
        spy_test.copy()
    )

    paper_test_plot["date"] = pd.to_datetime(
        paper_test_plot["date"]
    )

    confirmed_test_plot["date"] = pd.to_datetime(
        confirmed_test_plot["date"]
    )

    spy_test_plot["date"] = pd.to_datetime(
        spy_test_plot["date"]
    )

    paper_test_plot["growth"] = (
        paper_test_plot["end_equity"]
        / INITIAL_CAPITAL
    )

    confirmed_test_plot["growth"] = (
        confirmed_test_plot["end_equity"]
        / INITIAL_CAPITAL
    )

    spy_test_plot["growth"] = (
        1
        + spy_test_plot["return"]
    ).cumprod()

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        paper_test_plot["date"],
        paper_test_plot["growth"],
        label="Paper Dynamic"
    )

    ax.plot(
        confirmed_test_plot["date"],
        confirmed_test_plot["growth"],
        label="Confirmed Breakout v2"
    )

    ax.plot(
        spy_test_plot["date"],
        spy_test_plot["growth"],
        label="SPY Buy & Hold"
    )

    ax.set_title(
        "Out-of-Sample Performance — 2023 to April 2024"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(
            interval=2
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%b %Y"
        )
    )

    ax.legend()
    ax.grid(alpha=0.2)

    fig.autofmt_xdate(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_oos_equity_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 13. OOS DRAWDOWN FIGURE

    paper_dd = build_drawdown(
        paper_test_daily
    )

    confirmed_dd = build_drawdown(
        confirmed_test_daily
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        paper_dd["date"],
        paper_dd["drawdown"],
        label="Paper Dynamic"
    )

    ax.plot(
        confirmed_dd["date"],
        confirmed_dd["drawdown"],
        label="Confirmed Breakout v2"
    )

    ax.set_title(
        "Out-of-Sample Drawdowns — 2023 to April 2024"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")

    ax.legend()
    ax.grid(alpha=0.2)

    fig.autofmt_xdate(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_oos_drawdowns.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # COMPLETE

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(
        "\nData range:",
        paper_features["date"].min(),
        "to",
        paper_features["date"].max()
    )

    print(
        "\nTrain:",
        paper_train["date"].min(),
        "to",
        paper_train["date"].max()
    )

    print(
        "\nTest:",
        paper_test["date"].min(),
        "to",
        paper_test["date"].max()
    )

    print(
        "\nFinal comparison saved to:"
    )

    print(
        RESULTS_DIR
        / "final_train_test_comparison.csv"
    )


if __name__ == "__main__":
    main()
