from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, informative
from pandas import DataFrame
import talib.abstract as ta
from technical import qtpylib


class btc_dca(IStrategy):
    """
    BTC DCA Strategy - Entry only (exit via minimal_roi placeholder).

    Entry conditions (all required):
      - RSI 4h < rsi_buy_threshold       (oversold on 4h)
      - RSI 1d < rsi_daily_threshold     (weakness confirmed on daily)
      - Close < BB lower band 4h         (price outside lower Bollinger Band)

    Cooldown: 42 candles (7 days on 4h timeframe) between entries.

    All entry parameters are hyperopt-optimizable.
    Exit is intentionally disabled (minimal_roi = 500%) for entry-only backtesting.
    """

    INTERFACE_VERSION = 3

    timeframe = "4h"

    can_short = False

    # Placeholder exit — never triggers during entry-only backtesting
    minimal_roi = {"0": 5.0}

    # Very loose stoploss — not the focus of this phase
    stoploss = -0.99

    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Enough candles for BB and RSI to warm up
    startup_candle_count: int = 200

    # Cooldown: 42 candles = 7 days on 4h — fixed, not hyperopt-optimizable
    # (freqtrade protections are evaluated once at startup, not per hyperopt iteration)
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 42,
            }
        ]

    # --- Hyperopt parameters ---

    # RSI 4h threshold: bot enters when RSI drops below this value
    rsi_buy_threshold = IntParameter(
        low=15, high=40, default=30, space="buy", optimize=True, load=True
    )

    # RSI daily threshold: filters out entries during strong downtrends
    rsi_daily_threshold = IntParameter(
        low=30, high=60, default=45, space="buy", optimize=True, load=True
    )

    # Bollinger Band window (number of candles)
    bb_window = IntParameter(
        low=10, high=50, default=20, space="buy", optimize=True, load=True
    )

    # Bollinger Band standard deviations
    bb_std = DecimalParameter(
        low=1.5, high=3.0, default=2.0, decimals=1, space="buy", optimize=True, load=True
    )

    # --- Informative timeframe: daily RSI ---

    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    # --- Indicators on 4h ---

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI on 4h
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger Bands on 4h (window and std are hyperopt-optimizable)
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=self.bb_window.value,
            stds=self.bb_std.value,
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        return dataframe

    # --- Entry signal ---

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < self.rsi_buy_threshold.value)       # oversold 4h
                & (dataframe["rsi_1d"] < self.rsi_daily_threshold.value) # weakness on daily
                & (dataframe["close"] < dataframe["bb_lowerband"])       # below BB lower
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    # --- Exit signal (disabled) ---

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # No exit signal — exits handled by minimal_roi placeholder only
        return dataframe
