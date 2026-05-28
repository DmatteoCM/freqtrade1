from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, stoploss_from_open
from pandas import DataFrame
from functools import reduce
import talib.abstract as ta
from technical import qtpylib


class ema_scalping(IStrategy):
    """
    EMA Scalping Strategy — Long & Short on 5m timeframe.

    Entry conditions (all required):
      Long:  EMA8 crosses above EMA13 + volume > volume MA + RSI < rsi_long_max
      Short: EMA8 crosses below EMA13 + volume > volume MA + RSI > rsi_short_min

    Exit: trailing stop (optimized via hyperopt)
    Stake: 10 USDC per trade, 2x isolated leverage
    Pairs: BTC/USDC, ETH/USDC, SOL/USDC (USDC-M Futures)

    All entry/exit parameters are hyperopt-optimizable.
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"

    can_short = True

    # No fixed ROI — exits handled by trailing stop only
    minimal_roi = {"0": 100}

    trailing_stop = False
    use_custom_stoploss = True

    stoploss = -0.05

    process_only_new_candles = True
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 50

    # --- Hyperopt parameters ---

    # EMA periods
    ema_fast = IntParameter(low=5, high=15, default=8, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(low=10, high=30, default=13, space="buy", optimize=True, load=True)

    # Volume filter: entry only when volume > volume_ma * this multiplier
    volume_ma_period = IntParameter(low=10, high=50, default=20, space="buy", optimize=True, load=True)
    volume_multiplier = DecimalParameter(low=0.1, high=2.0, default=1.0, decimals=1, space="buy", optimize=True, load=True)

    # RSI filter
    rsi_period = IntParameter(low=7, high=21, default=14, space="buy", optimize=True, load=True)
    rsi_long_max = IntParameter(low=45, high=70, default=60, space="buy", optimize=True, load=True)
    rsi_short_min = IntParameter(low=30, high=55, default=40, space="sell", optimize=True, load=True)

    # Trailing stop parameters
    trailing_stop_positive_opt = DecimalParameter(low=0.005, high=0.03, default=0.01, decimals=3, space="sell", optimize=True, load=True)
    trailing_stop_positive_offset_opt = DecimalParameter(low=0.01, high=0.05, default=0.02, decimals=3, space="sell", optimize=True, load=True)

    # Leverage
    leverage_value = 2.0

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side: str) -> float:
        return self.leverage_value

    def custom_stoploss(self, pair: str, trade, current_time, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> float:
        offset = self.trailing_stop_positive_offset_opt.value
        trail = self.trailing_stop_positive_opt.value

        if current_profit >= offset:
            return stoploss_from_open(trail, current_profit, is_short=trade.is_short)

        return self.stoploss

    # --- Indicators ---

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA fast and slow
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)

        # Volume moving average
        dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=self.volume_ma_period.value)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # EMA crossovers
        dataframe["ema_cross_up"] = qtpylib.crossed_above(dataframe["ema_fast"], dataframe["ema_slow"])
        dataframe["ema_cross_down"] = qtpylib.crossed_above(dataframe["ema_slow"], dataframe["ema_fast"])

        return dataframe

    # --- Entry signals ---

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cross_up = dataframe["ema_cross_up"].sum()
        vol_ok = (dataframe["volume"] > dataframe["volume_ma"] * self.volume_multiplier.value).sum()
        rsi_ok = (dataframe["rsi"] < self.rsi_long_max.value).sum()
        # Long: EMA cross up + volume filter + RSI not overbought
        dataframe.loc[
            (
                dataframe["ema_cross_up"]
                & (dataframe["volume"] > dataframe["volume_ma"] * self.volume_multiplier.value)
                & (dataframe["rsi"] < self.rsi_long_max.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # Short: EMA cross down + volume filter + RSI not oversold
        dataframe.loc[
            (
                dataframe["ema_cross_down"]
                & (dataframe["volume"] > dataframe["volume_ma"] * self.volume_multiplier.value)
                & (dataframe["rsi"] > self.rsi_short_min.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    # --- Exit signals (disabled — trailing stop handles exits) ---

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe