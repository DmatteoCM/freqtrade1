from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class EMAcross(IStrategy):
    """
    EMA fast/slow crossover.
    Entry: fast EMA crosses above slow EMA
    Exit:  fast EMA crosses below slow EMA
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"60": 0.02, "30": 0.03, "0": 0.05}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    startup_candle_count = 100
    process_only_new_candles = True

    ema_fast = IntParameter(5, 30, default=10, space="buy", optimize=True)
    ema_slow = IntParameter(30, 100, default=50, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val in self.ema_fast.range:
            dataframe[f"ema_fast_{val}"] = ta.EMA(dataframe, timeperiod=val)
        for val in self.ema_slow.range:
            dataframe[f"ema_slow_{val}"] = ta.EMA(dataframe, timeperiod=val)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = f"ema_fast_{self.ema_fast.value}"
        slow = f"ema_slow_{self.ema_slow.value}"
        dataframe.loc[
            qtpylib.crossed_above(dataframe[fast], dataframe[slow])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = f"ema_fast_{self.ema_fast.value}"
        slow = f"ema_slow_{self.ema_slow.value}"
        dataframe.loc[
            qtpylib.crossed_below(dataframe[fast], dataframe[slow])
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
