from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class MACDStrategy(IStrategy):
    """
    MACD crossover.
    Entry: MACD line crosses above signal line (histogram goes positive)
    Exit:  MACD line crosses below signal line (histogram goes negative)
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"60": 0.02, "30": 0.03, "0": 0.05}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    startup_candle_count = 100
    process_only_new_candles = True

    macd_fast = IntParameter(8, 20, default=12, space="buy", optimize=True)
    macd_slow = IntParameter(20, 40, default=26, space="buy", optimize=True)
    macd_signal = IntParameter(5, 15, default=9, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd, signal, hist = ta.MACD(
            dataframe["close"],
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value,
        )
        dataframe["macd"] = macd
        dataframe["signal"] = signal
        dataframe["hist"] = hist
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_above(dataframe["macd"], dataframe["signal"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_below(dataframe["macd"], dataframe["signal"])
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
