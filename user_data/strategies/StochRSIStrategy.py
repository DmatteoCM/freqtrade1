from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class StochRSIStrategy(IStrategy):
    """
    Stochastic RSI mean-reversion.
    Entry: StochRSI K crosses above D while both < 20 (oversold)
    Exit:  StochRSI K crosses below D while both > 80 (overbought)
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"30": 0.01, "15": 0.02, "0": 0.03}
    stoploss = -0.03
    trailing_stop = False

    timeframe = "5m"
    startup_candle_count = 100
    process_only_new_candles = True

    stochrsi_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    stoch_k = IntParameter(3, 10, default=3, space="buy", optimize=True)
    stoch_d = IntParameter(3, 10, default=3, space="buy", optimize=True)
    oversold = IntParameter(10, 30, default=20, space="buy", optimize=True)
    overbought = IntParameter(70, 90, default=80, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fastk, fastd = ta.STOCHRSI(
            dataframe["close"],
            timeperiod=self.stochrsi_period.value,
            fastk_period=self.stoch_k.value,
            fastd_period=self.stoch_d.value,
            fastd_matype=0,
        )
        dataframe["stochrsi_k"] = fastk
        dataframe["stochrsi_d"] = fastd
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_above(dataframe["stochrsi_k"], dataframe["stochrsi_d"])
            & (dataframe["stochrsi_k"] < self.oversold.value)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_below(dataframe["stochrsi_k"], dataframe["stochrsi_d"])
            & (dataframe["stochrsi_k"] > self.overbought.value)
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
