from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class EMA20_200(IStrategy):
    """
    EMA 20 / EMA 200 crossover — 5m spot
    Entry: EMA20 crosses above EMA200
    Exit:  EMA20 crosses below EMA200 or ROI hit
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count = 210

    # 1% di profitto in qualsiasi momento
    minimal_roi = {"0": 0.01}

    # stoploss fisso al 2%
    stoploss = -0.02

    trailing_stop = False
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_above(dataframe["ema20"], dataframe["ema200"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_below(dataframe["ema20"], dataframe["ema200"])
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
