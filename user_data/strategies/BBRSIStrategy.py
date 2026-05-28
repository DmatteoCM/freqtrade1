from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class BBRSIStrategy(IStrategy):
    """
    Bollinger Bands + RSI.
    Entry: price closes below lower BB AND RSI < rsi_buy threshold
    Exit:  RSI > rsi_sell threshold OR price closes above upper BB
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"40": 0.01, "20": 0.02, "0": 0.04}
    stoploss = -0.04
    trailing_stop = False

    timeframe = "5m"
    startup_candle_count = 50
    process_only_new_candles = True

    rsi_buy = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell = IntParameter(60, 80, default=70, space="sell", optimize=True)
    bb_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=self.bb_period.value,
            stds=self.bb_std.value,
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lower"]) / (
            dataframe["bb_upper"] - dataframe["bb_lower"]
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi"] < self.rsi_buy.value)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["rsi"] > self.rsi_sell.value)
            | (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
