from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class EMAcrossSpot(IStrategy):
    """
    EMA crossover spot — versione filtrata per mercati difficili.

    Entry: EMA fast incrocia sopra EMA slow
           + close sopra EMA slow (prezzo realmente in uptrend)
           + ADX sopra soglia (trend direzionale, non laterale)
           + volume sopra media
           + RSI non in ipercomprato

    Exit: ROI fisso + stoploss — nessun segnale inverso.
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count = 210
    process_only_new_candles = True

    minimal_roi = {"0": 0.005}

    stoploss = -0.02
    trailing_stop = False
    use_custom_stoploss = False

    # --- Parametri hyperopt ---

    ema_fast = IntParameter(5,  50,  default=20,  space="buy", optimize=True)
    ema_slow = IntParameter(20, 200, default=200, space="buy", optimize=True)

    adx_period    = IntParameter(7,  21, default=14, space="buy", optimize=True)
    adx_threshold = IntParameter(15, 40, default=25, space="buy", optimize=True)

    volume_ma_period  = IntParameter(10, 50,  default=20,  space="buy", optimize=True)
    volume_multiplier = DecimalParameter(0.5, 2.0, default=1.0, decimals=1, space="buy", optimize=True)

    rsi_period   = IntParameter(7,  21, default=14, space="buy", optimize=True)
    rsi_long_max = IntParameter(45, 75, default=65, space="buy", optimize=True)

    # --- Indicatori ---

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val in self.ema_fast.range:
            dataframe[f"ema_fast_{val}"] = ta.EMA(dataframe, timeperiod=val)
        for val in self.ema_slow.range:
            dataframe[f"ema_slow_{val}"] = ta.EMA(dataframe, timeperiod=val)

        for val in self.adx_period.range:
            dataframe[f"adx_{val}"] = ta.ADX(dataframe, timeperiod=val)

        for val in self.volume_ma_period.range:
            dataframe[f"volume_ma_{val}"] = ta.SMA(dataframe["volume"], timeperiod=val)

        for val in self.rsi_period.range:
            dataframe[f"rsi_{val}"] = ta.RSI(dataframe, timeperiod=val)

        return dataframe

    # --- Segnali di entrata ---

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast   = f"ema_fast_{self.ema_fast.value}"
        slow   = f"ema_slow_{self.ema_slow.value}"
        adx    = f"adx_{self.adx_period.value}"
        vol_ma = f"volume_ma_{self.volume_ma_period.value}"
        rsi    = f"rsi_{self.rsi_period.value}"

        dataframe.loc[
            qtpylib.crossed_above(dataframe[fast], dataframe[slow])
            & (dataframe["close"] > dataframe[slow])
            & (dataframe[adx] > self.adx_threshold.value)
            & (dataframe["volume"] > dataframe[vol_ma] * self.volume_multiplier.value)
            & (dataframe[rsi] < self.rsi_long_max.value)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    # --- Segnali di uscita ---

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
