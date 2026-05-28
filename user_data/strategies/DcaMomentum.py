from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import talib.abstract as ta

from freqtrade.exchange import timeframe_to_seconds
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.strategy_helper import merge_informative_pair
from pandas import DataFrame


class DcaMomentum(IStrategy):
    """
    DCA Momentum Strategy
    - Sopra EMA200 (Momentum): entra su crossover EMA5/20 con conferma ADX + volume
    - Sotto EMA200 (Accumulo): entra su RSI oversold + candela con ombra inferiore lunga
    - DCA fino a max_tranches con soglie RSI progressive
    - Uscita: prezzo > avg_cost, sopra BB superiore 1h, RSI(1h) > rsi_sell, min_tranches raggiunte
    """

    INTERFACE_VERSION = 3

    # Cooldown tra chiusura di un trade e riapertura sullo stesso pair
    protections = [{"method": "CooldownPeriod", "stop_duration_candles": 3}]

    position_adjustment_enable = True
    max_entry_position_adjustment = 9  # max_tranches - 1

    timeframe = "15m"
    startup_candle_count = 220  # EMA200 + buffer
    process_only_new_candles = True

    # ROI disabilitato: uscita gestita da custom_exit
    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    # ------------------------------------------------------------------ #
    #  Parametri fissi                                                     #
    # ------------------------------------------------------------------ #
    tranche_usd: float = 6.0
    max_tranches: int = 10
    min_tranches: int = 5
    ema_fast_period: int = 5
    ema_slow_period: int = 20
    ema_long_period: int = 200
    rsi_period: int = 14
    adx_period: int = 14
    vol_period: int = 10
    bb_period: int = 20
    bb_mult: float = 2.0
    bb_timeframe: str = "1h"
    cooldown_bars: int = 3

    # ------------------------------------------------------------------ #
    #  Parametri ottimizzabili (hyperopt)                                  #
    # ------------------------------------------------------------------ #
    rsi_t1    = IntParameter(25, 45, default=35, space="buy",  optimize=True)
    rsi_t2    = IntParameter(20, 35, default=30, space="buy",  optimize=True)
    rsi_t3    = IntParameter(15, 28, default=25, space="buy",  optimize=True)
    adx_thresh = IntParameter(15, 35, default=20, space="buy", optimize=True)
    vol_mult  = DecimalParameter(1.0, 2.5, default=1.3, decimals=1, space="buy", optimize=True)
    rsi_sell  = IntParameter(55, 80, default=60, space="sell", optimize=True)

    # ------------------------------------------------------------------ #
    #  Configurazione grafici FreqUI                                      #
    # ------------------------------------------------------------------ #
    @property
    def plot_config(self):
        return {
            "main_plot": {
                "ema_fast":          {"color": "#3498db", "type": "line"},
                "ema_slow":          {"color": "#e67e22", "type": "line"},
                "ema_long":          {"color": "#e74c3c", "type": "line"},
                "bb_upper_1h":       {"color": "#9b59b6", "type": "line"},
                "bb_lower_1h":       {"color": "#9b59b6", "type": "line"},
            },
            "subplots": {
                "RSI": {
                    "rsi":      {"color": "#2ecc71"},
                    "rsi_1h":   {"color": "#8e44ad"},
                },
                "ADX / DI": {
                    "adx":      {"color": "#f39c12"},
                    "plus_di":  {"color": "#27ae60"},
                    "minus_di": {"color": "#c0392b"},
                },
                "Volume": {
                    "volume":   {"color": "#bdc3c7", "type": "bar"},
                    "vol_ma":   {"color": "#e74c3c", "type": "line"},
                },
                "Exit signal": {
                    "exit_condition": {"color": "#e74c3c", "type": "bar"},
                },
            },
        }

    # ------------------------------------------------------------------ #
    #  Informative pairs                                                   #
    # ------------------------------------------------------------------ #
    def informative_pairs(self):
        return [(pair, self.bb_timeframe) for pair in self.dp.current_whitelist()]

    # ------------------------------------------------------------------ #
    #  Indicatori                                                          #
    # ------------------------------------------------------------------ #
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- 5m ---
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_period)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_period)
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=self.ema_long_period)
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=self.rsi_period)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=self.adx_period)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=self.adx_period)
        dataframe["vol_ma"] = dataframe["volume"].rolling(window=self.vol_period).mean()
        dataframe["body"] = (dataframe["close"] - dataframe["open"]).abs()
        dataframe["lower_shadow"] = (
            dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        )

        # --- 1h informative: BB + RSI ---
        inf_df = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.bb_timeframe
        )
        if not inf_df.empty:
            bb_upper, _, bb_lower = ta.BBANDS(
                inf_df["close"],
                timeperiod=self.bb_period,
                nbdevup=self.bb_mult,
                nbdevdn=self.bb_mult,
                matype=0,
            )
            inf_df["bb_upper"] = bb_upper
            inf_df["bb_lower"] = bb_lower
            inf_df["rsi"] = ta.RSI(inf_df["close"], timeperiod=self.rsi_period)

            dataframe = merge_informative_pair(
                dataframe,
                inf_df,
                self.timeframe,
                self.bb_timeframe,
                ffill=True,
            )

        # Colonna exit_condition per visualizzazione in FreqUI
        bb_col = f"bb_upper_{self.bb_timeframe}"
        rsi_col = f"rsi_{self.bb_timeframe}"
        if bb_col in dataframe.columns and rsi_col in dataframe.columns:
            dataframe["exit_condition"] = (
                (dataframe["close"] >= dataframe[bb_col])
                & (dataframe[rsi_col] > self.rsi_sell.value)
            ).astype(int)
        else:
            dataframe["exit_condition"] = 0

        return dataframe

    # ------------------------------------------------------------------ #
    #  Helper: soglia RSI per la tranche N-esima                          #
    # ------------------------------------------------------------------ #
    def _rsi_thresh(self, next_tranche: int) -> int:
        if next_tranche <= 3:
            return self.rsi_t1.value
        if next_tranche <= 7:
            return self.rsi_t2.value
        return self.rsi_t3.value

    # ------------------------------------------------------------------ #
    #  Segnali di ingresso (prima tranche)                                #
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        above = dataframe["close"] > dataframe["ema_long"]

        ema_cross = (dataframe["ema_fast"] > dataframe["ema_slow"]) & (
            dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1)
        )

        momentum = (
            ema_cross
            & (dataframe["adx"] > self.adx_thresh.value)
            & (dataframe["plus_di"] > dataframe["minus_di"])
            & (dataframe["volume"] > dataframe["vol_ma"] * self.vol_mult.value)
        )

        accumulo = (
            (dataframe["rsi"] < self.rsi_t1.value)
            & (dataframe["lower_shadow"] > dataframe["body"] * 1.5)
            & (dataframe["body"] > 0)
        )

        entry = (
            ((above & momentum) | (~above & accumulo)) & (dataframe["volume"] > 0)
        )

        # Cooldown: sopprime il segnale se c'è stato un ingresso nelle ultime cooldown_bars candele
        recent = (
            entry.rolling(window=self.cooldown_bars, min_periods=1).max().shift(1).fillna(0)
        )
        dataframe.loc[entry & (recent == 0), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Uscita gestita interamente da custom_exit
        dataframe["exit_long"] = 0
        return dataframe

    # ------------------------------------------------------------------ #
    #  Logica di uscita                                                   #
    # ------------------------------------------------------------------ #
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        # Minimo tranche richieste
        if trade.nr_of_successful_entries < self.min_tranches:
            return None

        # Esce solo in profitto (include commissioni)
        if current_profit <= 0:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        bb_col = f"bb_upper_{self.bb_timeframe}"
        rsi_col = f"rsi_{self.bb_timeframe}"

        if bb_col not in dataframe.columns or rsi_col not in dataframe.columns:
            return None

        bb_upper = last[bb_col]
        rsi_1h = last[rsi_col]

        if pd.isna(bb_upper) or pd.isna(rsi_1h):
            return None

        if current_rate >= bb_upper and rsi_1h > self.rsi_sell.value:
            return "dca_exit"

        return None

    # ------------------------------------------------------------------ #
    #  DCA: tranche successive                                            #
    # ------------------------------------------------------------------ #
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:

        nr = trade.nr_of_successful_entries

        if nr >= self.max_tranches:
            return None

        # Non aprire nuove tranche se c'è già un ordine in attesa
        if trade.has_open_orders:
            return None

        # Cooldown: minimo cooldown_bars candele dall'ultimo riempimento
        last_filled = trade.date_last_filled_utc
        if last_filled.tzinfo is None:
            last_filled = last_filled.replace(tzinfo=timezone.utc)
        ct = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)

        cooldown_secs = timeframe_to_seconds(self.timeframe) * self.cooldown_bars
        if (ct - last_filled).total_seconds() < cooldown_secs:
            return None

        # Leggi gli indicatori dell'ultima candela chiusa
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty or len(dataframe) < 2:
            return None

        last = dataframe.iloc[-1]
        prev = dataframe.iloc[-2]

        above = bool(last["close"] > last["ema_long"])
        next_tranche = nr + 1
        rsi_thresh = self._rsi_thresh(next_tranche)

        if above:
            signal = (
                bool(last["ema_fast"] > last["ema_slow"])
                and bool(prev["ema_fast"] <= prev["ema_slow"])
                and bool(last["adx"] > self.adx_thresh.value)
                and bool(last["plus_di"] > last["minus_di"])
                and bool(last["volume"] > last["vol_ma"] * self.vol_mult.value)
            )
        else:
            signal = (
                bool(last["rsi"] < rsi_thresh)
                and bool(last["lower_shadow"] > last["body"] * 1.5)
                and bool(last["body"] > 0)
            )

        if not signal:
            return None

        stake = min(self.tranche_usd, max_stake)
        if min_stake and stake < min_stake:
            return None

        return stake
