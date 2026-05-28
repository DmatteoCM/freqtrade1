from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import talib.abstract as ta

from freqtrade.exchange import timeframe_to_seconds
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.strategy_helper import merge_informative_pair
from pandas import DataFrame


class DcaAccumulo(IStrategy):
    """
    DCA Accumulo Strategy — derivata da DcaMomentum
    - Entra solo in regime Accumulo (sotto EMA200): RSI oversold + candela martello
    - DCA fino a max_tranches con soglie RSI progressive
    - Uscita: profitto > 0, sopra BB superiore 1h, RSI(1h) > rsi_sell
      (nessun minimo di tranche richieste)

    RISULTATI BACKTEST (DOGE/USDC, 90gg Feb-Mag 2026):
    - Profitto totale: +8.13 USDC (+0.81%) vs +15.61 USDC di DcaMomentum
    - Win rate altissimo (96.6%) ma profitto assoluto dimezzato
    - Rimuovere il regime Momentum (sopra EMA200) ha tolto le entrate
      sui trend rialzisti, che erano le più redditizie nel periodo testato
    - Potenzialmente valida in mercati ribassisti, inferiore in trend bullish
    """

    INTERFACE_VERSION = 3

    protections = [{"method": "CooldownPeriod", "stop_duration_candles": 3}]

    position_adjustment_enable = True
    max_entry_position_adjustment = 9  # max_tranches - 1

    timeframe = "15m"
    startup_candle_count = 220
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    # ------------------------------------------------------------------ #
    #  Parametri fissi                                                     #
    # ------------------------------------------------------------------ #
    tranche_usd: float = 6.0
    max_tranches: int = 10
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
    rsi_sell  = IntParameter(55, 80, default=60, space="sell", optimize=True)

    # ------------------------------------------------------------------ #
    #  Configurazione grafici FreqUI                                      #
    # ------------------------------------------------------------------ #
    @property
    def plot_config(self):
        return {
            "main_plot": {
                "ema_long":    {"color": "#e74c3c", "type": "line"},
                "bb_upper_1h": {"color": "#9b59b6", "type": "line"},
                "bb_lower_1h": {"color": "#9b59b6", "type": "line"},
            },
            "subplots": {
                "RSI": {
                    "rsi":    {"color": "#2ecc71"},
                    "rsi_1h": {"color": "#8e44ad"},
                },
                "Volume": {
                    "volume": {"color": "#bdc3c7", "type": "bar"},
                    "vol_ma": {"color": "#e74c3c", "type": "line"},
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
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=self.ema_long_period)
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=self.rsi_period)
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
        # Solo regime Accumulo: sotto EMA200, RSI oversold, candela martello
        below = dataframe["close"] < dataframe["ema_long"]

        accumulo = (
            (dataframe["rsi"] < self.rsi_t1.value)
            & (dataframe["lower_shadow"] > dataframe["body"] * 1.5)
            & (dataframe["body"] > 0)
        )

        entry = below & accumulo & (dataframe["volume"] > 0)

        # Cooldown: sopprime il segnale se c'è stato un ingresso nelle ultime cooldown_bars candele
        recent = (
            entry.rolling(window=self.cooldown_bars, min_periods=1).max().shift(1).fillna(0)
        )
        dataframe.loc[entry & (recent == 0), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty or len(dataframe) < 2:
            return None

        last = dataframe.iloc[-1]
        next_tranche = nr + 1
        rsi_thresh = self._rsi_thresh(next_tranche)

        # Solo regime Accumulo per le tranche successive
        signal = (
            bool(last["close"] < last["ema_long"])
            and bool(last["rsi"] < rsi_thresh)
            and bool(last["lower_shadow"] > last["body"] * 1.5)
            and bool(last["body"] > 0)
        )

        if not signal:
            return None

        stake = min(self.tranche_usd, max_stake)
        if min_stake and stake < min_stake:
            return None

        return stake
