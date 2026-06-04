from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.strategy_helper import merge_informative_pair
from pandas import DataFrame


class DcaBBRSI(IStrategy):
    """
    DCA con Bollinger Bands + RSI — tre livelli di RSI, exit trailing come DcaProgressive.

    Filtro BTC macro (1h):
        btc_ema20 >= btc_ema50 * btc_ema_ratio  OR  btc_rsi > btc_rsi_min
        → blocca entry e DCA quando BTC è in trend ribassista forte

    Entry iniziale:
        BTC ok AND close < BB lower (15m) AND rsi < rsi_buy

    DCA sopra costo medio:
        BTC ok AND close < BB lower (15m) AND rsi < rsi_dca_above  (più permissivo)

    DCA sotto costo medio:
        BTC ok AND close < BB lower (15m) AND rsi < rsi_dca_below

    Exit (trailing):
        close >= BB upper (1h) AND rsi_1h > rsi_sell → attiva trailing di trailing_pct%

    Time exit:
        dopo time_exit_days giorni se profit < time_exit_loss_threshold; hard cap a time_exit_days*2

    Tranche: [6, 9, 15, 25, 45] USDC → max 100 USDC per trade
    Stoploss: disabilitato (-99%) finché non scatta il trailing
    """

    INTERFACE_VERSION = 3

    position_adjustment_enable = True
    use_custom_stoploss = True
    max_entry_position_adjustment = 4  # 5 tranche totali

    timeframe = "15m"
    inf_timeframe = "1h"
    startup_candle_count = 50
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    tranche_sizes: list = [6, 9, 15, 25, 45]

    # --- Parametri ottimizzabili ---
    bb_period      = IntParameter(10, 30,  default=20,  space="buy",  optimize=True)
    bb_std         = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True)
    rsi_buy        = IntParameter(20, 45,  default=30,  space="buy",  optimize=True)
    rsi_dca_above  = IntParameter(30, 55,  default=40,  space="buy",  optimize=True)
    rsi_dca_below  = IntParameter(10, 30,  default=30,  space="buy",  optimize=True)
    btc_ema_ratio  = DecimalParameter(0.90, 1.00, default=0.97, decimals=2, space="buy", optimize=True)
    btc_rsi_min    = IntParameter(20, 50, default=35, space="buy", optimize=True)
    rsi_sell       = IntParameter(55, 80,  default=72,  space="sell", optimize=True)
    trailing_pct   = DecimalParameter(0.1, 1.0, default=0.2, decimals=1, space="sell", optimize=True)
    time_exit_days = IntParameter(5, 20, default=10, space="sell", optimize=True)
    time_exit_loss_threshold = DecimalParameter(-0.15, -0.01, default=-0.09, decimals=2, space="sell", optimize=True)

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "bb_lower":    {"color": "#3498db", "type": "line"},
                "bb_mid":      {"color": "#95a5a6", "type": "line"},
                "bb_upper":    {"color": "#3498db", "type": "line"},
                "bb_upper_1h": {"color": "#9b59b6", "type": "line"},
            },
            "subplots": {
                "RSI": {
                    "rsi":    {"color": "#e74c3c"},
                    "rsi_1h": {"color": "#8e44ad"},
                },
            },
        }

    def informative_pairs(self):
        pairs = [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist()]
        btc_pair = f"BTC/{self.config['stake_currency']}"
        pairs.append((btc_pair, self.inf_timeframe))
        return pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=self.bb_period.value,
            stds=self.bb_std.value,
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_mid"]   = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]

        # --- 1h informative: BB upper + RSI ---
        inf_df = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_timeframe)
        if not inf_df.empty:
            bb_upper_1h, _, _ = ta.BBANDS(
                inf_df["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
            )
            inf_df["bb_upper"] = bb_upper_1h
            inf_df["rsi"] = ta.RSI(inf_df["close"], timeperiod=14)
            dataframe = merge_informative_pair(
                dataframe, inf_df, self.timeframe, self.inf_timeframe, ffill=True
            )

        # --- BTC 1h macro filter: EMA20, EMA50, RSI ---
        btc_pair = f"BTC/{self.config['stake_currency']}"
        btc_df = self.dp.get_pair_dataframe(pair=btc_pair, timeframe=self.inf_timeframe)
        if not btc_df.empty:
            btc_df["btc_ema20"] = ta.EMA(btc_df["close"], timeperiod=20)
            btc_df["btc_ema50"] = ta.EMA(btc_df["close"], timeperiod=50)
            btc_df["btc_rsi"]   = ta.RSI(btc_df["close"], timeperiod=14)
            # rinomino open/high/low/close/volume per evitare conflitti col merge
            btc_df = btc_df[["date", "btc_ema20", "btc_ema50", "btc_rsi"]].copy()
            btc_df.rename(columns={"date": "date"}, inplace=True)
            dataframe = merge_informative_pair(
                dataframe, btc_df, self.timeframe, self.inf_timeframe, ffill=True
            )

        return dataframe

    def _btc_ok(self, dataframe: DataFrame) -> pd.Series:
        ema_col   = "btc_ema20_1h"
        ema50_col = "btc_ema50_1h"
        rsi_col   = "btc_rsi_1h"
        if ema_col not in dataframe.columns:
            return pd.Series(True, index=dataframe.index)
        return (
            (dataframe[ema_col] >= dataframe[ema50_col] * self.btc_ema_ratio.value)
            | (dataframe[rsi_col] > self.btc_rsi_min.value)
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        condition = (
            self._btc_ok(dataframe)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi"] < self.rsi_buy.value)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[condition, "enter_long"] = 1
        dataframe.loc[condition, "enter_tag"]  = "bb_rsi"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        ct = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        open_date = trade.open_date_utc
        if open_date.tzinfo is None:
            open_date = open_date.replace(tzinfo=timezone.utc)
        age = ct - open_date

        if age >= timedelta(days=self.time_exit_days.value * 2):
            return "time_exit_hard"
        if age >= timedelta(days=self.time_exit_days.value) and current_profit < self.time_exit_loss_threshold.value:
            return "time_exit"

        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        if current_profit <= 0:
            return -0.99

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return -0.99

        last = dataframe.iloc[-1]
        bb_upper_1h = last.get(f"bb_upper_{self.inf_timeframe}")
        rsi_1h = last.get(f"rsi_{self.inf_timeframe}")

        if pd.isna(bb_upper_1h) or pd.isna(rsi_1h):
            return -0.99

        if current_rate >= bb_upper_1h and rsi_1h > self.rsi_sell.value:
            return -(self.trailing_pct.value / 100)

        return -0.99

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

        if nr >= len(self.tranche_sizes):
            return None

        if trade.has_open_orders:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None
        last = dataframe.iloc[-1]
        bb_lower_ok = bool(last["close"] < last["bb_lower"])

        # Filtro BTC macro
        ema_col   = "btc_ema20_1h"
        ema50_col = "btc_ema50_1h"
        rsi_col   = "btc_rsi_1h"
        if ema_col in dataframe.columns:
            btc_ok = (
                float(last[ema_col]) >= float(last[ema50_col]) * self.btc_ema_ratio.value
                or float(last[rsi_col]) > self.btc_rsi_min.value
            )
            if not btc_ok:
                return None

        avg_cost = trade.stake_amount / trade.amount
        above_avg = current_rate >= avg_cost

        if above_avg:
            if not (bb_lower_ok and bool(last["rsi"] < self.rsi_dca_above.value)):
                return None
        else:
            if not (bb_lower_ok and bool(last["rsi"] < self.rsi_dca_below.value)):
                return None

        stake = min(self.tranche_sizes[nr], max_stake)
        if min_stake and stake < min_stake:
            return None

        return stake
