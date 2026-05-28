from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.strategy.strategy_helper import merge_informative_pair
from pandas import DataFrame


class DcaProgressiveV2(IStrategy):
    """
    DcaProgressiveV2 — Ibrido multi-TF

    Ingressi (da AdaptiveSpotX):
      - Filtro BTC macro su 4h
      - Cond 1 pullback:      trend 4h + RSI 1h + RSI/volume/CTI 5m
      - Cond 2 mean_rev:      RSI 1h oversold + BB inferiore 1h+5m
      - Cond 3 bull_breakout: regime toro 4h + volume spike + EMA cross 5m

    Uscita (da DcaProgressive):
      - Trailing stop attivo solo quando close >= BB_sup_1h E RSI_1h > rsi_sell
      - Time exit dopo 10 giorni se profit < soglia, hard cap 20 giorni
      - stoploss = -0.99 (virtualmente disabilitato), minimal_roi disabilitato

    DCA: 2 livelli a trigger ottimizzabili, importi fissi 30+50 USDC
    Timeframe: 5m  |  Informativi: 1h, 4h
    """

    INTERFACE_VERSION = 3

    protections = [{"method": "CooldownPeriod", "stop_duration_candles": 3}]

    position_adjustment_enable = True
    use_custom_stoploss = True

    timeframe = "5m"
    startup_candle_count = 480
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    # Importi DCA fissi in USDC
    # Entry: stake_amount (20) + DCA1: 30 + DCA2: 50 = 100 USDC max per trade
    dca1_amount: float = 30.0
    dca2_amount: float = 50.0

    # ------------------------------------------------------------------ #
    #  Parametri hyperopt                                                  #
    # ------------------------------------------------------------------ #

    # BTC macro filter
    btc_filter_rsi_min   = IntParameter(35, 55, default=45, space="buy", optimize=True, load=True)
    btc_filter_ema_ratio = DecimalParameter(0.93, 1.00, default=0.97, decimals=2, space="buy", optimize=True, load=True)

    # Condizione 1: Trend Pullback
    cond1_enabled     = CategoricalParameter([True, False], default=True,  space="buy", optimize=True, load=True)
    cond1_rsi_1h_min  = IntParameter(20, 42,    default=30,   space="buy", optimize=True, load=True)
    cond1_rsi_1h_max  = IntParameter(44, 65,    default=55,   space="buy", optimize=True, load=True)
    cond1_rsi_5m_max  = IntParameter(35, 55,    default=45,   space="buy", optimize=True, load=True)
    cond1_cmf_1h_min  = DecimalParameter(-0.30,  0.00, default=-0.15, decimals=2, space="buy", optimize=True, load=True)
    cond1_volume_mult = DecimalParameter(1.0,    2.0,  default=1.1,   decimals=1, space="buy", optimize=True, load=True)
    cond1_cti_max     = DecimalParameter(0.3,    0.8,  default=0.5,   decimals=1, space="buy", optimize=True, load=True)

    # Condizione 2: Mean-Reversion Bounce
    cond2_enabled    = CategoricalParameter([True, False], default=True,  space="buy", optimize=True, load=True)
    cond2_rsi_1h_max = IntParameter(25, 45, default=35, space="buy", optimize=True, load=True)
    cond2_rsi_5m_max = IntParameter(22, 40, default=32, space="buy", optimize=True, load=True)
    cond2_cmf_1h_min = DecimalParameter(-0.50, -0.10, default=-0.35, decimals=2, space="buy", optimize=True, load=True)

    # Condizione 3: Bull Breakout
    cond3_enabled     = CategoricalParameter([True, False], default=False, space="buy", optimize=True, load=True)
    cond3_rsi_4h_min  = IntParameter(45, 60, default=50, space="buy", optimize=True, load=True)
    cond3_rsi_4h_max  = IntParameter(55, 75, default=65, space="buy", optimize=True, load=True)
    cond3_rsi_5m_min  = IntParameter(45, 62, default=52, space="buy", optimize=True, load=True)
    cond3_rsi_5m_max  = IntParameter(65, 80, default=72, space="buy", optimize=True, load=True)
    cond3_volume_mult = DecimalParameter(1.5,  3.5,  default=2.0,  decimals=1, space="buy", optimize=True, load=True)

    # DCA triggers
    dca1_trigger = DecimalParameter(-0.15, -0.05, default=-0.08, decimals=2, space="buy", optimize=True, load=True)
    dca2_trigger = DecimalParameter(-0.25, -0.10, default=-0.14, decimals=2, space="buy", optimize=True, load=True)

    # Exit
    rsi_sell                 = IntParameter(55, 80, default=72, space="sell", optimize=True, load=True)
    trailing_pct             = DecimalParameter(0.2, 3.0, default=0.2, decimals=1, space="sell", optimize=True, load=True)
    time_exit_loss_threshold = DecimalParameter(-0.10, 0.00, default=-0.09, decimals=2, space="sell", optimize=True, load=True)

    bb_timeframe: str = "1h"

    bb_timeframe: str = "1h"

    # ------------------------------------------------------------------ #
    #  Informative pairs                                                   #
    # ------------------------------------------------------------------ #
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative = [(pair, tf) for pair in pairs for tf in ["1h", "4h"]]
        stake  = self.config.get("stake_currency", "USDT")
        stable = {"USDT", "USDC", "BUSD", "DAI", "FDUSD", "USD"}
        btc_pair = f"BTC/{stake}" if stake in stable else "BTC/USDT"
        informative.append((btc_pair, "4h"))
        return informative

    # ------------------------------------------------------------------ #
    #  Helper indicatori per timeframe informativi                         #
    # ------------------------------------------------------------------ #
    def _indicators_4h(self, df: DataFrame) -> DataFrame:
        df["ema_20"]  = ta.EMA(df, timeperiod=20)
        df["ema_50"]  = ta.EMA(df, timeperiod=50)
        df["ema_200"] = ta.EMA(df, timeperiod=200)
        df["rsi"]     = ta.RSI(df, timeperiod=14)
        bull = (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]) & (df["rsi"] > 50)
        bear = (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]) & (df["rsi"] < 50)
        df["regime"] = np.where(bull, 1, np.where(bear, -1, 0))
        return df

    def _indicators_1h(self, df: DataFrame) -> DataFrame:
        df["ema_20"]    = ta.EMA(df, timeperiod=20)
        df["ema_50"]    = ta.EMA(df, timeperiod=50)
        df["rsi"]       = ta.RSI(df, timeperiod=14)
        bb              = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df["bb_upper"]  = bb["upperband"]
        df["bb_lower"]  = bb["lowerband"]
        df["cmf"]       = pta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
        df["volume_ma"] = ta.SMA(df["volume"], timeperiod=20)
        return df

    def _btc_indicators_4h(self, df: DataFrame) -> DataFrame:
        df["ema_20"] = ta.EMA(df, timeperiod=20)
        df["ema_50"] = ta.EMA(df, timeperiod=50)
        df["rsi"]    = ta.RSI(df, timeperiod=14)
        df.rename(columns=lambda c: f"btc_{c}" if c != "date" else c, inplace=True)
        return df

    # ------------------------------------------------------------------ #
    #  Indicatori                                                          #
    # ------------------------------------------------------------------ #
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        stake    = self.config.get("stake_currency", "USDT")
        stable   = {"USDT", "USDC", "BUSD", "DAI", "FDUSD", "USD"}
        btc_pair = f"BTC/{stake}" if stake in stable else "BTC/USDT"

        # 4h corrente
        info_4h = self.dp.get_pair_dataframe(metadata["pair"], "4h")
        if not info_4h.empty:
            info_4h = self._indicators_4h(info_4h)
            df = merge_informative_pair(df, info_4h, self.timeframe, "4h", ffill=True)
            df.drop(columns=["date_4h"], inplace=True, errors="ignore")

        # 1h corrente
        info_1h = self.dp.get_pair_dataframe(metadata["pair"], "1h")
        if not info_1h.empty:
            info_1h = self._indicators_1h(info_1h)
            df = merge_informative_pair(df, info_1h, self.timeframe, "1h", ffill=True)
            df.drop(columns=["date_1h"], inplace=True, errors="ignore")

        # BTC 4h macro filter (salta se il pair corrente È BTC)
        if metadata["pair"] != btc_pair:
            btc_4h = self.dp.get_pair_dataframe(btc_pair, "4h")
            if not btc_4h.empty:
                btc_4h = self._btc_indicators_4h(btc_4h)
                df = merge_informative_pair(df, btc_4h, self.timeframe, "4h", ffill=True)
                df.drop(columns=["date_4h"], inplace=True, errors="ignore")

        # Indicatori 5m
        df["rsi"]       = ta.RSI(df, timeperiod=14)
        df["ema_8"]     = ta.EMA(df, timeperiod=8)
        df["ema_21"]    = ta.EMA(df, timeperiod=21)
        df["volume_ma"] = ta.SMA(df["volume"], timeperiod=20)
        bb5             = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df["bb_upper"]  = bb5["upperband"]
        df["bb_lower"]  = bb5["lowerband"]
        df["cti"]       = pta.cti(df["close"], length=20)

        return df

    # ------------------------------------------------------------------ #
    #  Segnali di ingresso                                                 #
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_tag"]  = ""

        def col(name, default):
            return df[name] if name in df.columns else pd.Series(default, index=df.index)

        # Filtro BTC macro
        btc_ok = (
            col("btc_ema_20_4h", df["close"]) >= col("btc_ema_50_4h", df["close"]) * self.btc_filter_ema_ratio.value
        ) | (col("btc_rsi_4h", 50) > self.btc_filter_rsi_min.value)

        # Cond 1: Trend Pullback
        cond1 = (
            self.cond1_enabled.value
            & btc_ok
            & (col("regime_4h", 0) >= 0)
            & (col("rsi_4h",    50) < 65)
            & col("rsi_1h", 50).between(self.cond1_rsi_1h_min.value, self.cond1_rsi_1h_max.value)
            & (col("cmf_1h", 0.0)  > self.cond1_cmf_1h_min.value)
            & (df["close"]         > col("ema_50_1h", df["close"]) * 0.985)
            & (df["rsi"]           < self.cond1_rsi_5m_max.value)
            & (df["volume"]        > df["volume_ma"] * self.cond1_volume_mult.value)
            & (df["cti"]           < self.cond1_cti_max.value)
        )

        # Cond 2: Mean-Reversion Bounce
        cond2 = (
            self.cond2_enabled.value
            & btc_ok
            & (col("rsi_1h",  50)  < self.cond2_rsi_1h_max.value)
            & (df["close"]         <= col("bb_lower_1h", df["close"]) * 1.02)
            & (col("cmf_1h", 0.0)  > self.cond2_cmf_1h_min.value)
            & (df["rsi"]           < self.cond2_rsi_5m_max.value)
            & (df["close"]         <= df["bb_lower"] * 1.01)
            & (~cond1)
        )

        # Cond 3: Bull Breakout
        cond3 = (
            self.cond3_enabled.value
            & btc_ok
            & (col("regime_4h", 0) == 1)
            & col("rsi_4h", 50).between(self.cond3_rsi_4h_min.value, self.cond3_rsi_4h_max.value)
            & (df["volume"]  > df["volume_ma"] * self.cond3_volume_mult.value)
            & df["rsi"].between(self.cond3_rsi_5m_min.value, self.cond3_rsi_5m_max.value)
            & (df["close"]   > df["ema_8"])
            & (df["ema_8"]   > df["ema_21"])
            & (~cond1)
            & (~cond2)
        )

        df.loc[cond1, ["enter_long", "enter_tag"]] = [1, "pullback"]
        df.loc[cond2, ["enter_long", "enter_tag"]] = [1, "mean_rev"]
        df.loc[cond3, ["enter_long", "enter_tag"]] = [1, "bull_breakout"]

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        return df

    # ------------------------------------------------------------------ #
    #  Time exit condizionale                                              #
    # ------------------------------------------------------------------ #
    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        ct = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        open_date = trade.open_date_utc
        if open_date.tzinfo is None:
            open_date = open_date.replace(tzinfo=timezone.utc)
        age = ct - open_date

        if age >= timedelta(days=20):
            return "time_exit_max"
        if age >= timedelta(days=10) and current_profit < self.time_exit_loss_threshold.value:
            return "time_exit"
        return None

    # ------------------------------------------------------------------ #
    #  Trailing stop: si attiva quando close >= BB_sup_1h E RSI_1h alto   #
    # ------------------------------------------------------------------ #
    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs) -> float:
        if current_profit <= 0:
            return -0.99

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return -0.99

        last    = dataframe.iloc[-1]
        bb_col  = f"bb_upper_{self.bb_timeframe}"
        rsi_col = f"rsi_{self.bb_timeframe}"

        if bb_col not in dataframe.columns or rsi_col not in dataframe.columns:
            return -0.99

        bb_upper = last[bb_col]
        rsi_1h   = last[rsi_col]

        if pd.isna(bb_upper) or pd.isna(rsi_1h):
            return -0.99

        if current_rate >= bb_upper and rsi_1h > self.rsi_sell.value:
            return -(self.trailing_pct.value / 100)

        return -0.99

    # ------------------------------------------------------------------ #
    #  DCA                                                                 #
    # ------------------------------------------------------------------ #
    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs) -> Optional[float]:
        buys = trade.nr_of_successful_entries

        if buys >= 3 or trade.has_open_orders:
            return None

        if buys == 1 and current_profit < self.dca1_trigger.value:
            stake = min(self.dca1_amount, max_stake)
            return stake if not min_stake or stake >= min_stake else None

        if buys == 2 and current_profit < self.dca2_trigger.value:
            stake = min(self.dca2_amount, max_stake)
            return stake if not min_stake or stake >= min_stake else None

        return None
