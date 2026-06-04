from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.strategy_helper import merge_informative_pair
from pandas import DataFrame


class DcaTimeBTC_Market(IStrategy):
    """
    Variante DcaTimeBTC: entry a market sulla candela dopo la chiusura del trade precedente.
    Nessun filtro sull'entry — cooldown_bars=1 garantisce 1 candela di pausa dopo il close.
    DCA e exit invariati rispetto a DcaTimeBTC.
    """

    INTERFACE_VERSION = 3

    position_adjustment_enable = True
    max_entry_position_adjustment = -1
    use_custom_stoploss = True

    timeframe = "15m"
    inf_timeframe = "1h"
    startup_candle_count = 50
    process_only_new_candles = True
    cooldown_bars = 1

    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    # --- Parametri (usati per DCA e exit, non per entry) ---
    bb_period = IntParameter(10, 30,  default=30,  space="buy", optimize=True)
    bb_std    = DecimalParameter(1.5, 3.0, default=1.5, decimals=1, space="buy", optimize=True)
    rsi_buy   = IntParameter(30, 80,  default=80,  space="buy", optimize=False)

    dca_drop_1  = DecimalParameter(0.5,  60.0, default=2.62,  decimals=2, space="buy", optimize=True)
    dca_drop_2  = DecimalParameter(0.5,  60.0, default=6.58, decimals=2, space="buy", optimize=True)
    dca_drop_3  = DecimalParameter(0.5,  60.0, default=2.33,  decimals=2, space="buy", optimize=True)
    dca_drop_4  = DecimalParameter(0.5,  60.0, default=4.65,  decimals=2, space="buy", optimize=True)
    dca_drop_5  = DecimalParameter(0.5,  60.0, default=6.55, decimals=2, space="buy", optimize=True)
    dca_drop_6  = DecimalParameter(0.5,  60.0, default=8.0,  decimals=2, space="buy", optimize=True)
    dca_drop_7  = DecimalParameter(0.5,  60.0, default=10.0, decimals=2, space="buy", optimize=True)
    dca_drop_8  = DecimalParameter(0.5,  60.0, default=12.0, decimals=2, space="buy", optimize=True)
    dca_drop_9  = DecimalParameter(0.5,  60.0, default=14.0, decimals=2, space="buy", optimize=True)
    dca_drop_10 = DecimalParameter(0.5,  60.0, default=16.0, decimals=2, space="buy", optimize=True)

    rsi_sell     = IntParameter(60, 80, default=70, space="sell", optimize=True)
    trailing_pct = DecimalParameter(0.05, 2.0, default=1.64, decimals=2, space="sell", optimize=True)

    @property
    def _dca_bands(self) -> list:
        return sorted([
            self.dca_drop_1.value  / 100,
            self.dca_drop_2.value  / 100,
            self.dca_drop_3.value  / 100,
            self.dca_drop_4.value  / 100,
            self.dca_drop_5.value  / 100,
            self.dca_drop_6.value  / 100,
            self.dca_drop_7.value  / 100,
            self.dca_drop_8.value  / 100,
            self.dca_drop_9.value  / 100,
            self.dca_drop_10.value / 100,
        ])

    def informative_pairs(self):
        return [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        inf_df = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_timeframe)
        if not inf_df.empty:
            bb_upper, _, bb_lower = ta.BBANDS(
                inf_df["close"],
                timeperiod=self.bb_period.value,
                nbdevup=self.bb_std.value,
                nbdevdn=self.bb_std.value,
                matype=0,
            )
            inf_df["bb_upper"] = bb_upper
            inf_df["bb_lower"] = bb_lower
            inf_df["rsi"] = ta.RSI(inf_df["close"], timeperiod=14)
            dataframe = merge_informative_pair(
                dataframe, inf_df, self.timeframe, self.inf_timeframe, ffill=True
            )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Entry a market: sempre aperto, nessun filtro
        dataframe.loc[dataframe["volume"] > 0, "enter_long"] = 1
        dataframe.loc[dataframe["volume"] > 0, "enter_tag"] = "entry"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe

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
        if trade.has_open_orders:
            return None

        stake = 6.0
        if min_stake and stake < min_stake:
            return None
        if stake > max_stake:
            return None

        avg_cost = trade.stake_amount / trade.amount

        # --- Reset fasce quando il trade torna in positivo ---
        if current_profit > 0:
            if trade.get_custom_data("bands_unlocked", default=0) > 0:
                trade.set_custom_data("bands_unlocked", 0)
                trade.set_custom_data("bands_consumed", 0)
            return None

        # --- Aggiorna fasce sbloccate (mai decresce) ---
        bands = self._dca_bands
        currently_triggered = sum(1 for b in bands if current_rate <= avg_cost * (1 - b))
        stored_unlocked = trade.get_custom_data("bands_unlocked", default=0)
        bands_unlocked = max(currently_triggered, stored_unlocked)

        if bands_unlocked > stored_unlocked:
            trade.set_custom_data("bands_unlocked", bands_unlocked)

        bands_consumed = trade.get_custom_data("bands_consumed", default=0)
        pending = bands_unlocked - bands_consumed

        # --- Cooldown: max 1 DCA ogni 48 ore ---
        filled_times = [
            o.order_filled_date for o in trade.orders
            if o.ft_order_side == "buy" and o.status == "closed" and o.order_filled_date is not None
        ]
        if filled_times:
            last_entry = max(filled_times)
            if (current_time - last_entry).total_seconds() < 48 * 3600:
                return None

        # --- Indicatori 1h ---
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None
        last        = dataframe.iloc[-1]
        bb_lower_1h = last.get(f"bb_lower_{self.inf_timeframe}")
        rsi_1h      = last.get(f"rsi_{self.inf_timeframe}")
        if pd.isna(bb_lower_1h) or pd.isna(rsi_1h):
            return None
        signal = last["close"] < bb_lower_1h and rsi_1h < self.rsi_buy.value

        # --- DCA below: fasce pendenti ---
        if pending > 0 and signal:
            bulk_stake = stake * pending
            if bulk_stake > max_stake:
                bulk_stake = max_stake
                pending = int(bulk_stake // stake)
            trade.set_custom_data("bands_consumed", bands_consumed + pending)
            return bulk_stake, f"dca_below_{bands_consumed + 1}"

        # --- DCA time: prezzo fermo in rosso tra fasce ---
        if pending == 0 and current_profit <= 0 and signal:
            return stake, "dca_time"

        # --- DCA above ---
        if pending == 0 and current_rate > avg_cost and signal:
            return stake, "dca_above"

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

        last        = dataframe.iloc[-1]
        bb_upper_1h = last.get(f"bb_upper_{self.inf_timeframe}")
        rsi_1h      = last.get(f"rsi_{self.inf_timeframe}")

        if pd.isna(bb_upper_1h) or pd.isna(rsi_1h):
            return -0.99

        if current_rate >= bb_upper_1h and rsi_1h > self.rsi_sell.value:
            if current_profit < self.trailing_pct.value / 100:
                return -0.99
            trailing  = -(self.trailing_pct.value / 100)
            breakeven = -(current_profit / (1 + current_profit))
            return max(trailing, breakeven)

        return -0.99
