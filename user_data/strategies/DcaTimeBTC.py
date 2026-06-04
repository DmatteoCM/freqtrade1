from datetime import datetime
from typing import Optional

import pandas as pd
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.strategy_helper import merge_informative_pair
from pandas import DataFrame


class DcaTimeBTC(IStrategy):
    """
    DCA a tempo su BTC.

    Entry:  close < BB lower 1h  AND  RSI 1h < rsi_buy
    DCA:    stesso segnale, stake fisso 6 USDC, max 1 ordine ogni 24h, solo se profit <= 0
    Exit:   close >= BB upper 1h AND RSI 1h > rsi_sell → trailing di trailing_pct%
            floor al breakeven: non chiude mai in negativo
    """

    INTERFACE_VERSION = 3

    position_adjustment_enable = True
    max_entry_position_adjustment = -1
    use_custom_stoploss = True

    timeframe = "15m"
    inf_timeframe = "1h"
    startup_candle_count = 50
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    bb_period = IntParameter(10, 30,  default=20,  space="buy", optimize=True)
    bb_std    = DecimalParameter(1.5, 3.0, default=2.1, decimals=1, space="buy", optimize=True)
    rsi_buy   = IntParameter(30, 80,  default=80,  space="buy", optimize=False)

    rsi_sell     = IntParameter(60, 80, default=76, space="sell", optimize=True)
    trailing_pct = DecimalParameter(0.05, 2.0, default=1.94, decimals=2, space="sell", optimize=True)

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
        bb_lower_col = f"bb_lower_{self.inf_timeframe}"
        rsi_col      = f"rsi_{self.inf_timeframe}"
        if bb_lower_col not in dataframe.columns:
            return dataframe
        condition = (
            (dataframe["close"] < dataframe[bb_lower_col])
            & (dataframe[rsi_col] < self.rsi_buy.value)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[condition, "enter_long"] = 1
        dataframe.loc[condition, "enter_tag"] = "entry"
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

        if current_profit > 0:
            return None

        stake = 6.0
        if min_stake and stake < min_stake:
            return None
        if stake > max_stake:
            return None

        # Cooldown: max 1 DCA ogni 24 ore
        filled_times = [
            o.order_filled_date for o in trade.orders
            if o.ft_order_side == "buy" and o.status == "closed" and o.order_filled_date is not None
        ]
        if filled_times:
            last_entry = max(filled_times)
            if (current_time - last_entry).total_seconds() < 24 * 3600:
                return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None
        last        = dataframe.iloc[-1]
        bb_lower_1h = last.get(f"bb_lower_{self.inf_timeframe}")
        rsi_1h      = last.get(f"rsi_{self.inf_timeframe}")
        if pd.isna(bb_lower_1h) or pd.isna(rsi_1h):
            return None

        if last["close"] < bb_lower_1h and rsi_1h < self.rsi_buy.value:
            self.dp.send_msg(
                f"⏱️ DCA | BTC: {current_rate:.0f} USDC | Profit: {current_profit * 100:.2f}%"
            )
            return stake, "dca"

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
