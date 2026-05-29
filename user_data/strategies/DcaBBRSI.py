from datetime import datetime, timezone
from typing import Optional

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class DcaBBRSI(IStrategy):
    """
    DCA con Bollinger Bands + RSI.

    Entry iniziale e DCA sopra costo medio:
        close < BB lower AND RSI < rsi_buy

    DCA sotto costo medio (mediazione al ribasso):
        step progressivo dal costo medio corrente
        step_n = dca_step * (dca_multiplier ^ (n-1))
        es. step=3%, mult=1.8 → -3% / -5.4% / -9.7% / -17.5%

    Uscita: profitto fisso >= profit_target_usdc
    Tranche: [6, 9, 15, 25, 45] USDC → max 100 USDC per trade
    Stoploss: disabilitato (-99%)
    """

    INTERFACE_VERSION = 3

    position_adjustment_enable = True
    max_entry_position_adjustment = 4  # 5 tranche totali

    timeframe = "5m"
    startup_candle_count = 50
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    tranche_sizes: list = [6, 9, 15, 25, 45]

    # --- Parametri ottimizzabili ---
    bb_period    = IntParameter(10, 30,  default=20,  space="buy",  optimize=True)
    bb_std       = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True)
    rsi_buy      = IntParameter(20, 45,  default=30,  space="buy",  optimize=True)
    dca_step     = DecimalParameter(0.02, 0.10, default=0.03, decimals=2, space="buy", optimize=True)
    dca_multiplier = DecimalParameter(1.3, 2.5, default=1.8, decimals=1, space="buy", optimize=True)
    profit_target_usdc = DecimalParameter(2.0, 20.0, default=5.0, decimals=1, space="sell", optimize=True)

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "bb_lower": {"color": "#3498db", "type": "line"},
                "bb_mid":   {"color": "#95a5a6", "type": "line"},
                "bb_upper": {"color": "#3498db", "type": "line"},
            },
            "subplots": {
                "RSI": {
                    "rsi": {"color": "#e74c3c"},
                },
            },
        }

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

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        condition = (
            (dataframe["close"] < dataframe["bb_lower"])
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
        if trade.calc_profit(rate=current_rate) >= self.profit_target_usdc.value:
            return "profit_target"
        return None

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

        above_avg = current_rate >= trade.open_rate

        if above_avg:
            # DCA sopra costo medio: stessa condizione BB+RSI dell'entry iniziale
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            if dataframe.empty:
                return None
            last = dataframe.iloc[-1]
            if not (
                bool(last["close"] < last["bb_lower"])
                and bool(last["rsi"] < self.rsi_buy.value)
            ):
                return None
        else:
            # DCA sotto costo medio: step percentuale progressivo
            # nr=1 → step base, nr=2 → step*mult, nr=3 → step*mult^2, ecc.
            dca_index = nr - 1
            required_drop = self.dca_step.value * (self.dca_multiplier.value ** dca_index)
            drop_from_avg = (current_rate - trade.open_rate) / trade.open_rate
            if drop_from_avg > -required_drop:
                return None

        stake = min(self.tranche_sizes[nr], max_stake)
        if min_stake and stake < min_stake:
            return None

        return stake
