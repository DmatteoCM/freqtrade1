from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
import logging

logger = logging.getLogger(__name__)

class SimpleDCA(IStrategy):
    """
    DCA con doppio regime EMA5/EMA200 su 1d:
    - Bear (EMA5 < EMA200 1d): entry/DCA se RSI <= rsi_bear (default 23)
      Fallback bear: se no RSI in 3gg e cooldown 72h, entry/DCA su BB lower 4h
    - Bull (EMA5 > EMA200 1d): entry/DCA se RSI <= rsi_bull (default 30)
    Exit:
    - Bear: profit target 2.5%
    - Bull: RSI > 70 e current_profit > 0
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    stoploss = -0.99
    use_custom_stoploss = False

    minimal_roi = {"0": 100}
    trailing_stop = False

    position_adjustment_enable = True
    max_open_trades = 1

    startup_candle_count = 200

    # ─────────────────────────────────────────
    # PARAMETRI
    # ─────────────────────────────────────────
    rsi_bear          = IntParameter(10, 30, default=23, space="buy", optimize=False)
    rsi_bull          = IntParameter(20, 40, default=30, space="buy", optimize=False)
    cooldown_hours    = IntParameter(1, 72, default=6, space="buy", optimize=False)
    dca_stake_amount  = DecimalParameter(5.0, 50.0, default=10.0, decimals=1, space="buy", optimize=False)
    profit_target_pct = DecimalParameter(0.005, 0.10, default=0.025, decimals=3, space="sell", optimize=False)

    # ─────────────────────────────────────────
    # INDICATORI
    # ─────────────────────────────────────────
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, '1d') for pair in pairs] + [(pair, '4h') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # EMA5 e EMA200 su 1d per regime bull/bear
        inf_1d = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1d')
        inf_1d['ema5'] = ta.EMA(inf_1d['close'], timeperiod=5)
        inf_1d['ema200'] = ta.EMA(inf_1d['close'], timeperiod=200)
        dataframe = merge_informative_pair(dataframe, inf_1d, self.timeframe, '1d', ffill=True)

        # BB lower band su 4h — fallback entry/DCA in bear
        inf_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='4h')
        bbands = ta.BBANDS(inf_4h, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        inf_4h['bb_lower'] = bbands['lowerband']
        inf_4h_slim = inf_4h[['date', 'bb_lower']].copy()
        dataframe = merge_informative_pair(dataframe, inf_4h_slim, self.timeframe, '4h', ffill=True)

        return dataframe

    # ─────────────────────────────────────────
    # ENTRY
    # ─────────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_bull = dataframe['ema5_1d'] > dataframe['ema200_1d']

        bear_rsi_entry = (~is_bull) & (dataframe['rsi'] <= self.rsi_bear.value)
        bull_entry     = is_bull & (dataframe['rsi'] <= self.rsi_bull.value)
        bear_bb_entry  = (~is_bull) & (dataframe['close'] <= dataframe['bb_lower_4h'])

        dataframe.loc[bear_rsi_entry, 'enter_long'] = 1
        dataframe.loc[bear_rsi_entry, 'enter_tag'] = 'bear_entry'

        dataframe.loc[bull_entry, 'enter_long'] = 1
        dataframe.loc[bull_entry, 'enter_tag'] = 'bull_entry'

        # BB entry solo dove RSI non ha già triggerato (RSI ha priorità)
        dataframe.loc[bear_bb_entry & ~bear_rsi_entry, 'enter_long'] = 1
        dataframe.loc[bear_bb_entry & ~bear_rsi_entry, 'enter_tag'] = 'bear_bb_entry'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe

    # ─────────────────────────────────────────
    # EXIT
    # ─────────────────────────────────────────
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]
        ema5_1d = last_candle.get('ema5_1d')
        ema200_1d = last_candle.get('ema200_1d')

        if ema5_1d is None or ema200_1d is None:
            if current_profit >= self.profit_target_pct.value:
                return "profit_target"
            return None

        is_bull = ema5_1d > ema200_1d

        if is_bull:
            if last_candle['rsi'] > 70 and current_profit > 0:
                return "bull_rsi_exit"
        else:
            if current_profit >= self.profit_target_pct.value:
                return "profit_target"

        return None

    # ─────────────────────────────────────────
    # DCA
    # ─────────────────────────────────────────
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                               current_rate: float, current_profit: float,
                               min_stake: float, max_stake: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe is None or dataframe.empty or len(dataframe) < 72:
            return None

        last_candle = dataframe.iloc[-1]
        ema5_1d = last_candle.get('ema5_1d')
        ema200_1d = last_candle.get('ema200_1d')

        if ema5_1d is None or ema200_1d is None:
            return None

        is_bull = ema5_1d > ema200_1d
        threshold = self.rsi_bull.value if is_bull else self.rsi_bear.value
        rsi = last_candle['rsi']
        last_fill = trade.date_last_filled_utc

        # ── RSI path: cooldown 6h dall'ultimo fill (qualsiasi) ──
        if current_time - last_fill >= timedelta(hours=self.cooldown_hours.value):
            if rsi <= threshold:
                return self.dca_stake_amount.value

        # ── BB fallback: solo in bear, cooldown 72h, no RSI negli ultimi 3gg ──
        if not is_bull:
            if current_time - last_fill >= timedelta(hours=72):
                rsi_last_3d = dataframe['rsi'].iloc[-72:]
                if not (rsi_last_3d <= self.rsi_bear.value).any():
                    bb_lower = last_candle.get('bb_lower_4h')
                    if bb_lower and last_candle['close'] <= bb_lower:
                        return self.dca_stake_amount.value

        return None
