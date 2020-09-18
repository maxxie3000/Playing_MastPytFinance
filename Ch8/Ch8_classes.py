# Writing an event-driven broker class

from abc import abstractmethod

class Broker(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.__price_event_handler = None
        self.__order_event_handler = None
        self.__position_event_handler = None

    @property
    def on_price_event(self):
        """
        Listerners will receive: symbol, bid, ask
        """
        return self.__price_event_handler

    @on_price_event.setter
    def on_price_event(self, event_handler):
        self.__price_event_handler = event_handler

    @property
    def on_order_event(self):
        """
        listeners will receive: transaction_id
        """
        return self.__order_event_handler

    @on_order_event.setter
    def on_order_event(self, event_handler):
        self.__order_event_handler = event_handler

    @property
    def on_position_event(self):
        """
        listeners will receive:
        symbol, is_long, units, unrealized_pnl, pnl
        """
        return self.__position_event_handler

    @on_position_event.setter
    def on_position_event(self, position_handler):
        self.__position_event_handler = position_handler

    @abstractmethod
    def get_prices(self, symbols=[]):
        """
        query market prices from a Broker
        :param symbols: list of symbols recognized by your Broker
        """
        raise NotImplementedError('Method is required!')

    @abstractmethod
    def stream_prices(self, symbols=[]):
        """
        Continuously stream prices from a broker
        :param symbols: list of symbols recognized by your Broker
        """
        raise NotImplementedError('Method is required!')

    @abstractmethod
    def send_market_order(self, symbol, quantity, is_buy):
        raise NotImplementedError('Method is required!')

# Specifics for OANDA Broker
import v20

class OandaBroker(Broker):
    practice_api_host = 'api-fxpractice.oanda.com'
    practice_stream_host = 'stream-fxpractice.oanda.com'

    live_api_host = 'api-fxtrade.oanda.com'
    live_stream_host = 'stream-fxtrade.oanda.com'

    port = '443'

    def __init__(self, accountid, token, is_live=False):
        if is_live:
            host = self.live_api_host
            stream_host = self.live_stream_host
        else:
            host = self.practice_api_host
            stream_host = self.practice_stream_host

        super(OandaBroker, self).__init__(host, self.port)

        self.accountid = accountid
        self.token = token

        self.api = v20.Context(host, self.port, token=token)
        self.stream_api = v20.Context(stream_host, self.port, token=token)

    def get_prices(self, symbols = []):
        response = self.api.pricing.get(
            self.accountid,
            instruments=",".join(symbols),
            snapshots=True,
            includeUnitsAvailable=False
        )
        body = response.body
        prices = body.get('prices', [])
        for price in prices:
            self.process_price(price)

    def process_price(self, price):
        symbol = price.instrument

        if not symbol:
            print('Price symbol is empty!')
            return

        bids = price.bids or []
        price_bucket_bid = bids[0] if bids and len(bids) > 0 else None
        bid = price_bucket_bid.price if price_bucket_bid else 0

        asks = price.asks or []
        price_bucket_ask = asks[0] if asks and len(asks) > 0 else None
        ask = price_bucket_ask.price if price_bucket_ask else 0

        self.on_price_event(symbol, bid, ask)

    def stream_prices(self, symbols=[]):
        response = self.stream_api.pricing.stream(
            self.accountid,
            instruments=','.join(symbols),
            snapshot=True
        )

        for msg_type, msg in response.parts():
            if msg_type == 'pricing.Heartbeat':
                continue
            elif msg_type == 'pricing.ClientPrice':
                self.process_price(msg)

    def send_market_order(self, symbol, quantity, is_buy):
        response = self.api.order.market(
            self.accountid,
            units=abs(quantity) * (1 if is_buy else -1),
            instrument=symbol,
            type='MARKET'
        )
        if response.status != 201:
            self.on_order_event(symbol, quantity, is_buy, None, 'NOT_FILLED')
            return

        body = response.body
        if 'orderCancelTransaction' in body:
            self.on_order_event(symbol, quantity, is_buy, None, 'NOT_FILLED')
            return

        transaction_id = body.get('LastTransactionID', None)
        self.on_order_event(symbol, quantity, is_buy, transaction_id, 'FILLED')

    # Fetch all the available position information
    def get_positions(self):
        response = self.api.position.list(self.accountid)
        body = response.body
        positions = body.get('positions', [])
        for position in positions:
            symbol = position.instrument
            unrealized_pnl = position.unrealizedPL
            pnl = position.pl
            long = position.long
            short = position.short

            if short.units:
                self.on_position_event(
                    symbol, False, short.units, unrealized_pnl, pnl)
            elif long.units:
            	self.on_position_event(
                    symbol, True, long.units, unrealized_pnl, pnl)
            else:
                self.on_position_event(
                    symbol, None, 0, unrealized_pnl, pnl)

account_id = '101-004-16480823-001'
api_token = 'b33fef44363f65440eca9d5ffa2c1c9c-8f6cf3bad5a3d6f6f4b40ccd63f7b3d0'

broker = OandaBroker(account_id, api_token)

import datetime as dt

symbol = 'EUR_USD'

def on_price_event(symbol, bid, ask):
    print(
        dt.datetime.now(), '[PRICE]', symbol,
        'bid: ', bid, 'ask: ', ask
    )

broker.on_price_event = on_price_event
# broker.get_prices(symbols=[symbol])

def on_order_event(symbol, quantity, is_buy, transaction_id, status):
    print('check')
    print(f"""{dt.datetime.now()}, [ORDER], transaction_id: {transaction_id},
            status: {status}, symbol: {symbol}, quantity: {quantity}, is_buy: {is_buy}
        """)

broker.on_order_event = on_order_event
# broker.send_market_order(symbol, 1, True)

def on_position_event(symbol, is_long, units, upnl, pnl):
    print(
        f"""{dt.datetime.now()}, [POSITION], symbol: {symbol}, is_long: {is_long},
        units: {units}, upnl: {upnl}, pnl: {pnl}"""
    )

broker.on_position_event = on_position_event
#broker.get_positions()

# Building a mean-revering algorithmic trading system
import time
import datetime as dt
import pandas as pd

class MeanReversionTrader(object):
    def __init__(self, broker, symbol = None, units=1, resample_interval='60s',
        mean_periods=5):
        """
        A trading platform that trades on one side based on a
        mean-reverting algorithm.

        :param broker: Broker object
        :param symbol: A str object recognized by the broker for trading
        :param units: Number of units to trade
        :param resample_interval: Frequency for resampling price time series
        :param mean_periods: Numer of resampled intervals for
                    calculating the average price
        """
        self.broker = self.setup_broker(broker)

        self.resample_interval = resample_interval
        self.mean_periods = mean_periods
        self.symbol = symbol
        self.units = units

        self.df_prices = pd.DataFrame(columns=[symbol])
        self.pnl, self.upnl = 0, 0

        self.mean = 0
        self.bid_price, self.ask_price = 0, 0
        self.position = 0
        self.is_order_pending = False
        self.is_next_signal_cycle = False

    def setup_broker(self, broker):
         broker.on_price_event = self.on_price_event
         broker.on_order_event = self.on_order_event
         broker.on_position_event = self.on_position_event
         return broker

    # Simply assigning three class methods as listeners on aby broker-generated event
    def on_price_event(self, symbol, bid, ask):
        print(f"{dt.datetime.now()}, [PRICE] {symbol}, bid: {bid}, ask: {ask}")
        self.bid_price = bid
        self.ask_price = ask
        self.df_prices.loc[pd.Timestamp.now(), symbol] = (bid + ask) / 2

        self.df_resampled = self.df_prices\
                    .resample(self.resample_interval)\
                    .ffill()\
                    .dropna()
        self.resampled_len = len(self.df_resampled.index)

        if self.resampled_len > self.mean_periods:
            self.df_resampled = self.df_resampled.iloc[self.mean_periods - 1 : ]

        self.get_positions()
        self.generate_signals_and_think()

        self.print_state()

    def get_positions(self):
        try:
            self.broker.get_positions()
        except Exception as ex:
            print('get_positions error: ', ex)

    @property
    def position_state(self):
            if self.position == 0:
                return 'FLAT'
            elif self.position > 0:
                return 'LONG'
            elif self.position < 0:
                return 'SHORT'

    def on_order_event(self, symbol, quantity, is_buy, transaction_id, status):
        print(f'{dt.datetime.now()}, [ORDER], transaction_id: {transaction_id},',
            f'status: {status}, symbol: {symbol}, quantity: {quantity},',
            f'is_buy: {is_buy}')
        if status == 'FILLED':
            self.is_order_pending = False
            self.is_next_signal_cycle = False

            self.get_positions() # Update positions before thinking
            self.generate_signals_and_think()

    def on_position_event(self, symbol, is_long, units, upnl, pnl):
        if symbol == self.symbol:
            self.position = abs(units) * (1 if is_long else -1)
            self.pnl = pnl
            self.upnl = upnl
            self.print_state()

    def print_state(self):
        print(f'{dt.datetime.now()} {self.symbol}, {self.position_state},',
            f'{abs(self.position)}, upnl: {self.upnl}, pnl: {self.pnl}')

    def generate_signals_and_think(self):
        # df_resampled = self.df_prices\
            # .resample(self.resample_interval)\
            # .ffill()\
            # .dropna()
        # resampled_len = len(df_resampled.index)

        if self.resampled_len < self.mean_periods:
            print(
              f'Insufficient data size to calculate logic. Need',
                f'{self.mean_periods - self.resampled_len} more.'
                )
            return

        mean = self.df_resampled.tail(self.mean_periods).mean()[self.symbol]
        std = self.df_resampled.tail(self.mean_periods).std()[self.symbol]

        # Signal flag calculation
        is_signal_buy = mean-(0.3*std) > self.ask_price
        is_signal_sell = mean+(0.3*std) < self.bid_price

        print(
            'is_signal_buy: ', is_signal_buy,
            '\nis_signal_sell: ', is_signal_sell,
            f'\naverage_price: {mean:.5f}',
            'bid: ', self.bid_price,
            'ask: ', self.ask_price
        )

        self.think(is_signal_buy, is_signal_sell)

    def think(self, is_signal_buy, is_signal_sell):
        if self.is_order_pending:
            return

        if self.position == 0:
            self.think_when_flat_position(is_signal_buy, is_signal_sell)
        elif self.position > 0:
            self.think_when_position_long(is_signal_sell)
        elif self.position < 0:
            self.think_when_position_short(is_signal_buy)

    def think_when_flat_position(self, is_signal_buy, is_signal_sell):
        if is_signal_buy and self.is_next_signal_cycle:
            print(f'Opening position, BUY, {self.symbol}, {self.units} units')
            self.is_order_pending = True
            self.send_market_order(self.symbol, self.units, True)
            return
        if is_signal_sell and self.is_next_signal_cycle:
            print(f'Opening position, SELL, {self.symbol}, {self.units} units')
            self.is_order_pending = True
            self.send_market_order(self.symbol, self.units, False)
            return

        if not is_signal_buy and not is_signal_sell:
            self.is_next_signal_cycle = True

    def think_when_position_long(self, is_signal_sell):
        if is_signal_sell:
            print(f'Opening position, SELL, {self.symbol}, {self.units} units')
            self.is_order_pending = True
            self.send_market_order(self.symbol, self.units, False)
            return

    def think_when_position_short(self, is_signal_buy):
        if is_signal_buy:
            print(f'Opening position, BUY, {self.symbol}, {self.units} units')
            self.is_order_pending = True
            self.send_market_order(self.symbol, self.units, True)
            return

    def send_market_order(self, symbol, quantity, is_buy):
        self.broker.send_market_order(symbol, quantity, is_buy)

    def run(self):
        self.get_positions()
        self.broker.stream_prices(symbols=[self.symbol])

if __name__ == "__main__":
    trader = MeanReversionTrader(broker, symbol='EUR_USD', units=5000,
                        resample_interval='60s', mean_periods=5)
    trader.run()
