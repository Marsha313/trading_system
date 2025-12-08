import requests
import time
import hmac
import hashlib
import urllib.parse
import math
from typing import Dict, List, Optional, Tuple
import json
import threading
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from enum import Enum
import logging
import sys
from datetime import datetime
import argparse
import yaml

# 设置日志
def setup_logging(config_name="default", log_filename=None, log_level="INFO"):
    """设置日志配置"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    if log_filename is None:
        log_filename = f"logs/market_maker_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        if not log_filename.startswith('logs/'):
            log_filename = f"logs/{log_filename}"
        if not log_filename.endswith('.log'):
            log_filename += '.log'
    
    # 设置日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    level = level_map.get(log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 日志文件: {log_filename}")
    
    return logger

logger = setup_logging()

class TradingStrategy(Enum):
    MARKET_ONLY = "market_only"
    LIMIT_MARKET = "limit_market"
    BOTH = "both"
    LIMIT_BOTH = "limit_both"
    AUTO = "auto"

@dataclass
class OrderBook:
    bids: List[List[float]]
    asks: List[List[float]]
    update_time: float

@dataclass
class AccountBalance:
    free: float
    locked: float

@dataclass
class StrategyPerformance:
    strategy: TradingStrategy
    success_count: int = 0
    total_count: int = 0
    avg_execution_time: float = 0.0
    total_volume: float = 0.0
    last_execution_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100
    
    @property
    def avg_volume_per_trade(self) -> float:
        if self.success_count == 0:
            return 0.0
        return self.total_volume / self.success_count

@dataclass
class TradingPairConfig:
    symbol: str
    base_asset: str
    quote_asset: str = 'USDT'
    fixed_buy_quantity: float = 10
    target_volume: float = 1000
    max_spread: float = 0.002
    max_price_change: float = 0.005
    min_depth_multiplier: float = 2
    strategy: TradingStrategy = TradingStrategy.LIMIT_BOTH
    min_price_increment: float = 0.0001
    min_5min_volume: float = 0.0
    
    # 新增：价格相关配置
    price_gap_threshold: int = 8  # 价差阈值（以tick为单位）
    stability_period: int = 5  # 价格稳定性检测周期（秒）
    wait_time: int = 30  # 交易完成后的等待时间（秒）
    sampling_rate: float = 0.1  # 市场数据采样频率（秒）
    max_price_adjustment: int = 5  # 最大价格调整tick数
    daily_volume_target: float = 4000  # 每日目标成交量（USDT）
    
    # 交易所限制参数
    tick_size: float = 0.00001  # 最小价格变动单位
    step_size: float = 1  # 最小数量变动单位
    min_notional: float = 5.0  # 最小订单金额（USDT）
    price_precision: int = 4  # 价格精度（小数点位数）
    quantity_precision: int = 1  # 数量精度（小数点位数）
    
    # 限价单偏移配置
    bid_offset_ticks: int = 0
    ask_offset_ticks: int = 0
    dynamic_offset_enabled: bool = True
    min_offset_ticks: int = 0
    max_offset_ticks: int = 20
    spread_threshold_for_offset: float = 0.001
    depth_weight_factor: float = 0.5
    volatility_weight_factor: float = 0.3

@dataclass
class HistoricalVolume:
    account1_volume: float = 0.0
    account2_volume: float = 0.0
    account1_trade_count: int = 0
    account2_trade_count: int = 0

@dataclass
class AccountConfig:
    api_key: str
    secret_key: str
    name: str

@dataclass
class MasterConfig:
    trading_config_path: str
    account1: AccountConfig
    account2: AccountConfig

class AsterDexClient:
    def __init__(self, api_key: str, secret_key: str, account_name: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.account_name = account_name
        self.base_url = 'https://sapi.asterdex.com'  # 硬编码基础URL
        self._balance_cache = None
        self.logger = logging.getLogger(f"{__name__}.{account_name}")
        
    def _sign_request(self, params: Dict) -> str:
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        url = f"{self.base_url}{endpoint}"
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params['signature'] = self._sign_request(params)
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, data=params, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, data=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API请求错误 ({self.account_name}): {e}")
            if str(e).find('Too Many Requests') != -1:
                self.logger.error("请求过多，可能被限流,等待30s")
                time.sleep(30)
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"错误响应: {e.response.text}")
            return {'error': str(e),'text': getattr(e.response, 'text', '')}
    
    def create_order(self, symbol: str, side: str, order_type: str, 
                    quantity: float, min_price_increment:float, step_size: float = 1.0, price: Optional[float] = None) -> Dict:
        """创建订单 - 使用服务器生成的订单ID"""
        endpoint = "/api/v1/order"
        
        # 根据step_size格式化数量
        if step_size <= 0:
            step_size = 1.0  # 默认值
            
        # 计算格式化后的数量
        if step_size < 1:
            # 小数步长，例如0.01
            formatted_quantity = math.floor(quantity / step_size) * step_size
            # 保留小数点后位数
            decimals = len(str(step_size).split('.')[1]) if '.' in str(step_size) else 0
            formatted_quantity = round(formatted_quantity, decimals)
        else:
            # 整数步长
            formatted_quantity = math.floor(quantity / step_size) * step_size
        
        # 确保数量大于0
        if formatted_quantity <= 0:
            self.logger.error(f"计算后的数量无效: {formatted_quantity}, 原始数量: {quantity}, step_size: {step_size}")
            return {'error': f'Invalid quantity: {formatted_quantity}'}
        
        formatted_price = None
        if price is not None and order_type != 'MARKET':
            num_length = 0
            change = min_price_increment
            while change < 1:
                num_length += 1
                change = change * 10
            formatted_price = round(price, num_length)
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': formatted_quantity
        }
        
        if formatted_price is not None:
            params['price'] = formatted_price
            params['timeInForce'] = 'GTC'
        
        self.logger.info(f"📤 发送订单请求:")
        self.logger.info(f"   交易对: {symbol}")
        self.logger.info(f"   方向: {side}")
        self.logger.info(f"   类型: {order_type}")
        self.logger.info(f"   数量: {quantity} -> {formatted_quantity} (step_size: {step_size})")
        if formatted_price:
            self.logger.info(f"   价格: {price} -> {formatted_price}")
        
        return self._request('POST', endpoint, params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """取消订单 - 使用服务器订单ID，如果取消失败则当作订单已成交"""
        endpoint = "/api/v1/order"
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
            
        result = self._request('DELETE', endpoint, params, signed=True)
        
        # 如果取消失败，检查是否是订单已成交的情况
        if 'error' in result or 'code' in result:
            error_msg = str(result.get('error', result.get('msg', 'Unknown error'))) + str(result.get('text', ''))
            
            # 如果错误信息表明订单不存在或已成交，当作订单已成交处理
            if any(keyword in error_msg for keyword in ['does not exist', 'not found', 'already filled', 'filled']):
                self.logger.info(f"⚠️ 取消订单失败，订单可能已成交: {error_msg}")
                # 返回一个模拟的成功响应，表示订单已成交
                return {'orderId': order_id, 'status': 'FILLED'}
            else:
                self.logger.error(f"❌ 取消订单失败: {error_msg}")
        
        return result
    
    def get_order(self, symbol: str, order_id: int) -> Dict:
        """查询订单状态 - 使用服务器订单ID"""
        endpoint = "/api/v1/order"
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
            
        return self._request('GET', endpoint, params, signed=True)
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """获取当前挂单"""
        endpoint = "/api/v1/openOrders"
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        data = self._request('GET', endpoint, params, signed=True)
        
        if isinstance(data, list):
            return data
        else:
            self.logger.error(f"获取挂单失败: {data}")
            return []

    def cancel_all_orders(self, symbol: str = None) -> bool:
        """取消指定交易对的所有挂单"""
        try:
            open_orders = self.get_open_orders(symbol)
            if not open_orders:
                self.logger.info(f"✅ {self.account_name} 没有需要取消的挂单")
                return True
            
            self.logger.info(f"🔄 {self.account_name} 开始取消 {len(open_orders)} 个挂单")
            success_count = 0
            
            for order in open_orders:
                order_id = order.get('orderId')
                order_symbol = order.get('symbol')
                
                try:
                    cancel_result = self.cancel_order(order_symbol, order_id)
                    
                    if 'orderId' in cancel_result:
                        success_count += 1
                        self.logger.info(f"✅ 取消挂单成功: {order_symbol} - {order_id}")
                    else:
                        self.logger.error(f"❌ 取消挂单失败: {order_symbol} - {order_id}: {cancel_result}")
                        
                except Exception as e:
                    self.logger.error(f"❌ 取消挂单异常: {order_symbol} - {order_id}: {e}")
            
            self.logger.info(f"📊 {self.account_name} 取消挂单完成: 成功 {success_count}/{len(open_orders)}")
            return success_count == len(open_orders)
            
        except Exception as e:
            self.logger.error(f"❌ 取消所有挂单时出错: {e}")
            return False
    
    def get_order_book(self, symbol: str, limit: int = 10) -> OrderBook:
        """获取订单簿"""
        endpoint = "/api/v1/depth"
        params = {
            'symbol': symbol,
            'limit': limit
        }
        data = self._request('GET', endpoint, params)
        
        if not data or 'bids' not in data:
            return OrderBook(bids=[], asks=[], update_time=time.time())
            
        bids = [[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])]
        asks = [[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])]
        
        return OrderBook(bids=bids, asks=asks, update_time=time.time())
    
    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 5) -> List[List]:
        """获取K线数据"""
        endpoint = "/api/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        data = self._request('GET', endpoint, params)
        
        if isinstance(data, list):
            return data
        else:
            self.logger.error(f"获取K线数据失败: {data}")
            return []
    
    def get_account_balance(self, force_refresh: bool = False) -> Dict[str, AccountBalance]:
        """获取账户余额"""
        if self._balance_cache is not None and not force_refresh:
            return self._balance_cache
        
        endpoint = "/api/v1/account"
        data = self._request('GET', endpoint, signed=True)
        
        balances = {}
        if 'balances' in data:
            for balance in data['balances']:
                asset = balance['asset']
                balances[asset] = AccountBalance(
                    free=float(balance.get('free', 0)),
                    locked=float(balance.get('locked', 0))
                )
        
        self._balance_cache = balances
        return balances
    
    def get_asset_balance(self, asset: str, force_refresh: bool = False) -> float:
        """获取指定资产的可用余额"""
        balances = self.get_account_balance(force_refresh)
        if asset in balances:
            return balances[asset].free + balances[asset].locked
        return 0.0
    
    def refresh_balance_cache(self):
        """强制刷新余额缓存"""
        self._balance_cache = None
        return self.get_account_balance(force_refresh=True)
    
    def get_all_user_trades(self, symbol: str, start_time: int = None, end_time: int = None) -> List[Dict]:
        """获取所有账户成交历史"""
        all_trades = []
        limit = 1000
        from_id = 1
        max_attempts = 1000
        attempt_count = 0
        
        self.logger.info(f"开始获取 {symbol} 的所有成交历史，从ID=1开始...")
        
        while attempt_count < max_attempts:
            attempt_count += 1
            try:
                params = {
                    'symbol': symbol,
                    'limit': limit,
                    'fromId': from_id
                }
                
                if start_time:
                    params['startTime'] = start_time
                if end_time:
                    params['endTime'] = end_time
                
                endpoint = "/api/v1/userTrades"
                data = self._request('GET', endpoint, params, signed=True)
                
                if not isinstance(data, list):
                    self.logger.error(f"获取成交历史失败: {data}")
                    break
                
                if not data:
                    self.logger.info("没有更多成交记录了")
                    break
                
                filtered_trades = [trade for trade in data if trade.get('symbol') == symbol]
                
                if not filtered_trades:
                    self.logger.info("没有找到指定交易对的成交记录")
                    break
                
                all_trades.extend(filtered_trades)
                
                if len(data) < limit:
                    self.logger.info("已获取所有成交记录")
                    break
                
                max_trade_id = max(int(trade['id']) for trade in filtered_trades)
                from_id = max_trade_id + 1
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"获取成交历史时出错: {e}")
                break
        
        if attempt_count >= max_attempts:
            self.logger.warning(f"达到最大尝试次数 {max_attempts}，停止获取")
        
        self.logger.info(f"总共获取到 {len(all_trades)} 条 {symbol} 的成交记录")
        return all_trades

class SmartMarketMaker:
    def __init__(self, account_config_file: str = "account.yaml", log_filename: str = None):
        self.account_config_file = account_config_file
        
        # 加载账户配置
        if os.path.exists(account_config_file):
            with open(account_config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            self.master_config = MasterConfig(
                trading_config_path=config_data['trading_config_path'],
                account1=AccountConfig(
                    api_key=config_data['account1']['api_key'],
                    secret_key=config_data['account1']['secret_key'],
                    name=config_data['account1']['name']
                ),
                account2=AccountConfig(
                    api_key=config_data['account2']['api_key'],
                    secret_key=config_data['account2']['secret_key'],
                    name=config_data['account2']['name']
                )
            )
            
            # 加载交易配置
            if os.path.exists(self.master_config.trading_config_path):
                with open(self.master_config.trading_config_path, 'r', encoding='utf-8') as f:
                    trading_config_data = yaml.safe_load(f)
                
                # 获取交易对名称
                symbol = trading_config_data.get('symbol', 'GUAUSDT')
                base_asset = symbol.replace('USDT', '')
                
                # 获取策略
                strategy_str = trading_config_data.get('strategy', 'AUTO').upper()
                strategy = getattr(TradingStrategy, strategy_str, TradingStrategy.AUTO)
                
                # 创建交易对配置
                pair_config = TradingPairConfig(
                    symbol=symbol,
                    base_asset=base_asset,
                    fixed_buy_quantity=float(trading_config_data.get('base_quantity', 200)),
                    target_volume=float(trading_config_data.get('daily_volume_target', 4000)),
                    max_spread=float(trading_config_data.get('max_spread', 0.002)),
                    strategy=strategy,
                    min_price_increment=float(trading_config_data.get('tick_size', 0.00001)),
                    min_5min_volume=float(trading_config_data.get('min_5min_volume', 0.0)),
                    
                    # 价格相关配置
                    price_gap_threshold=int(trading_config_data.get('price_gap_threshold', 8)),
                    stability_period=int(trading_config_data.get('stability_period', 5)),
                    wait_time=int(trading_config_data.get('wait_time', 30)),
                    sampling_rate=float(trading_config_data.get('sampling_rate', 0.1)),
                    max_price_adjustment=int(trading_config_data.get('max_price_adjustment', 5)),
                    daily_volume_target=float(trading_config_data.get('daily_volume_target', 4000)),
                    
                    # 交易所限制参数
                    tick_size=float(trading_config_data.get('tick_size', 0.00001)),
                    step_size=float(trading_config_data.get('step_size', 1)),
                    min_notional=float(trading_config_data.get('min_notional', 5.0)),
                    price_precision=int(trading_config_data.get('price_precision', 4)),
                    quantity_precision=int(trading_config_data.get('quantity_precision', 1)),
                    
                    # 限价单偏移配置
                    bid_offset_ticks=int(trading_config_data.get('bid_offset_ticks', 0)),
                    ask_offset_ticks=int(trading_config_data.get('ask_offset_ticks', 0)),
                    dynamic_offset_enabled=trading_config_data.get('dynamic_offset_enabled', True),
                    min_offset_ticks=int(trading_config_data.get('min_offset_ticks', 0)),
                    max_offset_ticks=int(trading_config_data.get('max_offset_ticks', 20)),
                    spread_threshold_for_offset=float(trading_config_data.get('spread_threshold_for_offset', 0.001)),
                    depth_weight_factor=float(trading_config_data.get('depth_weight_factor', 0.5)),
                    volatility_weight_factor=float(trading_config_data.get('volatility_weight_factor', 0.3))
                )
                
                self.trading_pairs = [pair_config]
                config_name = os.path.splitext(os.path.basename(account_config_file))[0]
                
                # 获取日志级别
                log_level = trading_config_data.get('log_level', 'INFO')
                
                # 设置日志
                self.logger = setup_logging(config_name, log_filename, log_level)
                self.logger.info(f"📁 使用账户配置文件: {account_config_file}")
                self.logger.info(f"📁 使用交易配置文件: {self.master_config.trading_config_path}")
            else:
                raise FileNotFoundError(f"交易配置文件 {self.master_config.trading_config_path} 不存在")
        else:
            raise FileNotFoundError(f"账户配置文件 {account_config_file} 不存在")
        
        # Aster配置（硬编码，如果需要可以移到配置文件）
        self.aster_asset = 'ASTER'
        self.aster_symbol = 'ASTERUSDT'
        self.min_aster_balance = 10.0
        self.aster_buy_quantity = 5.0
        self.aster_order_timeout = 10.0
        
        # 其他配置
        self.check_interval = 1.0
        self.max_retry = 3
        self.order_timeout = 10.0
        
        self.client1 = AsterDexClient(
            self.master_config.account1.api_key,
            self.master_config.account1.secret_key,
            self.master_config.account1.name
        )
        self.client2 = AsterDexClient(
            self.master_config.account2.api_key,
            self.master_config.account2.secret_key,
            self.master_config.account2.name
        )
        
        self.current_pair_index = 0
        
        self.total_volume = 0
        self.is_running = False
        
        self.pair_states = {}
        self.historical_volumes = {}
        self.strategy_performance = {}
        
        for pair in self.trading_pairs:
            self.pair_states[pair.symbol] = {
                'order_book': OrderBook(bids=[], asks=[], update_time=0),
                'last_prices': [],
                'price_history_size': 10,
                'trade_count': 0,
                'successful_trades': 0,
                'limit_sell_success_count': 0,
                'market_sell_success_count': 0,
                'limit_sell_attempt_count': 0,
                'partial_limit_sell_count': 0,
                'limit_both_success_count': 0,
                'limit_market_success_count': 0,
                'volume': 0,
                'current_strategy': pair.strategy,
                'limit_buy_attempt_count': 0,
                'limit_buy_success_count': 0,
                'partial_limit_buy_count': 0,
                'market_buy_success_count': 0,
                'last_volume_check': 0,
                'current_5min_volume': 0.0,
                'recent_execution_prices': []
            }
            
            self.historical_volumes[pair.symbol] = HistoricalVolume()
            self.strategy_performance[pair.symbol] = {
                TradingStrategy.LIMIT_BOTH: StrategyPerformance(TradingStrategy.LIMIT_BOTH),
                TradingStrategy.MARKET_ONLY: StrategyPerformance(TradingStrategy.MARKET_ONLY),
                TradingStrategy.LIMIT_MARKET: StrategyPerformance(TradingStrategy.LIMIT_MARKET)
            }
        
        self.aster_buy_attempts = 0
        self.aster_buy_success = 0
        self.aster_buy_failed = 0

    def get_current_trading_pair(self) -> TradingPairConfig:
        """获取当前交易对"""
        return self.trading_pairs[self.current_pair_index]

    def switch_to_next_pair(self):
        """切换到下一个交易对"""
        self.current_pair_index = (self.current_pair_index + 1) % len(self.trading_pairs)
        current_pair = self.get_current_trading_pair()
        self.logger.info(f"🔄 切换到交易对: {current_pair.symbol} (策略: {current_pair.strategy.value})")
        if self.current_pair_index == 0:
            self.logger.info("🔁 已循环回到第一个交易对, 等待1s")
            time.sleep(1)

    def cancel_all_open_orders_before_start(self):
        """启动前取消所有相关交易对的挂单"""
        self.logger.info("🔄 开始取消所有相关交易对的挂单...")
        
        symbols = [pair.symbol for pair in self.trading_pairs]
        self.logger.info(f"📋 需要清理的交易对: {', '.join(symbols)}")
        
        success1 = True
        success2 = True
        
        for symbol in symbols:
            self.logger.info(f"🔄 清理交易对 {symbol} 的挂单...")
            success1 = success1 and self.client1.cancel_all_orders(symbol)
            success2 = success2 and self.client2.cancel_all_orders(symbol)
        
        if success1 and success2:
            self.logger.info("✅ 所有挂单清理完成")
        else:
            self.logger.warning("⚠️ 部分挂单清理可能失败，但程序将继续运行")
        
        time.sleep(2)

    def get_5min_volume_from_klines(self, pair: TradingPairConfig) -> float:
        """通过K线数据获取指定交易对最近5分钟的总成交量"""
        try:
            # 获取最近5根1分钟K线数据
            klines_data = self.client1.get_klines(pair.symbol, interval='1m', limit=5)
            
            if not klines_data:
                self.logger.warning(f"无法获取 {pair.symbol} 的K线数据")
                return 0.0
            
            total_volume = 0.0
            for kline in klines_data:
                # K线数据格式: [开盘时间, 开盘价, 最高价, 最低价, 收盘价, 成交量, ...]
                # 索引5是成交量
                volume = float(kline[5])
                total_volume += volume
            
            self.logger.debug(f"{pair.symbol} 最近5分钟K线成交量: {total_volume:.2f}")
            return total_volume
            
        except Exception as e:
            self.logger.error(f"获取 {pair.symbol} K线数据时出错: {e}")
            return 0.0

    def update_volume_data(self, pair: TradingPairConfig):
        """更新成交量数据"""
        current_time = time.time()
        state = self.pair_states[pair.symbol]
        
        # 每分钟更新一次成交量数据，避免频繁调用API
        if current_time - state.get('last_volume_check', 0) >= 60:
            try:
                new_volume = self.get_5min_volume_from_klines(pair)
                state['current_5min_volume'] = new_volume
                state['last_volume_check'] = current_time
                
                self.logger.debug(f"{pair.symbol} 5分钟成交量已更新: {new_volume:.2f}")
                
            except Exception as e:
                self.logger.error(f"更新 {pair.symbol} 成交量数据时出错: {e}")
                # 如果更新失败，使用上一次的值

    def check_volume_requirement(self, pair: TradingPairConfig) -> bool:
        """检查成交量要求是否满足"""
        if pair.min_5min_volume <= 0:
            return True  # 如果没有设置最小成交量要求，则直接返回True
        
        current_volume = self.pair_states[pair.symbol].get('current_5min_volume', 0.0)
        
        if current_volume >= pair.min_5min_volume:
            self.logger.info(f"✅ {pair.symbol} 成交量要求满足: {current_volume:.2f} >= {pair.min_5min_volume:.2f}")
            return True
        else:
            self.logger.info(f"⏳ {pair.symbol} 成交量不足: {current_volume:.2f} < {pair.min_5min_volume:.2f}")
            return False

    def check_and_buy_aster_if_needed(self) -> bool:
        """检查并购买Aster代币（如果需要）"""
        self.logger.info("🔍 检查Aster代币余额...")
        
        aster_balance1 = self.client1.get_asset_balance(self.aster_asset)
        aster_balance2 = self.client2.get_asset_balance(self.aster_asset)
        
        self.logger.info(f"Aster余额: 账户1={aster_balance1:.4f}, 账户2={aster_balance2:.4f}, 要求={self.min_aster_balance:.4f}")
        
        if aster_balance1 >= self.min_aster_balance and aster_balance2 >= self.min_aster_balance:
            self.logger.info("✅ Aster余额充足，继续对冲交易")
            return True
        
        self.logger.warning("⚠️ Aster余额不足，开始购买Aster代币...")
        
        success_count = 0
        if aster_balance1 < self.min_aster_balance:
            if self.buy_aster_for_account(self.client1, 'ACCOUNT1'):
                success_count += 1
        
        if aster_balance2 < self.min_aster_balance:
            if self.buy_aster_for_account(self.client2, 'ACCOUNT2'):
                success_count += 1
        
        aster_balance1_after = self.client1.get_asset_balance(self.aster_asset, force_refresh=True)
        aster_balance2_after = self.client2.get_asset_balance(self.aster_asset, force_refresh=True)
        
        final_success = (aster_balance1_after >= self.min_aster_balance and 
                        aster_balance2_after >= self.min_aster_balance)
        
        if final_success:
            self.logger.info("✅ Aster购买完成，余额充足，继续对冲交易")
        else:
            self.logger.error("❌ Aster购买失败，余额仍不足，暂停对冲交易")
        
        return final_success

    def buy_aster_for_account(self, client: AsterDexClient, account_name: str) -> bool:
        """为指定账户购买Aster代币"""
        self.logger.info(f"🔄 为{account_name}购买Aster代币...")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            self.aster_buy_attempts += 1
            
            try:
                aster_order_book = client.get_order_book(self.aster_symbol, limit=5)
                if not aster_order_book.bids or not aster_order_book.asks:
                    self.logger.error(f"❌ 无法获取Aster市场价格")
                    continue
                
                best_bid = aster_order_book.bids[0][0]
                best_ask = aster_order_book.asks[0][0]
                
                buy_price = best_bid + 0.0001
                
                usdt_balance = client.get_asset_balance('USDT')
                required_usdt = self.aster_buy_quantity * buy_price
                
                if usdt_balance < required_usdt:
                    self.logger.error(f"❌ {account_name} USDT余额不足: 需要{required_usdt:.2f}, 当前{usdt_balance:.2f}")
                    return False
                
                self.logger.info(f"📤 提交Aster限价买单: {account_name}, 数量={self.aster_buy_quantity}, 价格={buy_price:.6f}")
                
                buy_order = client.create_order(
                    symbol=self.aster_symbol,
                    side='BUY',
                    order_type='LIMIT',
                    quantity=self.aster_buy_quantity,
                    min_price_increment=0.00001,
                    price=buy_price
                )
                
                if 'orderId' not in buy_order:
                    self.logger.error(f"❌ Aster买单失败: {buy_order}")
                    continue
                
                order_id = buy_order['orderId']
                self.logger.info(f"✅ Aster限价买单已提交: {order_id}")
                
                order_filled = self.wait_for_aster_order_completion(client, order_id)
                
                if order_filled:
                    self.aster_buy_success += 1
                    self.logger.info(f"✅ {account_name} Aster购买成功")
                    client.refresh_balance_cache()
                    return True
                else:
                    self.logger.warning(f"⚠️ {account_name} Aster订单未完全成交，取消订单")
                    client.cancel_order(self.aster_symbol, order_id)
                    
                    client.refresh_balance_cache()
                    
                    current_aster_balance = client.get_asset_balance(self.aster_asset)
                    if current_aster_balance >= self.min_aster_balance:
                        self.logger.info(f"✅ {account_name} Aster余额已满足要求（可能有部分成交）")
                        return True
                    
                    if attempt < max_attempts - 1:
                        wait_time = 5
                        self.logger.info(f"等待{wait_time}秒后重试Aster购买...")
                        time.sleep(wait_time)
            
            except Exception as e:
                self.logger.error(f"❌ {account_name} Aster购买过程中出错: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(5)
        
        self.aster_buy_failed += 1
        self.logger.error(f"❌ {account_name} Aster购买失败，已达到最大尝试次数")
        return False

    def wait_for_aster_order_completion(self, client: AsterDexClient, order_id: int) -> bool:
        """等待Aster订单完成"""
        start_time = time.time()
        
        while time.time() - start_time < self.aster_order_timeout:
            try:
                order_status = client.get_order(self.aster_symbol, order_id)
                status = order_status.get('status')
                
                if status == 'FILLED':
                    self.logger.info("✅ Aster订单完全成交")
                    return True
                elif status == 'PARTIALLY_FILLED':
                    executed_qty = float(order_status.get('executedQty', 0))
                    orig_qty = float(order_status.get('origQty', 0))
                    fill_rate = (executed_qty / orig_qty) * 100
                    self.logger.info(f"🔄 Aster订单部分成交: {executed_qty:.4f}/{orig_qty:.4f} ({fill_rate:.1f}%)")
                elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    self.logger.warning(f"⚠️ Aster订单失败: {status}")
                    return False
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"查询Aster订单状态时出错: {e}")
                time.sleep(1)
        
        self.logger.warning("⚠️ Aster订单等待超时")
        return False

    def calculate_historical_volume(self):
        """计算每个交易对的历史现货交易量"""
        self.logger.info("📊 正在计算各交易对的历史交易量...")
        
        for pair in self.trading_pairs:
            self.logger.info(f"计算交易对 {pair.symbol} 的历史交易量...")
            
            historical_volume = self.historical_volumes[pair.symbol]
            
            try:
                trades_account1 = self.client1.get_all_user_trades(symbol=pair.symbol)
                
                for trade in trades_account1:
                    if trade.get('symbol') == pair.symbol:
                        quote_qty = float(trade.get('quoteQty', 0))
                        historical_volume.account1_volume += quote_qty
                        historical_volume.account1_trade_count += 1
                        
                self.logger.info(f"✅ 账户1 {pair.symbol} 历史交易: {historical_volume.account1_trade_count} 笔, 交易量: {historical_volume.account1_volume:.2f} USDT")
                        
            except Exception as e:
                self.logger.error(f"❌ 获取账户1 {pair.symbol} 历史交易量失败: {e}")
            
            try:
                trades_account2 = self.client2.get_all_user_trades(symbol=pair.symbol)
                
                for trade in trades_account2:
                    if trade.get('symbol') == pair.symbol:
                        quote_qty = float(trade.get('quoteQty', 0))
                        historical_volume.account2_volume += quote_qty
                        historical_volume.account2_trade_count += 1
                        
                self.logger.info(f"✅ 账户2 {pair.symbol} 历史交易: {historical_volume.account2_trade_count} 笔, 交易量: {historical_volume.account2_volume:.2f} USDT")
                        
            except Exception as e:
                self.logger.error(f"❌ 获取账户2 {pair.symbol} 历史交易量失败: {e}")
            
            total_volume = historical_volume.account1_volume + historical_volume.account2_volume
            total_trade_count = historical_volume.account1_trade_count + historical_volume.account2_trade_count
            self.logger.info(f"💰 {pair.symbol} 总历史交易: {total_trade_count} 笔, 交易量: {total_volume:.2f} USDT")

    def print_historical_volume_statistics(self):
        """打印各交易对的历史交易量统计"""
        self.logger.info("\n💰 各交易对历史交易量统计:")
        
        for pair in self.trading_pairs:
            historical_volume = self.historical_volumes[pair.symbol]
            total_volume = historical_volume.account1_volume + historical_volume.account2_volume
            total_trade_count = historical_volume.account1_trade_count + historical_volume.account2_trade_count
            
            self.logger.info(f"\n   {pair.symbol}:")
            self.logger.info(f"     账户1: {historical_volume.account1_trade_count} 笔, {historical_volume.account1_volume:.2f} USDT")
            self.logger.info(f"     账户2: {historical_volume.account2_trade_count} 笔, {historical_volume.account2_volume:.2f} USDT")
            self.logger.info(f"     总计: {total_trade_count} 笔, {total_volume:.2f} USDT")
        
        total_all_volume = sum(
            historical_volume.account1_volume + historical_volume.account2_volume 
            for historical_volume in self.historical_volumes.values()
        )
        total_all_trade_count = sum(
            historical_volume.account1_trade_count + historical_volume.account2_trade_count 
            for historical_volume in self.historical_volumes.values()
        )
        
        self.logger.info(f"\n   🌟 所有交易对总计:")
        self.logger.info(f"     总交易笔数: {total_all_trade_count} 笔")
        self.logger.info(f"     总交易量: {total_all_volume:.2f} USDT")

    def initialize_at_balance(self, pair: TradingPairConfig) -> bool:
        """初始化指定交易对的余额"""
        at_balance1 = self.client1.get_asset_balance(pair.base_asset)
        at_balance2 = self.client2.get_asset_balance(pair.base_asset)
        
        self.logger.info(f"检查{pair.base_asset}余额: 账户1={at_balance1:.4f}, 账户2={at_balance2:.4f}")
        
        if at_balance1 >= pair.fixed_buy_quantity/2 and at_balance2 >= pair.fixed_buy_quantity/2:
            self.logger.info(f"✅ 两个账户都有足够的{pair.base_asset}余额，无需初始化")
            return True
        
        if at_balance1 < pair.fixed_buy_quantity/2 and at_balance2 < pair.fixed_buy_quantity/2:
            self.logger.info(f"🔄 两个账户都没有足够的{pair.base_asset}余额，开始初始化...")
            
            usdt_balance1 = self.client1.get_asset_balance('USDT')
            usdt_balance2 = self.client2.get_asset_balance('USDT')
            
            if usdt_balance1 >= usdt_balance2 and usdt_balance1 > 0:
                buy_client = self.client1
                buy_client_name = 'ACCOUNT1'
                available_usdt = usdt_balance1
            elif usdt_balance2 > 0:
                buy_client = self.client2
                buy_client_name = 'ACCOUNT2'
                available_usdt = usdt_balance2
            else:
                self.logger.error(f"❌ 两个账户都没有足够的USDT进行{pair.base_asset}初始化买入")
                return False
            
            bid, ask, _, _ = self.get_best_bid_ask(pair)
            if bid == 0 or ask == 0:
                self.logger.error(f"❌ 无法获取{pair.symbol}市场价格，初始化失败")
                return False
            
            current_price = (bid + ask) / 2
            buy_quantity = min(pair.fixed_buy_quantity, (available_usdt * 0.5) / current_price)
            
            if buy_quantity <= 0:
                self.logger.error(f"❌ 计算出的{pair.base_asset}买入数量为0，初始化失败")
                return False
            
            self.logger.info(f"🎯 选择 {buy_client_name} 进行{pair.base_asset}初始化买入: 数量={buy_quantity:.4f}, 价格≈{current_price:.4f}")
            
            buy_order = buy_client.create_order(
                symbol=pair.symbol,
                side='BUY',
                order_type='MARKET',
                quantity=buy_quantity,
                min_price_increment=pair.min_price_increment
            )
            
            if 'orderId' not in buy_order:
                self.logger.error(f"❌ {pair.base_asset}初始化买入失败: {buy_order}")
                return False
            
            order_id = buy_order['orderId']
            self.logger.info(f"✅ {pair.base_asset}初始化买入订单已提交: {order_id}")
            
            success = self.wait_for_orders_completion([(buy_client, order_id)], pair.symbol)
            
            if success:
                self.logger.info(f"✅ {pair.base_asset}余额初始化成功")
                self.client1.refresh_balance_cache()
                self.client2.refresh_balance_cache()
                return True
            else:
                self.logger.error(f"❌ {pair.base_asset}初始化买入订单未成交")
                return False
        
        self.logger.info(f"✅ {pair.base_asset}余额状态正常，无需初始化")
        return True

    def get_cached_trade_direction(self, pair: TradingPairConfig) -> Tuple[str, str]:
        """获取指定交易对的缓存的交易方向"""
        cache_key = f"{pair.symbol}_trade_direction"
        if not hasattr(self, '_trade_direction_cache'):
            self._trade_direction_cache = {}
        
        if cache_key not in self._trade_direction_cache:
            self._trade_direction_cache[cache_key] = self.determine_trade_direction(pair)
        
        return self._trade_direction_cache[cache_key]

    def update_trade_direction_cache(self, pair: TradingPairConfig):
        """强制更新指定交易对的交易方向缓存"""
        cache_key = f"{pair.symbol}_trade_direction"
        if not hasattr(self, '_trade_direction_cache'):
            self._trade_direction_cache = {}
        
        self._trade_direction_cache[cache_key] = self.determine_trade_direction(pair)

    def determine_trade_direction(self, pair: TradingPairConfig) -> Tuple[str, str]:
        """自动判断指定交易对的交易方向：返回 (sell_client_name, buy_client_name)"""
        at_balance1 = self.client1.get_asset_balance(pair.base_asset)
        at_balance2 = self.client2.get_asset_balance(pair.base_asset)
        
        self.logger.info(f"{pair.base_asset}余额对比: 账户1={at_balance1:.4f}, 账户2={at_balance2:.4f}")
        
        if at_balance1 >= at_balance2:
            self.logger.info(f"🎯 {pair.symbol}选择策略: 账户1卖出，账户2买入")
            return 'ACCOUNT1', 'ACCOUNT2'
        else:
            self.logger.info(f"🎯 {pair.symbol}选择策略: 账户2卖出，账户1买入")
            return 'ACCOUNT2', 'ACCOUNT1'

    def get_current_trade_direction(self, pair: TradingPairConfig) -> Tuple[str, str]:
        """获取指定交易对的当前交易方向（使用缓存）"""
        return self.get_cached_trade_direction(pair)

    def update_order_book(self, pair: TradingPairConfig):
        """更新指定交易对的订单簿数据"""
        try:
            new_order_book = self.client1.get_order_book(pair.symbol, limit=10)
            if new_order_book.bids and new_order_book.asks:
                self.pair_states[pair.symbol]['order_book'] = new_order_book
                
                mid_price = (new_order_book.bids[0][0] + new_order_book.asks[0][0]) / 2
                state = self.pair_states[pair.symbol]
                state['last_prices'].append(mid_price)
                if len(state['last_prices']) > state['price_history_size']:
                    state['last_prices'].pop(0)
                    
        except Exception as e:
            self.logger.error(f"更新{pair.symbol}订单簿时出错: {e}")

    def get_best_bid_ask(self, pair: TradingPairConfig) -> Tuple[float, float, float, float]:
        """获取指定交易对的最优买卖价和深度"""
        order_book = self.pair_states[pair.symbol]['order_book']
        if not order_book.bids or not order_book.asks:
            return 0, 0, 0, 0
            
        best_bid = order_book.bids[0][0]
        best_ask = order_book.asks[0][0]
        bid_quantity = order_book.bids[0][1]
        ask_quantity = order_book.asks[0][1]
        
        return best_bid, best_ask, bid_quantity, ask_quantity

    def calculate_spread_percentage(self, bid: float, ask: float) -> float:
        """计算价差百分比"""
        if bid == 0 or ask == 0:
            return float('inf')
        return (ask - bid) / bid

    def calculate_price_volatility(self, pair: TradingPairConfig) -> float:
        """计算指定交易对的价格波动率"""
        state = self.pair_states[pair.symbol]
        if len(state['last_prices']) < 2:
            return 0
            
        returns = []
        for i in range(1, len(state['last_prices'])):
            if state['last_prices'][i-1] != 0:
                returns.append(abs(state['last_prices'][i] - state['last_prices'][i-1]) / state['last_prices'][i-1])
        
        return max(returns) if returns else 0

    def get_sell_quantity(self, pair: TradingPairConfig, sell_client_name: str = None) -> Tuple[float, str]:
        """获取指定交易对的实际可卖数量和卖出账户（使用缓存余额）"""
        if sell_client_name is None:
            sell_client_name, _ = self.get_current_trade_direction(pair)
        
        if sell_client_name == 'ACCOUNT1':
            available_at = self.client1.get_asset_balance(pair.base_asset)
            sell_account = 'ACCOUNT1'
        else:
            available_at = self.client2.get_asset_balance(pair.base_asset)
            sell_account = 'ACCOUNT2'
        
        return available_at, sell_account

    def check_buy_conditions_with_retry(self, pair: TradingPairConfig, max_retry: int = 3, wait_time: int = 20) -> bool:
        """检查指定交易对的买单条件，余额不足时等待并重试"""
        for attempt in range(max_retry):
            if self.check_buy_conditions(pair):
                return True
            else:
                if attempt < max_retry - 1:
                    self.logger.info(f"{pair.symbol} USDT余额不足，等待{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retry})")
                    
                    self.client1.refresh_balance_cache()
                    self.client2.refresh_balance_cache()
                    self.update_trade_direction_cache(pair)
                    
                    time.sleep(wait_time)
        
        return False

    def check_sell_conditions_with_retry(self, pair: TradingPairConfig, max_retry: int = 3, wait_time: int = 20) -> bool:
        """检查指定交易对的卖单条件，余额不足时等待并重试"""
        for attempt in range(max_retry):
            if self.check_sell_conditions(pair):
                return True
            else:
                if attempt < max_retry - 1:
                    self.logger.info(f"{pair.symbol} {pair.base_asset}余额不足，等待{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retry})")
                    
                    self.client1.refresh_balance_cache()
                    self.client2.refresh_balance_cache()
                    self.update_trade_direction_cache(pair)
                    
                    time.sleep(wait_time)
        
        return False
    
    def check_buy_conditions(self, pair: TradingPairConfig) -> bool:
        """检查指定交易对的买单条件：USDT余额是否足够（使用缓存余额）"""
        _, buy_client_name = self.get_current_trade_direction(pair)
        
        if buy_client_name == 'ACCOUNT1':
            available_usdt = self.client1.get_asset_balance('USDT')
        else:
            available_usdt = self.client2.get_asset_balance('USDT')
        
        bid, ask, _, _ = self.get_best_bid_ask(pair)
        if bid == 0 or ask == 0:
            return False
        
        current_price = (bid + ask) / 2
        required_usdt = pair.fixed_buy_quantity * current_price
        
        if available_usdt >= required_usdt:
            return True
        else:
            self.logger.warning(f"{pair.symbol} USDT余额不足: 需要{required_usdt:.2f}, 当前{available_usdt:.2f}")
            return False
    
    def check_sell_conditions(self, pair: TradingPairConfig) -> bool:
        """检查指定交易对的卖单条件：基础资产余额是否足够（至少要有一些可卖）"""
        sell_quantity, sell_account = self.get_sell_quantity(pair)
        if sell_quantity <= 0:
            self.logger.warning(f"账户 {sell_account} 无可卖{pair.base_asset}数量")
            return False
        return True

    def should_use_limit_strategy(self, pair: TradingPairConfig) -> bool:
        """判断是否应该使用限价策略"""
        bid, ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
        spread = self.calculate_spread_percentage(bid, ask)
        
        high_liquidity = (
            spread < pair.min_price_increment * 10 and
            bid_qty > pair.fixed_buy_quantity * 10 and
            ask_qty > pair.fixed_buy_quantity * 10
        )
        return high_liquidity

    def should_use_market_strategy(self, pair: TradingPairConfig) -> bool:
        """判断是否应该使用市价策略"""
        bid, ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
        spread = self.calculate_spread_percentage(bid, ask)
        
        low_liquidity = (
            spread > pair.min_price_increment * 20 or
            bid_qty < pair.fixed_buy_quantity * 2 or
            ask_qty < pair.fixed_buy_quantity * 2
        )
        return low_liquidity

    def auto_select_strategy_by_market_condition(self, pair: TradingPairConfig) -> TradingStrategy:
        """根据市场条件自动选择策略"""
        bid, ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
        spread = self.calculate_spread_percentage(bid, ask)
        volatility = self.calculate_price_volatility(pair)
        
        market_score = 0
        
        min_spread_threshold = pair.min_price_increment * 5
        if spread < min_spread_threshold:
            market_score += 3
        elif spread < min_spread_threshold * 2:
            market_score += 2
        elif spread < min_spread_threshold * 4:
            market_score += 1
        
        min_depth = min(bid_qty, ask_qty)
        required_depth = pair.fixed_buy_quantity * pair.min_depth_multiplier
        if min_depth > required_depth * 5:
            market_score += 3
        elif min_depth > required_depth * 3:
            market_score += 2
        elif min_depth > required_depth * 1.5:
            market_score += 1
        
        if volatility < 0.001:
            market_score += 3
        elif volatility < 0.003:
            market_score += 2
        elif volatility < 0.005:
            market_score += 1
        
        if market_score >= 7:
            return TradingStrategy.LIMIT_BOTH
        elif market_score >= 4:
            return TradingStrategy.LIMIT_MARKET
        else:
            return TradingStrategy.MARKET_ONLY

    def record_strategy_performance(self, pair: TradingPairConfig, strategy: TradingStrategy, 
                                  success: bool, execution_time: float, volume: float):
        """记录策略执行结果"""
        perf = self.strategy_performance[pair.symbol][strategy]
        perf.total_count += 1
        perf.last_execution_time = execution_time
        
        if success:
            perf.success_count += 1
            perf.total_volume += volume
        
        if perf.total_count == 1:
            perf.avg_execution_time = execution_time
        else:
            perf.avg_execution_time = (perf.avg_execution_time * (perf.total_count - 1) + execution_time) / perf.total_count

    def get_best_strategy(self, pair: TradingPairConfig) -> TradingStrategy:
        """根据历史性能选择最佳策略"""
        performances = self.strategy_performance[pair.symbol]
        
        valid_strategies = {
            strategy: perf for strategy, perf in performances.items() 
            if perf.total_count >= 5
        }
        
        if not valid_strategies:
            return self.auto_select_strategy_by_market_condition(pair)
        
        best_strategy = max(valid_strategies.items(), 
                           key=lambda x: x[1].success_rate)
        
        self.logger.info(f"🎯 {pair.symbol} 最佳策略推荐: {best_strategy[0].value} (成功率: {best_strategy[1].success_rate:.1f}%)")
        return best_strategy[0]

    def check_market_conditions(self, pair: TradingPairConfig) -> Tuple[bool, str]:
        """检查指定交易对的市场条件是否满足交易，返回状态和交易模式"""
        # if not self.check_and_buy_aster_if_needed():
        #     self.logger.error("❌ Aster余额检查失败，暂停交易")
        #     return False, "error"
        
        # 新增：检查成交量要求
        if not self.check_volume_requirement(pair):
            self.logger.info(f"⏳ {pair.symbol} 成交量不足，跳过交易")
            return False, "volume_insufficient"
        
        at_balance1 = self.client1.get_asset_balance(pair.base_asset)
        at_balance2 = self.client2.get_asset_balance(pair.base_asset)
        
        balance_threshold = pair.fixed_buy_quantity / 2
        both_accounts_sufficient = (at_balance1 >= balance_threshold and 
                                at_balance2 >= balance_threshold)
        
        if both_accounts_sufficient:
            self.logger.info(f"✅ 两个账户{pair.base_asset}余额都充足，使用仅卖出模式")
            return True, "sell_only"
        
        if at_balance1 < balance_threshold and at_balance2 < balance_threshold:
            self.logger.warning(f"⚠️ 两个账户都没有足够的{pair.base_asset}余额，尝试初始化...")
            if self.initialize_at_balance(pair):
                self.logger.info(f"✅ {pair.base_asset}余额初始化成功，继续交易")
            else:
                self.logger.error(f"❌ {pair.base_asset}余额初始化失败，暂停交易")
                return False, "error"
        
        if not self.check_sell_conditions_with_retry(pair, max_retry=3, wait_time=20):
            self.logger.error(f"{pair.symbol}卖单条件检查失败，{pair.base_asset}余额持续不足")
            return False, "error"
        
        if not self.check_buy_conditions_with_retry(pair, max_retry=3, wait_time=20):
            self.logger.error(f"{pair.symbol}买单条件检查失败，USDT余额持续不足")
            return False, "error"
        
        bid, ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
        
        if bid == 0 or ask == 0:
            return False, "error"
            
        spread = self.calculate_spread_percentage(bid, ask)
        if spread > pair.max_spread:
            self.logger.warning(f"{pair.symbol}价差过大: {spread:.4%} > {pair.max_spread:.4%}")
            return False, "error"
        
        volatility = self.calculate_price_volatility(pair)
        if volatility > pair.max_price_change:
            self.logger.warning(f"{pair.symbol}价格波动过大: {volatility:.4%} > {pair.max_price_change:.4%}")
            return False, "error"
        
        min_required_depth = pair.fixed_buy_quantity * pair.min_depth_multiplier
        if bid_qty < min_required_depth or ask_qty < min_required_depth:
            self.logger.warning(f"{pair.symbol}深度不足: 买一量={bid_qty:.2f}, 卖一量={ask_qty:.2f}, 要求={min_required_depth:.2f}")
            return False, "error"
            
        sell_quantity, sell_account = self.get_sell_quantity(pair)
        _, buy_account = self.get_current_trade_direction(pair)
        
        self.logger.info(f"✓ {pair.symbol}市场条件满足: 价差={spread:.4%}, 波动={volatility:.4%}")
        self.logger.info(f"  {pair.symbol}交易方向: {sell_account}卖出{sell_quantity:.4f}, {buy_account}买入{pair.fixed_buy_quantity:.4f}")
        return True, "normal"

    def calculate_dynamic_offset(self, pair: TradingPairConfig, bid: float, ask: float, 
                               bid_qty: float, ask_qty: float, side: str) -> int:
        """
        动态计算限价单偏移tick数
        
        Args:
            pair: 交易对配置
            bid: 买一价
            ask: 卖一价
            bid_qty: 买一量
            ask_qty: 卖一量
            side: 交易方向 'BUY' 或 'SELL'
        
        Returns:
            偏移的tick数
        """
        if not pair.dynamic_offset_enabled:
            # 使用静态配置
            return pair.bid_offset_ticks if side == 'BUY' else pair.ask_offset_ticks
        
        # 计算基本指标
        spread = self.calculate_spread_percentage(bid, ask)
        volatility = self.calculate_price_volatility(pair)
        mid_price = (bid + ask) / 2
        
        # 计算深度因子（0-1之间）
        depth_factor = 0.5
        if side == 'BUY':
            # 买单：买一深度越深，可以挂更远
            if bid_qty > pair.fixed_buy_quantity * 10:
                depth_factor = 1.0
            elif bid_qty > pair.fixed_buy_quantity * 5:
                depth_factor = 0.8
            elif bid_qty > pair.fixed_buy_quantity * 2:
                depth_factor = 0.5
            else:
                depth_factor = 0.2
        else:
            # 卖单：卖一深度越深，可以挂更远
            if ask_qty > pair.fixed_buy_quantity * 10:
                depth_factor = 1.0
            elif ask_qty > pair.fixed_buy_quantity * 5:
                depth_factor = 0.8
            elif ask_qty > pair.fixed_buy_quantity * 2:
                depth_factor = 0.5
            else:
                depth_factor = 0.2
        
        # 计算价差因子（价差越大，越应该靠近市场价）
        spread_factor = 1.0
        if spread > pair.spread_threshold_for_offset:
            # 价差超过阈值，减少偏移
            spread_factor = max(0.1, 1.0 - (spread - pair.spread_threshold_for_offset) / pair.spread_threshold_for_offset)
        
        # 计算波动率因子（波动率越大，越应该靠近市场价）
        volatility_factor = 1.0
        if volatility > 0.001:
            volatility_factor = max(0.2, 1.0 - volatility / 0.01)
        
        # 计算综合偏移因子
        total_factor = (
            depth_factor * pair.depth_weight_factor +
            spread_factor * (1 - pair.depth_weight_factor - pair.volatility_weight_factor) +
            volatility_factor * pair.volatility_weight_factor
        )
        
        # 计算动态偏移tick数
        base_offset = pair.min_offset_ticks + (pair.max_offset_ticks - pair.min_offset_ticks) * (1 - total_factor)
        dynamic_offset = int(round(base_offset))
        
        # 确保在最小和最大范围内
        dynamic_offset = max(pair.min_offset_ticks, min(pair.max_offset_ticks, dynamic_offset))
        
        self.logger.debug(f"{pair.symbol} {side}动态偏移计算: "
                         f"深度因子={depth_factor:.2f}, 价差因子={spread_factor:.2f}, "
                         f"波动因子={volatility_factor:.2f}, 总因子={total_factor:.2f}, "
                         f"动态偏移={dynamic_offset}tick")
        
        return dynamic_offset

    def calculate_limit_price(self, pair: TradingPairConfig, side: str, 
                            best_bid: float, best_ask: float, 
                            bid_qty: float, ask_qty: float) -> float:
        """
        计算限价单价格，考虑动态偏移
        
        Args:
            pair: 交易对配置
            side: 交易方向 'BUY' 或 'SELL'
            best_bid: 买一价
            best_ask: 卖一价
            bid_qty: 买一量
            ask_qty: 卖一量
        
        Returns:
            计算后的限价单价格
        """
        if side == 'BUY':
            # 计算买单偏移
            offset_ticks = self.calculate_dynamic_offset(pair, best_bid, best_ask, bid_qty, ask_qty, 'BUY')
            # 买单价格 = 买一价 - tick * n
            buy_price = best_bid - pair.min_price_increment * offset_ticks
            
            # 确保价格不会低于买一价太多（如果有下一个买单）
            if len(self.pair_states[pair.symbol]['order_book'].bids) > 1:
                next_bid = self.pair_states[pair.symbol]['order_book'].bids[1][0]
                buy_price = max(buy_price, next_bid + pair.min_price_increment)
            
            # 确保价格不会高于卖一价
            if buy_price >= best_ask:
                buy_price = best_ask - pair.min_price_increment
            
            self.logger.info(f"{pair.symbol}买单价格计算: "
                           f"买一价={best_bid:.6f}, 偏移={offset_ticks}tick, "
                           f"计算价={buy_price:.6f}")
            
            return buy_price
        
        else:  # SELL
            # 计算卖单偏移
            offset_ticks = self.calculate_dynamic_offset(pair, best_bid, best_ask, bid_qty, ask_qty, 'SELL')
            # 卖单价格 = 卖一价 + tick * m
            sell_price = best_ask + pair.min_price_increment * offset_ticks
            
            # 确保价格不会高于卖一价太多（如果有下一个卖单）
            if len(self.pair_states[pair.symbol]['order_book'].asks) > 1:
                next_ask = self.pair_states[pair.symbol]['order_book'].asks[1][0]
                sell_price = min(sell_price, next_ask - pair.min_price_increment)
            
            # 确保价格不会低于买一价
            if sell_price <= best_bid:
                sell_price = best_bid + pair.min_price_increment
            
            self.logger.info(f"{pair.symbol}卖单价格计算: "
                           f"卖一价={best_ask:.6f}, 偏移={offset_ticks}tick, "
                           f"计算价={sell_price:.6f}")
            
            return sell_price

    def execute_sell_only_strategy(self, pair: TradingPairConfig) -> bool:
        """仅卖出策略：当两个账户余额都充足时，只卖出其中一个账户的代币"""
        self.logger.info(f"执行仅卖出策略: {pair.symbol}")
        
        try:
            at_balance1 = self.client1.get_asset_balance(pair.base_asset)
            at_balance2 = self.client2.get_asset_balance(pair.base_asset)
            
            if at_balance1 >= at_balance2:
                sell_client = self.client1
                sell_client_name = 'ACCOUNT1'
                sell_quantity = min(at_balance1, pair.fixed_buy_quantity)
            else:
                sell_client = self.client2
                sell_client_name = 'ACCOUNT2'
                sell_quantity = min(at_balance2, pair.fixed_buy_quantity)
            
            self.logger.info(f"{pair.symbol}仅卖出详情: {sell_client_name}卖出={sell_quantity:.4f}")
            
            bid, ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
            use_limit_order = self.should_use_limit_strategy(pair)
            
            if use_limit_order and bid > 0 and ask > 0:
                # 使用新的限价单价格计算方法
                sell_price = self.calculate_limit_price(pair, 'SELL', bid, ask, bid_qty, ask_qty)
                
                sell_order = sell_client.create_order(
                    symbol=pair.symbol,
                    side='SELL',
                    order_type='LIMIT',
                    quantity=sell_quantity,
                    min_price_increment=pair.min_price_increment,
                    step_size=pair.step_size,  # 添加step_size参数
                    price=sell_price
                )
                
                if 'orderId' not in sell_order:
                    self.logger.error(f"{pair.symbol}限价卖单失败: {sell_order}")
                    return False
                
                order_id = sell_order['orderId']
                self.logger.info(f"{pair.symbol}限价卖单已挂出: 价格={sell_price:.6f}, 数量={sell_quantity:.4f}")
                
                success = self.wait_for_orders_completion([(sell_client, order_id)], pair.symbol)
                
                if not success:
                    self.logger.warning(f"{pair.symbol}限价卖单未成交，转为市价单")
                    sell_client.cancel_order(pair.symbol, order_id)
                    
                    sell_order = sell_client.create_order(
                        symbol=pair.symbol,
                        side='SELL',
                        order_type='MARKET',
                        quantity=sell_quantity,
                        min_price_increment=pair.min_price_increment,
                        step_size=pair.step_size  # 添加step_size参数
                    )
                    
                    if 'orderId' not in sell_order:
                        self.logger.error(f"{pair.symbol}市价卖单失败: {sell_order}")
                        return False
                    
                    order_id = sell_order['orderId']
                    success = self.wait_for_orders_completion([(sell_client, order_id)], pair.symbol)
            else:
                sell_order = sell_client.create_order(
                    symbol=pair.symbol,
                    side='SELL',
                    order_type='MARKET',
                    quantity=sell_quantity,
                    min_price_increment=pair.min_price_increment,
                    step_size=pair.step_size  # 添加step_size参数
                )
                
                if 'orderId' not in sell_order:
                    self.logger.error(f"{pair.symbol}市价卖单失败: {sell_order}")
                    return False
                
                order_id = sell_order['orderId']
                self.logger.info(f"{pair.symbol}市价卖单已提交")
                success = self.wait_for_orders_completion([(sell_client, order_id)], pair.symbol)
            
            if success:
                self.logger.info(f"✅ {pair.symbol}仅卖出策略执行成功")
                state = self.pair_states[pair.symbol]
                state['sell_only_success_count'] = state.get('sell_only_success_count', 0) + 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"{pair.symbol}仅卖出策略执行出错: {e}")
            return False
        
    def monitor_limit_orders(self, pair: TradingPairConfig, sell_client: AsterDexClient, buy_client: AsterDexClient,
                        sell_order_id: int, buy_order_id: int, sell_quantity: float, buy_quantity: float,
                        initial_sell_price: float, initial_buy_price: float, max_wait_time: float = None) -> Tuple[bool, bool, float, float, float, float]:
        """监控限价单状态，返回成交状态和最新价格"""
        
        if max_wait_time is None:
            max_wait_time = self.order_timeout
        
        start_time = time.time()
        sell_filled = False
        buy_filled = False
        sell_executed_qty = 0.0
        buy_executed_qty = 0.0
        current_sell_price = initial_sell_price
        current_buy_price = initial_buy_price
        last_market_check_time = start_time
        market_check_interval = 1.0
        
        self.logger.info(f"🔄 开始监控 {pair.symbol} 限价单，最大等待时间: {max_wait_time}秒")
        
        while time.time() - start_time < max_wait_time:
            current_time = time.time()
            elapsed_time = time.time() - start_time
            elapsed_percentage = (elapsed_time / max_wait_time) * 100
            
            # 第一步：先检查订单状态
            if not sell_filled:
                try:
                    sell_status = sell_client.get_order(pair.symbol, sell_order_id)
                    sell_status_value = sell_status.get('status')
                    sell_executed_qty = float(sell_status.get('executedQty', 0))
                    
                    if sell_status_value == 'FILLED':
                        sell_filled = True
                        self.logger.info(f"✅ {pair.symbol}限价卖单已完全成交")
                        
                        # 卖单成交后，买单需要继续保持"买一"价格等待
                        # 更新订单簿获取最新市场数据
                        self.update_order_book(pair)
                        current_bid, current_ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
                        
                        # 重新计算买单价格
                        expected_buy_price = self.calculate_limit_price(pair, 'BUY', current_bid, current_ask, bid_qty, ask_qty)
                        
                        if abs(current_buy_price - expected_buy_price) > pair.min_price_increment:
                            self.logger.info(f"🔄 卖单成交，检查买单价格是否需要调整")
                            
                            # 尝试取消买单
                            cancel_result = buy_client.cancel_order(pair.symbol, buy_order_id)
                            
                            # 如果取消成功或订单已成交，重新挂单
                            if 'orderId' in cancel_result or cancel_result.get('status') == 'FILLED':
                                if cancel_result.get('status') == 'FILLED':
                                    self.logger.info(f"✅ 调整买单时发现订单已成交")
                                    buy_filled = True
                                else:
                                    # 重新挂买单到计算的价格
                                    new_buy_price = expected_buy_price
                                    if new_buy_price >= current_ask:
                                        new_buy_price = current_ask - pair.min_price_increment
                                    
                                    buy_order = buy_client.create_order(
                                        symbol=pair.symbol,
                                        side='BUY',
                                        order_type='LIMIT',
                                        quantity=buy_quantity - buy_executed_qty,
                                        min_price_increment=pair.min_price_increment,
                                        price=new_buy_price
                                    )
                                    
                                    if 'orderId' in buy_order:
                                        current_buy_price = new_buy_price
                                        self.logger.info(f"✅ 买单已调整到新价格: {new_buy_price:.6f}")
                                    else:
                                        self.logger.error(f"❌ 买单调整失败")
                            else:
                                self.logger.warning(f"⚠️ 无法取消买单进行调整，可能已成交")
                        
                        self.logger.info(f"💰 卖单成交，买单保持在价格 {current_buy_price:.6f} 等待成交")
                except Exception as e:
                    self.logger.error(f"查询卖单状态时出错: {e}")
            
            if not buy_filled:
                try:
                    buy_status = buy_client.get_order(pair.symbol, buy_order_id)
                    buy_status_value = buy_status.get('status')
                    buy_executed_qty = float(buy_status.get('executedQty', 0))
                    
                    if buy_status_value == 'FILLED':
                        buy_filled = True
                        self.logger.info(f"✅ {pair.symbol}限价买单已完全成交")
                        
                        # 买单成交后，检查卖单价格是否仍有竞争力
                        # 更新订单簿获取最新市场数据
                        self.update_order_book(pair)
                        current_bid, current_ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
                        
                        # 重新计算卖单价格
                        expected_sell_price = self.calculate_limit_price(pair, 'SELL', current_bid, current_ask, bid_qty, ask_qty)
                        price_competitiveness_threshold = pair.min_price_increment * 2
                        is_sell_price_competitive = abs(current_sell_price - expected_sell_price) <= price_competitiveness_threshold
                        
                        if is_sell_price_competitive:
                            self.logger.info(f"💰 买单成交，卖单价格 {current_sell_price:.6f} 仍有竞争力（当前市场价: {expected_sell_price:.6f}），继续等待成交")
                        else:
                            self.logger.info(f"🔄 买单成交，卖单价格 {current_sell_price:.6f} 已无竞争力（当前市场价: {expected_sell_price:.6f}），尝试取消并重新挂单")
                            
                            # 尝试取消卖单
                            cancel_result = sell_client.cancel_order(pair.symbol, sell_order_id)
                            
                            # 如果取消成功或订单已成交，重新挂单
                            if 'orderId' in cancel_result or cancel_result.get('status') == 'FILLED':
                                if cancel_result.get('status') == 'FILLED':
                                    self.logger.info(f"✅ 取消卖单时发现订单已成交")
                                    sell_filled = True
                                else:
                                    # 重新挂卖单到计算的价格
                                    new_sell_price = expected_sell_price
                                    if new_sell_price <= current_bid:
                                        new_sell_price = current_bid + pair.min_price_increment
                                    
                                    remaining_sell_qty = sell_quantity - sell_executed_qty
                                    if remaining_sell_qty > 0:
                                        sell_order = sell_client.create_order(
                                            symbol=pair.symbol,
                                            side='SELL',
                                            order_type='LIMIT',
                                            quantity=remaining_sell_qty,
                                            min_price_increment=pair.min_price_increment,
                                            price=new_sell_price
                                        )
                                        if 'orderId' in sell_order:
                                            current_sell_price = new_sell_price
                                            self.logger.info(f"✅ 卖单已重新挂出: {new_sell_price:.6f}")
                                        else:
                                            self.logger.error(f"❌ 卖单重新挂单失败")
                            else:
                                self.logger.warning(f"⚠️ 无法取消卖单，可能已成交，继续监控")
                except Exception as e:
                    self.logger.error(f"查询买单状态时出错: {e}")
            
            # 如果双方都完全成交，立即返回
            if sell_filled and buy_filled:
                self.logger.info(f"🎉 {pair.symbol}限价单对冲完全成交!")
                return True, True, current_sell_price, current_buy_price, sell_executed_qty, buy_executed_qty
            
            # 第二步：更新订单簿信息
            self.update_order_book(pair)
            
            # 第三步：获取当前市场数据
            current_bid, current_ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
            
            # 第四步：定期检查市场变化（价格竞争力检查）
            if current_time - last_market_check_time >= market_check_interval:
                last_market_check_time = current_time
                
                # 重新计算卖单目标价格
                expected_sell_price = self.calculate_limit_price(pair, 'SELL', current_bid, current_ask, bid_qty, ask_qty)
                
                # 检查卖单价格是否仍然有竞争力
                if not sell_filled and expected_sell_price < current_sell_price - pair.min_price_increment:
                    self.logger.info(f"🔄 市场价格下跌，卖单价格 {current_sell_price:.6f} 已无优势，尝试取消并重新挂单")
                    
                    # 尝试取消卖单
                    cancel_result = sell_client.cancel_order(pair.symbol, sell_order_id)
                    
                    # 如果取消成功或订单已成交，重新挂单
                    if 'orderId' in cancel_result or cancel_result.get('status') == 'FILLED':
                        if cancel_result.get('status') == 'FILLED':
                            self.logger.info(f"✅ 取消卖单时发现订单已成交")
                            sell_filled = True
                        else:
                            # 重新挂卖单到计算的价格
                            new_sell_price = expected_sell_price
                            if new_sell_price <= current_bid:
                                new_sell_price = current_bid + pair.min_price_increment
                            
                            sell_order = sell_client.create_order(
                                symbol=pair.symbol,
                                side='SELL',
                                order_type='LIMIT',
                                quantity=sell_quantity - sell_executed_qty,
                                min_price_increment=pair.min_price_increment,
                                price=new_sell_price
                            )
                            
                            if 'orderId' in sell_order:
                                current_sell_price = new_sell_price
                                self.logger.info(f"✅ 卖单已重新挂出: {new_sell_price:.6f}")
                            else:
                                self.logger.error(f"❌ 卖单重新挂单失败")
                    else:
                        self.logger.warning(f"⚠️ 无法取消卖单，可能已成交，继续监控")
                
                # 重新计算买单目标价格
                expected_buy_price = self.calculate_limit_price(pair, 'BUY', current_bid, current_ask, bid_qty, ask_qty)
                
                # 检查买单价格是否仍然有竞争力 - 无论卖单是否成交
                if not buy_filled and expected_buy_price > current_buy_price + pair.min_price_increment:
                    self.logger.info(f"🔄 市场价格上涨，买单价格 {current_buy_price:.6f} 已无优势，尝试取消并重新挂单")
                    
                    # 尝试取消买单
                    cancel_result = buy_client.cancel_order(pair.symbol, buy_order_id)
                    
                    # 如果取消成功或订单已成交，重新挂单
                    if 'orderId' in cancel_result or cancel_result.get('status') == 'FILLED':
                        if cancel_result.get('status') == 'FILLED':
                            self.logger.info(f"✅ 取消买单时发现订单已成交")
                            buy_filled = True
                        else:
                            # 重新挂买单到计算的价格
                            new_buy_price = expected_buy_price
                            if new_buy_price >= current_ask:
                                new_buy_price = current_ask - pair.min_price_increment
                            
                            buy_order = buy_client.create_order(
                                symbol=pair.symbol,
                                side='BUY',
                                order_type='LIMIT',
                                quantity=buy_quantity - buy_executed_qty,
                                min_price_increment=pair.min_price_increment,
                                price=new_buy_price
                            )
                            
                            if 'orderId' in buy_order:
                                current_buy_price = new_buy_price
                                self.logger.info(f"✅ 买单已重新挂出: {new_buy_price:.6f}")
                            else:
                                self.logger.error(f"❌ 买单重新挂单失败")
                    else:
                        self.logger.warning(f"⚠️ 无法取消买单，可能已成交，继续监控")
            
            # 第五步：检查超时50%情况
            price_competitiveness_threshold = pair.min_price_increment * 2
            
            if elapsed_percentage >= 50 and elapsed_percentage < 100:
                if buy_filled and not sell_filled:
                    # 更新订单簿获取最新市场数据
                    self.update_order_book(pair)
                    current_bid, current_ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
                    
                    # 重新计算卖单目标价格
                    expected_sell_price = self.calculate_limit_price(pair, 'SELL', current_bid, current_ask, bid_qty, ask_qty)
                    is_sell_price_competitive = abs(current_sell_price - expected_sell_price) <= price_competitiveness_threshold
                    
                    if not is_sell_price_competitive:
                        self.logger.info(f"⏰ 超时50%，买单已成交但卖单价格无竞争力，尝试重新挂卖单到市场价")
                        
                        # 尝试取消卖单
                        cancel_result = sell_client.cancel_order(pair.symbol, sell_order_id)
                        
                        # 如果取消成功或订单已成交，重新挂单
                        if 'orderId' in cancel_result or cancel_result.get('status') == 'FILLED':
                            if cancel_result.get('status') == 'FILLED':
                                self.logger.info(f"✅ 取消卖单时发现订单已成交")
                                sell_filled = True
                            else:
                                # 重新挂卖单到计算的价格
                                new_sell_price = expected_sell_price
                                if new_sell_price <= current_bid:
                                    new_sell_price = current_bid + pair.min_price_increment
                                
                                remaining_sell_qty = sell_quantity - sell_executed_qty
                                if remaining_sell_qty > 0:
                                    sell_order = sell_client.create_order(
                                        symbol=pair.symbol,
                                        side='SELL',
                                        order_type='LIMIT',
                                        quantity=remaining_sell_qty,
                                        min_price_increment=pair.min_price_increment,
                                        price=new_sell_price
                                    )
                                    if 'orderId' in sell_order:
                                        current_sell_price = new_sell_price
                                        self.logger.info(f"✅ 卖单已重新挂出: {new_sell_price:.6f}")
                                    else:
                                        self.logger.error(f"❌ 卖单重新挂单失败")
                        else:
                            self.logger.warning(f"⚠️ 无法取消卖单，可能已成交，继续监控")
            
            time.sleep(0.5)
        
        # 监控超时，返回当前状态
        self.logger.info(f"⏰ {pair.symbol}监控超时，当前状态: 卖单成交={sell_filled}, 买单成交={buy_filled}")
        return sell_filled, buy_filled, current_sell_price, current_buy_price, sell_executed_qty, buy_executed_qty


    def format_price(self, price: float, pair: TradingPairConfig) -> float:
        """根据交易对的最小价格变动单位格式化价格"""
        if pair.min_price_increment <= 0:
            return round(price, 6)
        
        precision = self.get_price_precision(pair.min_price_increment)
        return round(price, precision)

    def get_price_precision(self, min_increment: float) -> int:
        """根据最小价格变动单位计算精度位数"""
        if min_increment >= 1:
            return 0
        elif min_increment >= 0.1:
            return 1
        elif min_increment >= 0.01:
            return 2
        elif min_increment >= 0.001:
            return 3
        elif min_increment >= 0.0001:
            return 4
        elif min_increment >= 0.00001:
            return 5
        elif min_increment >= 0.000001:
            return 6
        else:
            return 8

    def strategy_limit_both(self, pair: TradingPairConfig) -> bool:
        """策略1: 限价卖单 + 限价买单对冲，智能订单管理"""
        self.logger.info(f"执行策略1: {pair.symbol}限价单对冲")
        
        try:
            # 更新订单簿获取最新市场数据
            self.update_order_book(pair)
            
            # 获取初始市场数据
            initial_bid, initial_ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
            
            # 动态获取交易方向
            sell_client_name, buy_client_name = self.get_current_trade_direction(pair)
            sell_client = self.client1 if sell_client_name == 'ACCOUNT1' else self.client2
            buy_client = self.client1 if buy_client_name == 'ACCOUNT1' else self.client2
            
            # 获取实际数量
            sell_quantity, _ = self.get_sell_quantity(pair, sell_client_name)
            if sell_quantity > 5000:
                sell_quantity = 5000
            buy_quantity = pair.fixed_buy_quantity
            
            # 使用新的限价单价格计算方法
            sell_price = self.calculate_limit_price(pair, 'SELL', initial_bid, initial_ask, bid_qty, ask_qty)
            buy_price = self.calculate_limit_price(pair, 'BUY', initial_bid, initial_ask, bid_qty, ask_qty)
            
            # 确保价格合理
            if sell_price <= initial_bid:
                sell_price = initial_bid + pair.min_price_increment
            if buy_price >= initial_ask:
                buy_price = initial_ask - pair.min_price_increment
            
            self.logger.info(f"{pair.symbol}交易详情:")
            self.logger.info(f"  {sell_client_name}卖出: {sell_quantity:.4f} @ {sell_price:.6f}")
            self.logger.info(f"  {buy_client_name}买入: {buy_quantity:.4f} @ {buy_price:.6f}")
            self.logger.info(f"  初始市场: 买一={initial_bid:.6f}, 卖一={initial_ask:.6f}")
            
            # 同时挂限价单
            sell_order = sell_client.create_order(
                symbol=pair.symbol,
                side='SELL',
                order_type='LIMIT',
                quantity=sell_quantity,
                min_price_increment=pair.min_price_increment,
                step_size=pair.step_size,  # 添加step_size参数
                price=sell_price
            )
            
            if 'orderId' not in sell_order:
                self.logger.error(f"{pair.symbol}限价卖单失败: {sell_order}")
                return False
            
            sell_order_id = sell_order['orderId']
            
            buy_order = buy_client.create_order(
                symbol=pair.symbol,
                side='BUY',
                order_type='LIMIT',
                quantity=buy_quantity,
                min_price_increment=pair.min_price_increment,
                step_size=pair.step_size,  # 添加step_size参数
                price=buy_price
            )
            
            if 'orderId' not in buy_order:
                self.logger.error(f"{pair.symbol}限价买单失败: {buy_order}")
                # 尝试取消卖单，如果失败则当作已成交
                cancel_result = sell_client.cancel_order(pair.symbol, sell_order_id)
                if 'orderId' not in cancel_result and cancel_result.get('status') != 'FILLED':
                    self.logger.error(f"❌ 取消卖单失败且订单未成交")
                return False
            
            buy_order_id = buy_order['orderId']
            
            self.logger.info(f"{pair.symbol}限价单对冲已挂出: 卖单ID={sell_order_id}, 买单ID={buy_order_id}")
            
            # 第一次监控
            sell_filled, buy_filled, current_sell_price, current_buy_price, sell_executed_qty, buy_executed_qty = self.monitor_limit_orders(
                pair, sell_client, buy_client, sell_order_id, buy_order_id, 
                sell_quantity, buy_quantity, sell_price, buy_price
            )
            
            # 根据监控结果处理
            if sell_filled and buy_filled:
                # 双方都成交，交易成功
                state = self.pair_states[pair.symbol]
                state['limit_both_success_count'] += 1
                return True
            
            elif sell_filled and not buy_filled:
                # 卖单成交，买单未成交 → 继续监控买单
                self.logger.info(f"🔄 卖单已成交，买单未成交，继续监控买单")
                while True:
                    sell_filled, buy_filled, current_sell_price, current_buy_price, sell_executed_qty, buy_executed_qty = self.monitor_limit_orders(
                        pair, sell_client, buy_client, sell_order_id, buy_order_id, 
                        sell_quantity, buy_quantity, current_sell_price, current_buy_price, max_wait_time=30
                    )
                    
                    if buy_filled:
                        self.logger.info(f"🎉 买单最终成交! {pair.symbol}对冲交易完成")
                        state = self.pair_states[pair.symbol]
                        state['limit_both_success_count'] += 1
                        return True
            
            elif buy_filled and not sell_filled:
                # 买单成交，卖单未成交 → 卖单转为市价
                self.logger.info(f"🔄 买单已成交，卖单未成交，卖单转为市价单")
                
                # 尝试取消卖单
                cancel_result = sell_client.cancel_order(pair.symbol, sell_order_id)
                
                # 如果取消失败且不是因为订单已成交，则记录错误
                if 'orderId' not in cancel_result and cancel_result.get('status') != 'FILLED':
                    self.logger.error(f"❌ 取消卖单失败且订单未成交")
                    return False
                
                # 如果取消成功或订单已成交，处理剩余数量
                remaining_sell_qty = sell_quantity - sell_executed_qty
                if remaining_sell_qty > 0 and cancel_result.get('status') != 'FILLED':
                    market_sell = sell_client.create_order(
                        symbol=pair.symbol,
                        side='SELL',
                        order_type='MARKET',
                        quantity=remaining_sell_qty,
                        min_price_increment=pair.min_price_increment,
                        step_size=pair.step_size  # 添加step_size参数
                    )
                    if 'orderId' in market_sell:
                        self.logger.info(f"✅ 卖单市价单已提交")
                        state = self.pair_states[pair.symbol]
                        state['limit_both_success_count'] += 1
                        return True
                    else:
                        self.logger.error(f"❌ 卖单市价单失败")
                        return False
                else:
                    # 卖单已完全成交（部分成交情况或取消时发现已成交）
                    state = self.pair_states[pair.symbol]
                    state['limit_both_success_count'] += 1
                    return True
            
            else:
                # 双方都未成交 → 继续监控
                self.logger.info(f"🔄 双方都未成交，继续监控")
                while True:
                    sell_filled, buy_filled, current_sell_price, current_buy_price, sell_executed_qty, buy_executed_qty = self.monitor_limit_orders(
                        pair, sell_client, buy_client, sell_order_id, buy_order_id, 
                        sell_quantity, buy_quantity, current_sell_price, current_buy_price, max_wait_time=30
                    )
                    
                    if sell_filled or buy_filled:
                        break
                
                # 重新处理状态
                return self.strategy_limit_both(pair)
                
            return False
            
        except Exception as e:
            self.logger.error(f"{pair.symbol}策略1执行出错: {e}")
            try:
                self.client1.cancel_all_orders(pair.symbol)
                self.client2.cancel_all_orders(pair.symbol)
            except:
                pass
            return False

    def strategy_limit_market(self, pair: TradingPairConfig) -> bool:
        """策略3: 限价单 + 市价单对冲，先挂限价单，市价成交后检查限价单"""
        self.logger.info(f"执行策略3: {pair.symbol}限价+市价对冲")
        
        try:
            # 更新订单簿获取最新市场数据
            self.update_order_book(pair)
            
            # 获取市场数据
            bid, ask, bid_qty, ask_qty = self.get_best_bid_ask(pair)
            
            # 动态获取交易方向
            sell_client_name, buy_client_name = self.get_current_trade_direction(pair)
            sell_client = self.client1 if sell_client_name == 'ACCOUNT1' else self.client2
            buy_client = self.client1 if buy_client_name == 'ACCOUNT1' else self.client2
            
            # 获取实际数量
            sell_quantity, _ = self.get_sell_quantity(pair, sell_client_name)
            if sell_quantity > 5000:
                sell_quantity = 5000
            buy_quantity = pair.fixed_buy_quantity
            
            # 决定哪边用限价单，哪边用市价单
            # 默认：卖出用限价单（获取更好价格），买入用市价单（确保成交）
            use_limit_for_sell = True
            use_limit_for_buy = False
            
            # 设置限价单价格（使用新的计算方法）
            limit_price = 0
            if use_limit_for_sell:
                limit_price = self.calculate_limit_price(pair, 'SELL', bid, ask, bid_qty, ask_qty)
                if limit_price <= bid:
                    limit_price = bid + pair.min_price_increment
            else:
                limit_price = self.calculate_limit_price(pair, 'BUY', bid, ask, bid_qty, ask_qty)
                if limit_price >= ask:
                    limit_price = ask - pair.min_price_increment
            
            self.logger.info(f"{pair.symbol}交易详情:")
            if use_limit_for_sell:
                self.logger.info(f"  {sell_client_name}限价卖出: {sell_quantity:.4f} @ {limit_price:.6f}")
                self.logger.info(f"  {buy_client_name}市价买入: {buy_quantity:.4f}")
            else:
                self.logger.info(f"  {sell_client_name}市价卖出: {sell_quantity:.4f}")
                self.logger.info(f"  {buy_client_name}限价买入: {buy_quantity:.4f} @ {limit_price:.6f}")
            self.logger.info(f"  当前市场: 买一={bid:.6f}, 卖一={ask:.6f}")
            
            # 先挂限价单
            limit_order_id = None
            limit_client = None
            market_order_id = None
            market_client = None
            
            if use_limit_for_sell:
                # 挂限价卖单
                limit_order = sell_client.create_order(
                    symbol=pair.symbol,
                    side='SELL',
                    order_type='LIMIT',
                    quantity=sell_quantity,
                    min_price_increment=pair.min_price_increment,
                    step_size=pair.step_size,  # 添加step_size参数
                    price=limit_price
                )
                
                if 'orderId' not in limit_order:
                    self.logger.error(f"{pair.symbol}限价卖单失败: {limit_order}")
                    return False
                
                limit_order_id = limit_order['orderId']
                limit_client = sell_client
                self.logger.info(f"✅ 限价卖单已挂出: ID={limit_order_id}, 价格={limit_price:.6f}")
                
                # 等待片刻让限价单进入订单簿
                time.sleep(0.5)
                
                # 挂市价买单
                market_order = buy_client.create_order(
                    symbol=pair.symbol,
                    side='BUY',
                    order_type='MARKET',
                    quantity=buy_quantity,
                    min_price_increment=pair.min_price_increment,
                    step_size=pair.step_size  # 添加step_size参数
                )
                
                if 'orderId' not in market_order:
                    self.logger.error(f"{pair.symbol}市价买单失败: {market_order}")
                    sell_client.cancel_order(pair.symbol, limit_order_id)
                    return False
                
                market_order_id = market_order['orderId']
                market_client = buy_client
                self.logger.info(f"✅ 市价买单已提交: ID={market_order_id}")
                
            else:
                # 挂限价买单
                limit_order = buy_client.create_order(
                    symbol=pair.symbol,
                    side='BUY',
                    order_type='LIMIT',
                    quantity=buy_quantity,
                    min_price_increment=pair.min_price_increment,
                    step_size=pair.step_size,  # 添加step_size参数
                    price=limit_price
                )
                
                if 'orderId' not in limit_order:
                    self.logger.error(f"{pair.symbol}限价买单失败: {limit_order}")
                    return False
                
                limit_order_id = limit_order['orderId']
                limit_client = buy_client
                self.logger.info(f"✅ 限价买单已挂出: ID={limit_order_id}, 价格={limit_price:.6f}")
                
                # 等待片刻让限价单进入订单簿
                time.sleep(0.5)
                
                # 挂市价卖单
                market_order = sell_client.create_order(
                    symbol=pair.symbol,
                    side='SELL',
                    order_type='MARKET',
                    quantity=sell_quantity,
                    min_price_increment=pair.min_price_increment,
                    step_size=pair.step_size  # 添加step_size参数
                )
                
                if 'orderId' not in market_order:
                    self.logger.error(f"{pair.symbol}市价卖单失败: {market_order}")
                    buy_client.cancel_order(pair.symbol, limit_order_id)
                    return False
                
                market_order_id = market_order['orderId']
                market_client = sell_client
                self.logger.info(f"✅ 市价卖单已提交: ID={market_order_id}")
            
            # 等待市价单成交
            self.logger.info("🔄 等待市价单成交...")
            market_success = self.wait_for_orders_completion([(market_client, market_order_id)], pair.symbol)
            
            if not market_success:
                self.logger.error(f"❌ 市价单未成交，取消限价单")
                limit_client.cancel_order(pair.symbol, limit_order_id)
                return False
            
            self.logger.info("✅ 市价单已成交，检查限价单状态...")
            
            # 检查限价单状态
            limit_order_status = limit_client.get_order(pair.symbol, limit_order_id)
            limit_status = limit_order_status.get('status')
            limit_executed_qty = float(limit_order_status.get('executedQty', 0))
            
            if limit_status == 'FILLED':
                self.logger.info("✅ 限价单已完全成交，对冲交易完成!")
                state = self.pair_states[pair.symbol]
                state['limit_market_success_count'] = state.get('limit_market_success_count', 0) + 1
                return True
            elif limit_status == 'PARTIALLY_FILLED':
                self.logger.info(f"🔄 限价单部分成交: {limit_executed_qty:.4f}/{sell_quantity if use_limit_for_sell else buy_quantity:.4f}")
                
                # 取消剩余部分并转为市价单
                cancel_result = limit_client.cancel_order(pair.symbol, limit_order_id)
                
                if 'orderId' in cancel_result or cancel_result.get('status') == 'FILLED':
                    remaining_qty = (sell_quantity if use_limit_for_sell else buy_quantity) - limit_executed_qty
                    
                    if remaining_qty > 0:
                        self.logger.info(f"🔄 限价单剩余 {remaining_qty:.4f} 转为市价单")
                        
                        if use_limit_for_sell:
                            # 剩余卖单转为市价
                            market_sell = sell_client.create_order(
                                symbol=pair.symbol,
                                side='SELL',
                                order_type='MARKET',
                                quantity=remaining_qty,
                                min_price_increment=pair.min_price_increment,
                                step_size=pair.step_size  # 添加step_size参数
                            )
                        else:
                            # 剩余买单转为市价
                            market_buy = buy_client.create_order(
                                symbol=pair.symbol,
                                side='BUY',
                                order_type='MARKET',
                                quantity=remaining_qty,
                                min_price_increment=pair.min_price_increment,
                                step_size=pair.step_size  # 添加step_size参数
                            )
                        
                        if 'orderId' in (market_sell if use_limit_for_sell else market_buy):
                            self.logger.info("✅ 剩余数量市价单已提交")
                            # 等待市价单成交
                            success = self.wait_for_orders_completion([
                                (sell_client if use_limit_for_sell else buy_client, 
                                 market_sell['orderId'] if use_limit_for_sell else market_buy['orderId'])
                            ], pair.symbol)
                            
                            if success:
                                state = self.pair_states[pair.symbol]
                                state['limit_market_success_count'] = state.get('limit_market_success_count', 0) + 1
                                return True
                        else:
                            self.logger.error("❌ 剩余数量市价单失败")
                            return False
                else:
                    self.logger.error("❌ 取消限价单失败")
                    return False
            else:
                # 限价单未成交，转为市价单
                self.logger.info("🔄 限价单未成交，转为市价单")
                
                cancel_result = limit_client.cancel_order(pair.symbol, limit_order_id)
                
                if 'orderId' in cancel_result or cancel_result.get('status') == 'FILLED':
                    if cancel_result.get('status') == 'FILLED':
                        self.logger.info("✅ 取消时发现限价单已成交")
                        state = self.pair_states[pair.symbol]
                        state['limit_market_success_count'] = state.get('limit_market_success_count', 0) + 1
                        return True
                    
                    # 挂市价单完成剩余交易
                    if use_limit_for_sell:
                        market_sell = sell_client.create_order(
                            symbol=pair.symbol,
                            side='SELL',
                            order_type='MARKET',
                            quantity=sell_quantity,
                            min_price_increment=pair.min_price_increment,
                            step_size=pair.step_size  # 添加step_size参数
                        )
                    else:
                        market_buy = buy_client.create_order(
                            symbol=pair.symbol,
                            side='BUY',
                            order_type='MARKET',
                            quantity=buy_quantity,
                            min_price_increment=pair.min_price_increment,
                            step_size=pair.step_size  # 添加step_size参数
                        )
                    
                    if 'orderId' in (market_sell if use_limit_for_sell else market_buy):
                        self.logger.info("✅ 限价单转为市价单已提交")
                        # 等待市价单成交
                        success = self.wait_for_orders_completion([
                            (sell_client if use_limit_for_sell else buy_client, 
                             market_sell['orderId'] if use_limit_for_sell else market_buy['orderId'])
                        ], pair.symbol)
                        
                        if success:
                            state = self.pair_states[pair.symbol]
                            state['limit_market_success_count'] = state.get('limit_market_success_count', 0) + 1
                            return True
                    else:
                        self.logger.error("❌ 限价单转为市价单失败")
                        return False
                else:
                    self.logger.error("❌ 取消限价单失败")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"{pair.symbol}策略3执行出错: {e}")
            try:
                # 尝试取消所有挂单
                self.client1.cancel_all_orders(pair.symbol)
                self.client2.cancel_all_orders(pair.symbol)
            except:
                pass
            return False
        
    def strategy_market_only(self, pair: TradingPairConfig) -> bool:
        """策略2: 同时挂市价单对冲"""
        self.logger.info(f"执行策略2: {pair.symbol}同时市价单对冲")
        
        try:
            # 动态获取交易方向
            sell_client_name, buy_client_name = self.get_current_trade_direction(pair)
            sell_client = self.client1 if sell_client_name == 'ACCOUNT1' else self.client2
            buy_client = self.client1 if buy_client_name == 'ACCOUNT1' else self.client2
            
            # 卖单数量：实际持有量
            sell_quantity, _ = self.get_sell_quantity(pair, sell_client_name)
            # 买单数量：固定配置量
            buy_quantity = pair.fixed_buy_quantity
            
            self.logger.info(f"{pair.symbol}交易详情: {sell_client_name}卖出={sell_quantity:.4f}, {buy_client_name}买入={buy_quantity:.4f}")
            
            # 同时下市价单
            sell_order = sell_client.create_order(
                symbol=pair.symbol,
                side='SELL',
                order_type='MARKET',
                quantity=sell_quantity,
                min_price_increment=pair.min_price_increment,
                step_size=pair.step_size  # 添加step_size参数
            )
            
            if 'orderId' not in sell_order:
                self.logger.error(f"{pair.symbol}市价卖单失败: {sell_order}")
                return False
            
            sell_order_id = sell_order['orderId']
            
            buy_order = buy_client.create_order(
                symbol=pair.symbol,
                side='BUY',
                order_type='MARKET',
                quantity=buy_quantity,
                min_price_increment=pair.min_price_increment,
                step_size=pair.step_size  # 添加step_size参数
            )
            
            if 'orderId' not in buy_order:
                self.logger.error(f"{pair.symbol}市价买单失败: {buy_order}")
                sell_client.cancel_order(pair.symbol, sell_order_id)
                return False
            
            buy_order_id = buy_order['orderId']
            
            self.logger.info(f"{pair.symbol}市价单对冲已提交: 卖单ID={sell_order_id}, 买单ID={buy_order_id}")
            
            # 等待并检查成交
            success = self.wait_for_orders_completion([
                (sell_client, sell_order_id),
                (buy_client, buy_order_id)
            ], pair.symbol)
            
            if success:
                state = self.pair_states[pair.symbol]
                state['market_sell_success_count'] += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"{pair.symbol}策略2执行出错: {e}")
            return False
        
    def wait_for_orders_completion(self, orders: List[Tuple[AsterDexClient, int]], symbol: str) -> bool:
        """等待订单完成"""
        start_time = time.time()
        completed = [False] * len(orders)
        
        while time.time() - start_time < self.order_timeout:
            all_completed = True
            
            for i, (client, order_id) in enumerate(orders):
                if not completed[i]:
                    order_status = client.get_order(symbol, order_id)
                    if order_status.get('status') in ['FILLED', 'PARTIALLY_FILLED']:
                        completed[i] = True
                        self.logger.info(f"{symbol}订单 {order_id} 已成交")
                    elif order_status.get('status') in ['CANCELED', 'REJECTED', 'EXPIRED']:
                        self.logger.error(f"{symbol}订单 {order_id} 失败: {order_status.get('status')}")
                        for j, (other_client, other_id) in enumerate(orders):
                            if j != i and not completed[j]:
                                other_client.cancel_order(symbol, other_id)
                        return False
                    else:
                        all_completed = False
            
            if all_completed:
                return True
            
            time.sleep(0.5)
        
        self.logger.warning(f"{symbol}订单等待超时，取消未完成订单")
        for client, order_id in orders:
            if not any(c[1] == order_id and completed[i] for i, c in enumerate(orders)):
                client.cancel_order(symbol, order_id)
        
        return False

    def execute_trading_cycle(self, pair: TradingPairConfig) -> bool:
        """执行一个交易周期，根据余额情况选择交易模式"""
        market_ok, trade_mode = self.check_market_conditions(pair)
        
        if not market_ok:
            if trade_mode == "volume_insufficient":
                # 成交量不足时，只更新数据但不交易
                self.update_volume_data(pair)
                return False
            return False
        
        state = self.pair_states[pair.symbol]
        state['trade_count'] += 1
        
        start_time = time.time()
        success = False
        
        if trade_mode == "sell_only":
            success = self.execute_sell_only_strategy(pair)
            actual_strategy = TradingStrategy.MARKET_ONLY
        else:
            actual_strategy = pair.strategy
            if pair.strategy == TradingStrategy.AUTO:
                actual_strategy = self.get_best_strategy(pair)
                self.logger.info(f"🎯 {pair.symbol}自动选择策略: {actual_strategy.value}")
            
            if actual_strategy == TradingStrategy.LIMIT_BOTH:
                success = self.strategy_limit_both(pair)
            elif actual_strategy == TradingStrategy.MARKET_ONLY:
                success = self.strategy_market_only(pair)
            elif actual_strategy == TradingStrategy.LIMIT_MARKET:
                success = self.strategy_limit_market(pair)
                if not success:
                    self.logger.info(f"🔄 {pair.symbol}LIMIT_MARKET策略失败，尝试备用方案")
                    # 备用方案：直接使用市价对冲
                    success = self.strategy_market_only(pair)
        
        execution_time = time.time() - start_time
        
        if success:
            if trade_mode == "sell_only":
                trade_volume = pair.fixed_buy_quantity
            else:
                trade_volume = pair.fixed_buy_quantity * 2
                
            state['volume'] += trade_volume
            state['successful_trades'] += 1
            self.total_volume += trade_volume
            
            self.record_strategy_performance(pair, actual_strategy, True, execution_time, trade_volume)
            
            if trade_mode == "sell_only":
                self.logger.info(f"✓ {pair.symbol}仅卖出交易成功! (耗时: {execution_time:.2f}s)")
            else:
                sell_account, buy_account = self.get_current_trade_direction(pair)
                self.logger.info(f"✓ {pair.symbol}对冲交易成功! {sell_account}卖出 → {buy_account}买入 (策略: {actual_strategy.value}, 耗时: {execution_time:.2f}s)")
            
            self.logger.info(f"  {pair.symbol}本次交易量: {trade_volume:.4f}, 累计: {state['volume']:.2f}/{pair.target_volume}")
            
            self.update_cache_after_trade(pair)
        else:
            self.logger.error(f"✗ {pair.symbol}交易失败 (模式: {trade_mode}, 耗时: {execution_time:.2f}s)")
            self.record_strategy_performance(pair, actual_strategy, False, execution_time, 0)
            self.update_cache_after_failure(pair)
        
        return success

    def update_cache_after_trade(self, pair: TradingPairConfig):
        """交易成功后更新缓存数据"""
        self.logger.info(f"🔄 {pair.symbol}交易成功，更新缓存数据...")
        self.client1.refresh_balance_cache()
        self.client2.refresh_balance_cache()
        self.update_trade_direction_cache(pair)
        self.logger.info(f"✅ {pair.symbol}缓存数据已更新")

    def update_cache_after_failure(self, pair: TradingPairConfig):
        """交易失败后更新缓存数据"""
        self.logger.info(f"🔄 {pair.symbol}交易失败，更新缓存数据...")
        self.client1.refresh_balance_cache()
        self.client2.refresh_balance_cache()
        self.update_trade_direction_cache(pair)
        self.logger.info(f"✅ {pair.symbol}缓存数据已更新")

    def print_strategy_performance(self):
        """打印策略性能统计"""
        self.logger.info("\n📈 策略性能统计:")
        
        for pair in self.trading_pairs:
            self.logger.info(f"\n   {pair.symbol} (配置策略: {pair.strategy.value}):")
            
            performances = self.strategy_performance[pair.symbol]
            for strategy, perf in performances.items():
                if perf.total_count > 0:
                    self.logger.info(f"     {strategy.value}:")
                    self.logger.info(f"       执行次数: {perf.total_count}")
                    self.logger.info(f"       成功次数: {perf.success_count}")
                    self.logger.info(f"       成功率: {perf.success_rate:.1f}%")
                    self.logger.info(f"       平均执行时间: {perf.avg_execution_time:.2f}s")
                    self.logger.info(f"       总交易量: {perf.total_volume:.2f}")
                    if perf.success_count > 0:
                        self.logger.info(f"       平均交易量: {perf.avg_volume_per_trade:.2f}")
            
            best_strategy = self.get_best_strategy(pair)
            self.logger.info(f"     💡 推荐策略: {best_strategy.value}")

    def print_trading_statistics(self):
        """打印交易统计信息"""
        self.logger.info("\n📊 总体交易统计信息:")
        self.logger.info(f"   总交易量: {self.total_volume:.2f}")
        
        for pair in self.trading_pairs:
            state = self.pair_states[pair.symbol]
            self.logger.info(f"\n   {pair.symbol}统计 (配置策略: {pair.strategy.value}):")
            self.logger.info(f"     最小价格变动单位: {pair.min_price_increment}")
            self.logger.info(f"     5分钟最小成交量要求: {pair.min_5min_volume}")
            self.logger.info(f"     限价单偏移配置: 买单偏移{pair.bid_offset_ticks}tick, 卖单偏移{pair.ask_offset_ticks}tick")
            self.logger.info(f"     动态偏移: {'启用' if pair.dynamic_offset_enabled else '禁用'}")
            if pair.dynamic_offset_enabled:
                self.logger.info(f"     动态偏移范围: {pair.min_offset_ticks}-{pair.max_offset_ticks}tick")
            self.logger.info(f"     总尝试次数: {state['trade_count']}")
            self.logger.info(f"     成功交易次数: {state['successful_trades']}")
            
            if state['trade_count'] > 0:
                success_rate = (state['successful_trades'] / state['trade_count']) * 100
                self.logger.info(f"     成功率: {success_rate:.1f}%")
            
            self.logger.info(f"     卖单限价单尝试次数: {state['limit_sell_attempt_count']}")
            self.logger.info(f"     卖单限价单成功次数: {state['limit_sell_success_count']}")
            self.logger.info(f"     卖单限价单部分成交次数: {state['partial_limit_sell_count']}")
            
            if state['limit_sell_attempt_count'] > 0:
                limit_sell_success_rate = (state['limit_sell_success_count'] / state['limit_sell_attempt_count']) * 100
                self.logger.info(f"     卖单限价单成功率: {limit_sell_success_rate:.1f}%")
            
            self.logger.info(f"     卖单市价单成功次数: {state['market_sell_success_count']}")
            self.logger.info(f"     限价双方策略成功次数: {state.get('limit_both_success_count', 0)}")
            self.logger.info(f"     限价市价策略成功次数: {state.get('limit_market_success_count', 0)}")
            self.logger.info(f"     累计交易量: {state['volume']:.2f}/{pair.target_volume}")
        
        self.logger.info(f"\n   Aster购买统计:")
        self.logger.info(f"     Aster购买尝试次数: {self.aster_buy_attempts}")
        self.logger.info(f"     Aster购买成功次数: {self.aster_buy_success}")
        self.logger.info(f"     Aster购买失败次数: {self.aster_buy_failed}")

    def print_aster_statistics(self):
        """打印Aster相关统计"""
        aster_balance1 = self.client1.get_asset_balance(self.aster_asset)
        aster_balance2 = self.client2.get_asset_balance(self.aster_asset)
        
        self.logger.info("\n⭐ Aster代币统计:")
        self.logger.info(f"   账户1 Aster余额: {aster_balance1:.4f}")
        self.logger.info(f"   账户2 Aster余额: {aster_balance2:.4f}")
        self.logger.info(f"   最低要求余额: {self.min_aster_balance:.4f}")
        self.logger.info(f"   每次购买数量: {self.aster_buy_quantity:.4f}")

    def print_account_balances(self):
        """打印账户余额"""
        try:
            self.logger.info("\n💰 账户余额:")
            
            usdt_balance1 = self.client1.get_asset_balance('USDT')
            aster_balance1 = self.client1.get_asset_balance(self.aster_asset)
            usdt_balance2 = self.client2.get_asset_balance('USDT')
            aster_balance2 = self.client2.get_asset_balance(self.aster_asset)
            
            self.logger.info(f"   账户1: USDT={usdt_balance1:.2f}, {self.aster_asset}={aster_balance1:.2f}")
            self.logger.info(f"   账户2: USDT={usdt_balance2:.2f}, {self.aster_asset}={aster_balance2:.2f}")
            
            for pair in self.trading_pairs:
                at_balance1 = self.client1.get_asset_balance(pair.base_asset)
                at_balance2 = self.client2.get_asset_balance(pair.base_asset)
                
                self.logger.info(f"   {pair.base_asset}: 账户1={at_balance1:.4f}, 账户2={at_balance2:.4f}")
                
                sell_account, buy_account = self.get_current_trade_direction(pair)
                self.logger.info(f"   {pair.symbol}推荐方向: {sell_account}卖出 → {buy_account}买入 (策略: {pair.strategy.value})")
            
        except Exception as e:
            self.logger.error(f"获取余额时出错: {e}")

    def monitor_and_trade(self):
        """监控市场并执行交易"""
        self.logger.info("开始多交易对智能刷量交易...")
        self.is_running = True
        
        consecutive_failures = 0
        
        while self.is_running:
            try:
                current_pair = self.get_current_trading_pair()
                self.client1.cancel_all_orders(current_pair.symbol)
                self.client2.cancel_all_orders(current_pair.symbol)
                
                self.update_order_book(current_pair)
                self.update_volume_data(current_pair)
                
                if self.execute_trading_cycle(current_pair):
                    consecutive_failures = 0
                    state = self.pair_states[current_pair.symbol]
                    if state['successful_trades'] % 5 == 0:
                        self.print_account_balances()
                        self.print_trading_statistics()
                        self.print_strategy_performance()
                        self.print_aster_statistics()
                    
                    if state['volume'] >= current_pair.target_volume:
                        self.logger.info(f"🎉 {current_pair.symbol}达到目标交易量: {state['volume']:.2f}/{current_pair.target_volume}")
                        time.sleep(self.check_interval)
                        self.switch_to_next_pair()
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        self.logger.warning("连续多次交易失败，暂停2秒并切换到下一个交易对...")
                        time.sleep(2)
                        consecutive_failures = 0
                        self.switch_to_next_pair()
                
                current_state = self.pair_states[current_pair.symbol]
                progress = current_state['volume'] / current_pair.target_volume * 100
                success_rate = (current_state['successful_trades'] / current_state['trade_count'] * 100) if current_state['trade_count'] > 0 else 0
                
                # 显示当前5分钟成交量
                current_5min_volume = current_state.get('current_5min_volume', 0.0)
                volume_status = f", 5分钟成交量: {current_5min_volume:.2f}/{current_pair.min_5min_volume:.2f}" if current_pair.min_5min_volume > 0 else ""
                
                self.logger.info(f"{current_pair.symbol}进度: {progress:.1f}% ({current_state['volume']:.2f}/{current_pair.target_volume}), 成功率: {success_rate:.1f}%, 策略: {current_pair.strategy.value}{volume_status}")
                
                time.sleep(self.check_interval)
                self.switch_to_next_pair()
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"交易周期出错: {e}")
                time.sleep(self.check_interval)
        
        self.logger.info("交易已停止")

    def start(self):
        """启动交易程序"""
        config_name = os.path.splitext(os.path.basename(self.account_config_file))[0]
        self.logger.info("=" * 60)
        self.logger.info(f"多交易对智能刷量交易程序启动 [账户配置: {config_name}]")
        self.logger.info(f"交易对数量: {len(self.trading_pairs)}")
        for i, pair in enumerate(self.trading_pairs):
            volume_info = f", 5分钟最小成交量: {pair.min_5min_volume}" if pair.min_5min_volume > 0 else ""
            offset_info = f", 限价单偏移: 买单{pair.bid_offset_ticks}tick/卖单{pair.ask_offset_ticks}tick"
            self.logger.info(f"  {i+1}. {pair.symbol} (目标: {pair.target_volume}, 数量: {pair.fixed_buy_quantity}, 策略: {pair.strategy.value}{volume_info}{offset_info})")
        self.logger.info(f"Aster代币: {self.aster_asset}")
        self.logger.info(f"最低Aster余额: {self.min_aster_balance}")
        self.logger.info(f"默认策略: AUTO")
        self.logger.info("=" * 60)

        self.logger.info("\n🔄 启动前清理挂单...")
        self.cancel_all_open_orders_before_start()
        
        self.logger.info("🔄 初始化缓存数据...")
        self.client1.refresh_balance_cache()
        self.client2.refresh_balance_cache()
        
        for pair in self.trading_pairs:
            self.update_trade_direction_cache(pair)
        
        self.logger.info("✅ 缓存数据初始化完成")

        # 初始化成交量数据
        self.logger.info("🔄 初始化成交量数据...")
        for pair in self.trading_pairs:
            if pair.min_5min_volume > 0:
                initial_volume = self.get_5min_volume_from_klines(pair)
                self.pair_states[pair.symbol]['current_5min_volume'] = initial_volume
                self.logger.info(f"   {pair.symbol} 初始5分钟成交量: {initial_volume:.2f}")
        
        for pair in self.trading_pairs:
            self.logger.info(f"\n🔍 检查{pair.base_asset}余额状态...")
            if not self.initialize_at_balance(pair):
                self.logger.error(f"❌ {pair.base_asset}余额初始化失败")
        
        self.logger.info("\n🔍 检查Aster余额状态...")
        if not self.check_and_buy_aster_if_needed():
            self.logger.error("❌ Aster余额初始化失败，程序退出")
            return
        
        self.logger.info("\n📊 开始统计历史交易量...")
        self.calculate_historical_volume()
        
        self.logger.info("\n初始账户余额和推荐交易方向:")
        self.print_account_balances()
        self.print_aster_statistics()
        self.print_historical_volume_statistics()
        self.logger.info("")
        
        self.logger.info("\n5s后开始交易...")
        time.sleep(5)
        self.monitor_and_trade()
    
    def stop(self):
        """停止交易"""
        self.is_running = False
        self.logger.info("\n交易程序已停止")
        self.logger.info("=" * 50)
        self.logger.info("最终交易统计:")
        self.print_trading_statistics()
        self.logger.info("\n策略性能统计:")
        self.print_strategy_performance()
        self.logger.info("\nAster统计:")
        self.print_aster_statistics()
        self.logger.info("\n历史交易量统计:")
        self.print_historical_volume_statistics()
        self.logger.info("=" * 50)
        self.logger.info("最终账户余额:")
        self.print_account_balances()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多交易对智能刷量交易程序')
    parser.add_argument('-c', '--config', type=str, default='account.yaml', 
                       help='账户配置文件路径 (默认: account.yaml)')
    parser.add_argument('-l', '--list-configs', action='store_true',
                       help='列出可用的配置文件')
    parser.add_argument('--log', type=str, metavar='FILENAME',
                       help='自定义日志文件名 (不需要.log后缀)')
    
    args = parser.parse_args()
    
    if args.list_configs:
        config_files = [f for f in os.listdir('.') if f.endswith('.yaml') or f.endswith('.yml')]
        print("可用的配置文件:")
        for config_file in config_files:
            print(f"  - {config_file}")
        return
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        print("使用 -l 参数查看可用的配置文件")
        return
    
    maker = SmartMarketMaker(account_config_file=args.config, log_filename=args.log)
    
    try:
        maker.start()
    except KeyboardInterrupt:
        logger.info("\n收到停止信号...")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
    finally:
        maker.stop()

if __name__ == "__main__":
    main()