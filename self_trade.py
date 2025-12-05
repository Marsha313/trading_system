import json
import time
import hmac
import hashlib
import requests
import logging
import yaml
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import argparse
import sys
from collections import deque
import threading
from datetime import datetime, timedelta

# 先设置基础日志，后续会根据配置调整
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AccountConfig:
    api_key: str
    secret_key: str
    name: str

@dataclass
class TradingConfig:
    symbol: str
    order_amount: Decimal  # 订单金额（代替quantity）
    price_gap_threshold: Decimal  # 价差阈值（tick数）
    price_precision: int
    quantity_precision: int
    tick_size: Decimal  # 最小价格变动单位
    step_size: Decimal  # 最小数量变动单位
    min_notional: Decimal  # 最小名义金额
    stability_period: int = 5  # 稳定性检测周期（秒）
    max_price_adjustment: int = 5  # 最大价格调整tick数
    wait_time: int = 10  # 等待时间（秒）
    sampling_rate: float = 0.5  # 采样频率（秒）
    recv_window: int = 5000
    base_url: str = "https://sapi.asterdex.com"  # 改为现货API地址
    log_level: str = "INFO"
    log_file: str = "spot_auto_trading.log"
    daily_volume_target: Decimal = field(default_factory=lambda: Decimal('0'))  # 每日目标成交量


class ConfigLoader:
    """配置加载器"""

    @staticmethod
    def load_accounts_config(config_path: str) -> Tuple[Dict, str]:
        """加载账号配置文件"""
        try:
            logger.info(f"正在加载账号配置文件: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                accounts_config = yaml.safe_load(f)

            trading_config_path = accounts_config.get('trading_config_path', 'spot_trading_config.yaml')
            logger.info(f"找到交易配置文件路径: {trading_config_path}")

            # 验证账号配置
            if 'account1' not in accounts_config or 'account2' not in accounts_config:
                raise ValueError("账号配置文件中必须包含 account1 和 account2 配置")

            logger.info("账号配置文件加载成功")
            return accounts_config, trading_config_path

        except Exception as e:
            logger.error(f"加载账号配置文件失败: {e}")
            raise

    @staticmethod
    def load_trading_config(config_path: str) -> Dict:
        """加载交易配置文件"""
        try:
            logger.info(f"正在加载交易配置文件: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                trading_config = yaml.safe_load(f)

            # 验证必要的配置项
            required_fields = ['symbol', 'order_amount', 'price_gap_threshold', 'tick_size', 'step_size', 'min_notional']
            for field in required_fields:
                if field not in trading_config:
                    raise ValueError(f"交易配置文件中缺少必要字段: {field}")

            # 添加每日目标成交量参数（可选）
            if 'daily_volume_target' not in trading_config:
                trading_config['daily_volume_target'] = '0'
                logger.info("未设置每日目标成交量，使用默认值0（无限制）")

            logger.info("交易配置文件加载成功")
            return trading_config

        except Exception as e:
            logger.error(f"加载交易配置文件失败: {e}")
            raise

    @staticmethod
    def setup_logging(log_level: str, log_file: str):
        """设置日志"""
        logger.info(f"设置日志级别: {log_level}, 日志文件: {log_file}")

        # 清除之前的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 设置日志级别
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 文件处理器
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"文件日志处理器已设置: {log_file}")
        except Exception as e:
            logger.error(f"设置文件日志失败: {e}")

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.info("控制台日志处理器已设置")


class PriceStabilityMonitor:
    """价格稳定性监测器"""
    def __init__(self, stability_period: int, tick_size: Decimal, sampling_rate: float = 0.5):
        self.stability_period = stability_period
        self.tick_size = tick_size
        self.sampling_rate = sampling_rate
        # 根据采样频率计算需要的数据点数量
        self.max_samples = int(stability_period / sampling_rate)
        self.bid_prices: Deque[Decimal] = deque(maxlen=self.max_samples)
        self.ask_prices: Deque[Decimal] = deque(maxlen=self.max_samples)
        self.stable_start_time: Optional[float] = None
        logger.info(f"价格稳定性监测器初始化: 稳定周期={stability_period}秒, 采样频率={sampling_rate}秒, 最大样本数={self.max_samples}")

    def update_prices(self, bid: Decimal, ask: Decimal) -> Tuple[bool, bool]:
        """更新价格并检查稳定性和价差
        返回: (是否稳定, 是否有价差)
        """
        current_time = time.time()
        self.bid_prices.append(bid)
        self.ask_prices.append(ask)

        logger.debug(f"更新价格数据: 买一={bid}, 卖一={ask}, 当前样本数={len(self.bid_prices)}")

        # 检查是否有足够的数据点
        if len(self.bid_prices) < self.max_samples:
            logger.debug(f"数据点不足，需要{self.max_samples}个，当前{len(self.bid_prices)}个")
            return False, False

        # 计算最近稳定期内的价格波动（以tick为单位）
        recent_bids = list(self.bid_prices)
        recent_asks = list(self.ask_prices)

        bid_volatility_ticks = (max(recent_bids) - min(recent_bids)) / self.tick_size
        ask_volatility_ticks = (max(recent_asks) - min(recent_asks)) / self.tick_size

        # 检查价差 - 至少要有1个tick的价差
        current_gap_ticks = (ask - bid) / self.tick_size
        has_gap = current_gap_ticks >= Decimal('1')

        # 如果价格波动在1个tick以内，认为稳定
        is_stable = bid_volatility_ticks <= Decimal('1') and ask_volatility_ticks <= Decimal('1')

        if is_stable and has_gap:
            if self.stable_start_time is None:
                self.stable_start_time = current_time
                logger.info(f"开始检测价格稳定性，当前波动: 买={bid_volatility_ticks:.1f}tick, 卖={ask_volatility_ticks:.1f}tick, 价差={current_gap_ticks:.1f}tick")
            else:
                stability_duration = current_time - self.stable_start_time
                logger.debug(f"价格持续稳定: {stability_duration:.1f}秒")
                if stability_duration >= self.stability_period:
                    logger.info(f"价格稳定持续 {stability_duration:.1f} 秒，价差={current_gap_ticks:.1f}tick，满足条件")
                    return True, True
        else:
            if self.stable_start_time is not None:
                reason = []
                if not is_stable:
                    reason.append(f"价格不稳定(买={bid_volatility_ticks:.1f}tick, 卖={ask_volatility_ticks:.1f}tick)")
                if not has_gap:
                    reason.append(f"价差不足({current_gap_ticks:.1f}tick)")
                logger.info(f"价格稳定性被打破: {'，'.join(reason)}")
            self.stable_start_time = None

        return is_stable, has_gap


class AsterDexSpotAPIClient:
    def __init__(self, config: TradingConfig, account: AccountConfig):
        self.config = config
        self.account = account
        self.session = requests.Session()
        logger.info(f"初始化现货API客户端: {account.name}")

    def _sign_request(self, params: Dict) -> str:
        """生成签名"""
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(
            self.account.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        logger.debug(f"生成签名: {signature}")
        return signature

    def _request(self, method: str, endpoint: str, signed: bool = False, **params) -> Dict:
        """发送API请求"""
        url = f"{self.config.base_url}{endpoint}"

        if signed:
            timestamp = int(time.time() * 1000)
            params['timestamp'] = timestamp
            params['recvWindow'] = self.config.recv_window
            params['signature'] = self._sign_request(params)
            logger.debug(f"签名请求参数: {params}")

        headers = {
            'X-MBX-APIKEY': self.account.api_key
        }

        logger.debug(f"发送{method}请求到: {url}, 参数: {params}")

        try:
            if method == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, data=params, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, data=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            response.raise_for_status()
            result = response.json()
            logger.debug(f"API响应: {result}")
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"响应状态码: {e.response.status_code}")
                logger.error(f"响应内容: {e.response.text}")
            raise

    def get_order_book(self) -> Dict:
        """获取订单簿"""
        logger.debug("获取订单簿数据")
        return self._request('GET', '/api/v1/depth', symbol=self.config.symbol, limit=10)

    def get_exchange_info(self) -> Dict:
        """获取交易对信息"""
        logger.debug("获取交易所信息")
        return self._request('GET', '/api/v1/exchangeInfo')

    def place_order(self, side: str, price: Decimal, quantity: Decimal,
                   order_type: str = 'LIMIT', time_in_force: str = 'GTC') -> Dict:
        """下单"""
        logger.info(f"准备下单: {side} {quantity} {self.config.symbol} @ {price}")

        # 确保价格是tick_size的整数倍
        adjusted_price = self.adjust_to_tick_size(price)

        # 确保数量是step_size的整数倍
        adjusted_quantity = self.adjust_to_step_size(quantity)

        logger.debug(f"价格调整: {price} -> {adjusted_price}")
        logger.debug(f"数量调整: {quantity} -> {adjusted_quantity}")

        params = {
            'symbol': self.config.symbol,
            'side': side,
            'type': order_type,
            'quantity': str(adjusted_quantity),
            'price': str(adjusted_price),
            'timeInForce': time_in_force
        }

        logger.info(f"下单参数: {params}")
        return self._request('POST', '/api/v1/order', signed=True, **params)

    def place_market_order(self, side: str, quantity: Decimal) -> Dict:
        """下市价单"""
        logger.info(f"准备下市价单: {side} {quantity} {self.config.symbol}")

        # 确保数量是step_size的整数倍
        adjusted_quantity = self.adjust_to_step_size(quantity)

        logger.debug(f"市价单数量调整: {quantity} -> {adjusted_quantity}")

        params = {
            'symbol': self.config.symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': str(adjusted_quantity)
        }

        logger.info(f"市价单参数: {params}")
        return self._request('POST', '/api/v1/order', signed=True, **params)

    def adjust_to_tick_size(self, price: Decimal) -> Decimal:
        """调整价格到tick_size的整数倍"""
        tick_size = self.config.tick_size
        adjusted = (price / tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick_size
        result = adjusted.quantize(Decimal(f'1e-{self.config.price_precision}'))
        logger.debug(f"价格调整: {price} -> {result} (tick_size={tick_size})")
        return result

    def adjust_to_step_size(self, quantity: Decimal) -> Decimal:
        """调整数量到step_size的整数倍"""
        step_size = self.config.step_size
        adjusted = (quantity / step_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * step_size
        result = adjusted.quantize(Decimal(f'1e-{self.config.quantity_precision}'))
        logger.debug(f"数量调整: {quantity} -> {result} (step_size={step_size})")
        return result

    def calculate_quantity_from_amount(self, amount: Decimal, price: Decimal) -> Decimal:
        """根据金额和价格计算数量，确保满足最小金额要求和step_size"""
        # 计算基础数量
        base_quantity = amount / price

        # 调整到step_size的整数倍（向上取整，确保金额不低于设置值）
        step_size = self.config.step_size
        adjusted_quantity = (base_quantity / step_size).quantize(Decimal('1'), rounding=ROUND_UP) * step_size

        # 验证调整后的金额是否满足最小金额要求
        adjusted_amount = adjusted_quantity * price
        if adjusted_amount < self.config.min_notional:
            # 如果不满足，再增加一个step_size
            adjusted_quantity += step_size
            adjusted_amount = adjusted_quantity * price
            logger.debug(f"金额不足，增加数量以满足最小金额要求")

        result = adjusted_quantity.quantize(Decimal(f'1e-{self.config.quantity_precision}'))
        logger.debug(f"金额计算: 目标金额={amount}, 价格={price}, 基础数量={base_quantity}, 调整后数量={result}, 实际金额={adjusted_amount}")

        return result

    def get_order(self, order_id: int) -> Dict:
        """查询订单"""
        logger.debug(f"查询订单状态: {order_id}")
        return self._request('GET', '/api/v1/order', signed=True,
                           symbol=self.config.symbol, orderId=order_id)

    def cancel_order(self, order_id: int) -> Dict:
        """取消订单"""
        logger.info(f"取消订单: {order_id}")
        return self._request('DELETE', '/api/v1/order', signed=True,
                           symbol=self.config.symbol, orderId=order_id)

    def cancel_all_orders(self) -> Dict:
        """取消所有订单"""
        logger.info("取消所有订单")
        return self._request('DELETE', '/api/v1/allOpenOrders', signed=True,
                           symbol=self.config.symbol)

    def get_account_balance(self) -> List[Dict]:
        """获取账户余额"""
        logger.debug("获取账户余额")
        response = self._request('GET', '/api/v1/account', signed=True)
        return response.get('balances', [])

    def get_open_orders(self) -> List[Dict]:
        """获取当前挂单"""
        logger.debug("获取当前挂单")
        return self._request('GET', '/api/v1/openOrders', signed=True, symbol=self.config.symbol)

    def get_user_trades(self, start_time: int = None, end_time: int = None) -> List[Dict]:
        """获取用户成交历史"""
        logger.debug("获取用户成交历史")
        params = {'symbol': self.config.symbol}
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        return self._request('GET', '/api/v1/userTrades', signed=True, **params)

    def get_daily_volume(self) -> Decimal:
        """获取当日成交额（基于UTC时间）"""
        try:
            logger.info("获取当日成交额（UTC时间统计）...")
    
            # 使用UTC时间
            utc_now = datetime.utcnow()
            utc_today_midnight = datetime(utc_now.year, utc_now.month, utc_now.day, 0, 0, 0, 0)
    
            # 将UTC时间转换为时间戳（毫秒）
            utc_start_timestamp = int(utc_today_midnight.timestamp() * 1000)
            current_timestamp = int(time.time() * 1000)
    
            logger.info(f"UTC统计时间范围: {utc_today_midnight} 至 {utc_now}")
    
            # 查询今日（UTC）的成交历史
            trades = self.get_user_trades(start_time=utc_start_timestamp, end_time=current_timestamp)
    
            total_amount = Decimal('0')
    
            if isinstance(trades, list):
                for trade in trades:
                    quote_qty = Decimal(trade.get('quoteQty', '0'))
    
                    if quote_qty > 0:
                        total_amount += quote_qty
                    else:
                        # 备用计算逻辑
                        qty = Decimal(trade.get('qty', '0'))
                        price = Decimal(trade.get('price', '0'))
                        if price > 0 and qty > 0:
                            total_amount += qty * price
    
                logger.info(f"账户 {self.account.name} 今日（UTC）成交额: {total_amount}")
            else:
                logger.warning(f"获取成交记录返回格式异常: {trades}")
                total_amount = Decimal('0')
    
            return total_amount
    
        except Exception as e:
            logger.error(f"获取当日成交额失败: {e}")
            return Decimal('0')


class SelfTradeExecutor:
    """现货自成交执行器"""
    def __init__(self, config: TradingConfig, account1_client: AsterDexSpotAPIClient, account2_client: AsterDexSpotAPIClient):
        self.config = config
        self.account1_client = account1_client
        self.account2_client = account2_client
        logger.info("现货自成交执行器初始化完成")

    def place_simultaneous_orders(self, price: Decimal, quantity: Decimal) -> Tuple[bool, Decimal, Decimal, Decimal, Decimal]:
        """同时放置买卖订单进行自成交
        返回: (是否成功, 买入成交数量, 卖出成交数量, 买入均价, 卖出均价)
        """
        try:
            logger.info(f"准备同时下自成交单: 价格={price}, 数量={quantity}")
            
            # 调整价格和数量
            adjusted_price = self.account1_client.adjust_to_tick_size(price)
            adjusted_quantity = self.account1_client.adjust_to_step_size(quantity)
            
            logger.info(f"调整后参数: 价格={adjusted_price}, 数量={adjusted_quantity}")

            # 创建两个线程同时下单
            buy_order_result = None
            sell_order_result = None
            buy_error = None
            sell_error = None

            def place_buy_order():
                nonlocal buy_order_result, buy_error
                try:
                    logger.info(f"账户1下买单: BUY {adjusted_quantity} @ {adjusted_price}")
                    buy_order_result = self.account1_client.place_order('BUY', adjusted_price, adjusted_quantity)
                    logger.info(f"账户1买单成功: {buy_order_result}")
                except Exception as e:
                    buy_error = e
                    logger.error(f"账户1买单失败: {e}")

            def place_sell_order():
                nonlocal sell_order_result, sell_error
                try:
                    logger.info(f"账户2下卖单: SELL {adjusted_quantity} @ {adjusted_price}")
                    sell_order_result = self.account2_client.place_order('SELL', adjusted_price, adjusted_quantity)
                    logger.info(f"账户2卖单成功: {sell_order_result}")
                except Exception as e:
                    sell_error = e
                    logger.error(f"账户2卖单失败: {e}")

            # 同时启动两个线程下单
            buy_thread = threading.Thread(target=place_buy_order)
            sell_thread = threading.Thread(target=place_sell_order)
            
            buy_thread.start()
            sell_thread.start()
            
            # 等待两个线程完成
            buy_thread.join()
            sell_thread.join()

            # 检查下单结果
            if buy_error or sell_error:
                logger.error(f"下单失败: 买单错误={buy_error}, 卖单错误={sell_error}")
                # 取消已成功的订单
                if buy_order_result:
                    self.account1_client.cancel_order(buy_order_result['orderId'])
                if sell_order_result:
                    self.account2_client.cancel_order(sell_order_result['orderId'])
                return False, Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')

            # 等待订单成交
            logger.info("等待自成交订单成交...")
            time.sleep(2)

            # 检查成交情况
            buy_order_status = self.account1_client.get_order(buy_order_result['orderId'])
            sell_order_status = self.account2_client.get_order(sell_order_result['orderId'])

            buy_executed = Decimal(buy_order_status.get('executedQty', '0'))
            sell_executed = Decimal(sell_order_status.get('executedQty', '0'))
            buy_avg_price = Decimal(buy_order_status.get('avgPrice', '0')) if buy_order_status.get('avgPrice') else Decimal('0')
            sell_avg_price = Decimal(sell_order_status.get('avgPrice', '0')) if sell_order_status.get('avgPrice') else Decimal('0')

            logger.info(f"成交情况: 买单成交={buy_executed}, 卖单成交={sell_executed}")
            logger.info(f"成交均价: 买单均价={buy_avg_price}, 卖单均价={sell_avg_price}")

            return True, buy_executed, sell_executed, buy_avg_price, sell_avg_price

        except Exception as e:
            logger.error(f"自成交下单失败: {e}")
            return False, Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')

    def execute_self_trade_with_adjustment(self, price: Decimal, quantity: Decimal) -> Tuple[bool, Decimal, Decimal, Decimal, Decimal]:
        """执行自成交，如果未完全成交则使用市价单完成剩余部分
        返回: (是否完全成交, 买入总数量, 卖出总数量, 买入均价, 卖出均价)
        """
        logger.info(f"开始执行自成交: 目标价格={price}, 目标数量={quantity}")
    
        try:
            # 第一步：先在中间价下挂单
            success, buy_executed, sell_executed, buy_price, sell_price = self.place_simultaneous_orders(price, quantity)
    
            if not success:
                logger.error("初始下单失败")
                return False, Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')
    
            # 等待一段时间让订单成交
            logger.info("等待订单成交...")
            time.sleep(3)
    
            # 检查成交状态
            buy_remaining = quantity - buy_executed
            sell_remaining = quantity - sell_executed
    
            total_buy_qty = buy_executed
            total_sell_qty = sell_executed
            total_buy_cost = buy_executed * buy_price if buy_executed > 0 else Decimal('0')
            total_sell_cost = sell_executed * sell_price if sell_executed > 0 else Decimal('0')
    
            # 如果还有剩余未成交，使用市价单完成
            market_orders_placed = False
            if buy_remaining > self.config.step_size or sell_remaining > self.config.step_size:
                logger.info(f"检测到未成交部分，使用市价单完成: 买单剩余={buy_remaining}, 卖单剩余={sell_remaining}")
    
                # 确保剩余数量是最小交易单位的整数倍
                market_buy_qty = self.account1_client.adjust_to_step_size(buy_remaining) if buy_remaining > 0 else Decimal('0')
                market_sell_qty = self.account2_client.adjust_to_step_size(sell_remaining) if sell_remaining > 0 else Decimal('0')
    
                # 取消原来的挂单
                try:
                    self.account1_client.cancel_all_orders()
                    self.account2_client.cancel_all_orders()
                    logger.info("已取消所有挂单，准备下市价单")
                except Exception as e:
                    logger.error(f"取消挂单失败: {e}")
    
                # 下市价单完成剩余部分
                if market_buy_qty > 0:
                    try:
                        logger.info(f"账户1下市价买单完成剩余部分: {market_buy_qty}")
                        buy_market_result = self.account1_client.place_market_order('BUY', market_buy_qty)
                        market_orders_placed = True
                        time.sleep(2)
    
                        # 获取市价单成交详情
                        market_buy_status = self.account1_client.get_order(buy_market_result['orderId'])
                        market_buy_qty_executed = Decimal(market_buy_status.get('executedQty', '0'))
                        market_buy_price = Decimal(market_buy_status.get('avgPrice', '0')) if market_buy_status.get('avgPrice') else Decimal('0')
    
                        total_buy_qty += market_buy_qty_executed
                        total_buy_cost += market_buy_qty_executed * market_buy_price
    
                        logger.info(f"市价买单成交: {market_buy_qty_executed} @ 均价{market_buy_price}")
                    except Exception as e:
                        logger.error(f"市价买单失败: {e}")
    
                if market_sell_qty > 0:
                    try:
                        logger.info(f"账户2下市价卖单完成剩余部分: {market_sell_qty}")
                        sell_market_result = self.account2_client.place_market_order('SELL', market_sell_qty)
                        market_orders_placed = True
                        time.sleep(2)
    
                        # 获取市价单成交详情
                        market_sell_status = self.account2_client.get_order(sell_market_result['orderId'])
                        market_sell_qty_executed = Decimal(market_sell_status.get('executedQty', '0'))
                        market_sell_price = Decimal(market_sell_status.get('avgPrice', '0')) if market_sell_status.get('avgPrice') else Decimal('0')
    
                        total_sell_qty += market_sell_qty_executed
                        total_sell_cost += market_sell_qty_executed * market_sell_price
    
                        logger.info(f"市价卖单成交: {market_sell_qty_executed} @ 均价{market_sell_price}")
                    except Exception as e:
                        logger.error(f"市价卖单失败: {e}")
    
            # 计算平均价格
            avg_buy_price = total_buy_cost / total_buy_qty if total_buy_qty > 0 else Decimal('0')
            avg_sell_price = total_sell_cost / total_sell_qty if total_sell_qty > 0 else Decimal('0')
    
            # 检查是否完全成交
            is_fully_filled = total_buy_qty >= quantity and total_sell_qty >= quantity
    
            if is_fully_filled:
                logger.info(f"自成交完全成功: 买入{total_buy_qty}@均价{avg_buy_price}, 卖出{total_sell_qty}@均价{avg_sell_price}")
            else:
                logger.warning(f"自成交部分成功: 买入{total_buy_qty}/{quantity}, 卖出{total_sell_qty}/{quantity}")
    
            return is_fully_filled, total_buy_qty, total_sell_qty, avg_buy_price, avg_sell_price
    
        except Exception as e:
            logger.error(f"执行自成交失败: {e}")
            return False, Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')


class SpotSelfTradingBot:
    def __init__(self, accounts_config_path: str):
        self.accounts_config_path = accounts_config_path
        logger.info(f"初始化现货自交易机器人，配置文件: {accounts_config_path}")
        self.config = self._load_config()
        self.account1_client = AsterDexSpotAPIClient(self.config.trading, self.config.account1)
        self.account2_client = AsterDexSpotAPIClient(self.config.trading, self.config.account2)
        self.self_trade_executor = SelfTradeExecutor(self.config.trading, self.account1_client, self.account2_client)
        self.stability_monitor = PriceStabilityMonitor(
            self.config.trading.stability_period,
            self.config.trading.tick_size,
            self.config.trading.sampling_rate
        )
        self.is_running = False
        self.account1_daily_volume = Decimal('0')
        self.account2_daily_volume = Decimal('0')
        logger.info("现货自交易机器人初始化完成")

    def _load_config(self):
        """加载配置文件"""
        logger.info("开始加载配置文件...")

        # 加载账号配置
        accounts_config, trading_config_path = ConfigLoader.load_accounts_config(self.accounts_config_path)

        # 加载交易配置
        trading_config_data = ConfigLoader.load_trading_config(trading_config_path)

        # 设置日志
        ConfigLoader.setup_logging(
            trading_config_data.get('log_level', 'INFO'),
            trading_config_data.get('log_file', 'spot_auto_trading.log')
        )

        @dataclass
        class Config:
            account1: AccountConfig
            account2: AccountConfig
            trading: TradingConfig

        # 创建账号配置对象
        account1_data = accounts_config['account1']
        account2_data = accounts_config['account2']

        logger.info(f"账号1名称: {account1_data['name']}")
        logger.info(f"账号2名称: {account2_data['name']}")

        # 将字符串转换为Decimal
        trading_config_data['order_amount'] = Decimal(trading_config_data['order_amount'])
        trading_config_data['price_gap_threshold'] = Decimal(trading_config_data['price_gap_threshold'])
        trading_config_data['tick_size'] = Decimal(trading_config_data['tick_size'])
        trading_config_data['step_size'] = Decimal(trading_config_data['step_size'])
        trading_config_data['min_notional'] = Decimal(trading_config_data['min_notional'])
        trading_config_data['daily_volume_target'] = Decimal(trading_config_data['daily_volume_target'])

        logger.info("配置文件解析完成")
        return Config(
            account1=AccountConfig(**account1_data),
            account2=AccountConfig(**account2_data),
            trading=TradingConfig(**trading_config_data)
        )

    def analyze_order_book(self, order_book: Dict) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """分析订单簿，返回买卖一档价格"""
        try:
            best_bid = Decimal(order_book['bids'][0][0]) if order_book['bids'] else None
            best_ask = Decimal(order_book['asks'][0][0]) if order_book['asks'] else None
            logger.debug(f"订单簿分析: 买一={best_bid}, 卖一={best_ask}")
            return best_bid, best_ask
        except (IndexError, KeyError) as e:
            logger.error(f"分析订单簿失败: {e}")
            logger.debug(f"订单簿数据: {order_book}")
            return None, None

    def calculate_mid_price(self, bid: Decimal, ask: Decimal) -> Decimal:
        """计算中间价格"""
        mid_price = (bid + ask) / 2
        logger.debug(f"计算中间价: ({bid} + {ask}) / 2 = {mid_price}")
        return mid_price

    def has_sufficient_gap(self, bid: Decimal, ask: Decimal) -> bool:
        """检查买卖价差是否足够（基于tick数）"""
        if bid is None or ask is None:
            logger.warning("买卖价格为空，无法计算价差")
            return False

        gap_ticks = (ask - bid) / self.config.trading.tick_size
        meets_threshold = gap_ticks >= self.config.trading.price_gap_threshold

        logger.info(f"价差分析: 买卖价差={ask - bid:.6f}, tick数={gap_ticks:.1f}, 阈值={self.config.trading.price_gap_threshold}, 满足条件={meets_threshold}")

        return meets_threshold

    def calculate_order_quantity(self, price: Decimal) -> Decimal:
        """根据订单金额计算数量"""
        quantity = self.account1_client.calculate_quantity_from_amount(
            self.config.trading.order_amount, price
        )
        logger.info(f"计算订单数量: 价格={price}, 订单金额={self.config.trading.order_amount}, 数量={quantity}")
        return quantity

    def execute_self_trade(self, bid_price: Decimal, ask_price: Decimal) -> bool:
        """执行自交易开仓（现货不需要平仓概念，只需要执行自成交）"""
        try:
            logger.info("开始执行现货自交易...")
            # 计算中间价作为交易价格
            trade_price = self.calculate_mid_price(bid_price, ask_price)
            trade_price = self.account1_client.adjust_to_tick_size(trade_price)
            
            # 计算交易数量
            quantity = self.calculate_order_quantity(trade_price)
            
            logger.info(f"现货自交易参数: 价格={trade_price}, 数量={quantity}")

            # 执行自成交
            success, buy_qty, sell_qty, buy_price, sell_price = self.self_trade_executor.execute_self_trade_with_adjustment(
                trade_price, quantity
            )

            if success and buy_qty > 0 and sell_qty > 0:
                buy_amount = buy_qty * buy_price
                sell_amount = sell_qty * sell_price
                cost_difference = buy_amount - sell_amount
                logger.info(f"现货自交易成功: 买入{buy_qty}@均价{buy_price}, 卖出{sell_qty}@均价{sell_price}, 成本差异={cost_difference}")
                
                # 检查资产转移是否正确
                self.check_balances()
                
                return True
            else:
                logger.error("现货自交易失败")
                return False

        except Exception as e:
            logger.error(f"执行现货自交易失败: {e}")
            return False

    def check_balances(self):
        """检查账户余额"""
        try:
            logger.info("检查账户余额...")
            acc1_balance = self.account1_client.get_account_balance()
            acc2_balance = self.account2_client.get_account_balance()

            logger.info("=== 账户余额检查 ===")

            # 显示余额
            logger.info("账户1余额:")
            for asset in acc1_balance:
                if Decimal(asset['free']) > 0 or Decimal(asset['locked']) > 0:
                    logger.info(f"  {asset['asset']}: 可用={asset['free']}, 冻结={asset['locked']}")

            logger.info("账户2余额:")
            for asset in acc2_balance:
                if Decimal(asset['free']) > 0 or Decimal(asset['locked']) > 0:
                    logger.info(f"  {asset['asset']}: 可用={asset['free']}, 冻结={asset['locked']}")

        except Exception as e:
            logger.error(f"检查账户余额失败: {e}")

    def check_daily_volume_target(self) -> bool:
        """检查是否达到每日目标成交量
        返回: True=已达到目标，False=未达到目标
        """
        try:
            # 获取当日成交量
            self.account1_daily_volume = self.account1_client.get_daily_volume()
            self.account2_daily_volume = self.account2_client.get_daily_volume()
            
            daily_target = self.config.trading.daily_volume_target
            
            logger.info(f"成交量检查: 账户1={self.account1_daily_volume}, 账户2={self.account2_daily_volume}, 目标={daily_target}")
            
            # 如果目标为0，表示无限制
            if daily_target == Decimal('0'):
                logger.info("每日目标成交量为0（无限制）")
                return False
            
            # 检查两个账户是否都达到目标
            if self.account1_daily_volume >= daily_target and self.account2_daily_volume >= daily_target:
                logger.info(f"✅ 两个账户都已达到每日目标成交量: 账户1={self.account1_daily_volume}, 账户2={self.account2_daily_volume}, 目标={daily_target}")
                return True
            else:
                # 显示剩余量
                remaining1 = max(Decimal('0'), daily_target - self.account1_daily_volume)
                remaining2 = max(Decimal('0'), daily_target - self.account2_daily_volume)
                logger.info(f"📊 成交量进度: 账户1还需{remaining1}, 账户2还需{remaining2}")
                return False
                
        except Exception as e:
            logger.error(f"检查每日成交量失败: {e}")
            # 如果检查失败，继续执行程序
            return False

    def run(self):
        """运行现货自交易机器人 - 单次执行模式"""
        self.is_running = True
        logger.info("开始现货自交易机器人 - 单次执行模式")
        logger.info(f"交易配置: 交易对={self.config.trading.symbol}, 订单金额={self.config.trading.order_amount}, "
                   f"价差阈值={self.config.trading.price_gap_threshold}tick, "
                   f"tick大小={self.config.trading.tick_size}, step大小={self.config.trading.step_size}, "
                   f"最小金额={self.config.trading.min_notional}, "
                   f"每日目标成交量={self.config.trading.daily_volume_target}")

        try:
            # 检查交易所信息
            logger.info("检查交易所信息...")
            exchange_info = self.account1_client.get_exchange_info()
            symbol_info = next((s for s in exchange_info.get('symbols', [])
                              if s['symbol'] == self.config.trading.symbol), {})
            logger.info(f"交易对状态: {symbol_info.get('status', '未知')}")

            # 检查账户余额
            self.check_balances()

            while self.is_running:
                # 在开始前检查每日目标成交量
                logger.info("=== 检查每日成交量 ===")
                if self.check_daily_volume_target():
                    logger.info("两个账户都已达到每日目标成交量，程序退出")
                    break 
                
                trade_executed = False

                # 等待稳定价差并执行自交易
                logger.info("=== 等待自交易时机 ===")
                while self.is_running and not trade_executed:
                    try:
                        # 获取订单簿
                        logger.debug("获取订单簿数据...")
                        order_book = self.account1_client.get_order_book()
                        bid_price, ask_price = self.analyze_order_book(order_book)

                        if bid_price and ask_price:
                            gap_ticks = (ask_price - bid_price) / self.config.trading.tick_size

                            logger.info(f"市场状态 - 买一: {bid_price}, 卖一: {ask_price}, 价差: {gap_ticks:.1f}tick")

                            # 更新稳定性监测
                            is_stable, has_gap = self.stability_monitor.update_prices(bid_price, ask_price)

                            if is_stable and has_gap and self.has_sufficient_gap(bid_price, ask_price):
                                logger.info("检测到稳定价差，执行现货自交易...")

                                # 执行现货自交易
                                if self.execute_self_trade(bid_price, ask_price):
                                    trade_executed = True
                                    logger.info("现货自交易执行成功")
                                else:
                                    logger.error("现货自交易执行失败，继续监控...")
                            else:
                                if not is_stable:
                                    logger.info("价格不稳定，继续监控交易时机...")
                                elif not has_gap:
                                    logger.info("价差不足(小于1个tick)，继续监控交易时机...")
                                else:
                                    logger.info("价差不满足阈值条件，继续监控交易时机...")

                        # 等待采样间隔
                        logger.debug(f"等待下次采样... ({self.config.trading.sampling_rate}秒)")
                        time.sleep(self.config.trading.sampling_rate)

                    except Exception as e:
                        logger.error(f"交易执行错误: {e}")
                        time.sleep(5)

                time.sleep(self.config.trading.wait_time)
                
                # 最终检查成交量
                logger.info("=== 最终成交量检查 ===")
                if self.check_daily_volume_target():
                    logger.info("已达到每日目标成交量")
                else:
                    logger.info("未达到每日目标成交量，但已完成一次交易")
                
            # 程序完成
            if trade_executed:
                logger.info("✅ 程序执行完成：成功完成一次现货自交易")
            else:
                logger.error("❌ 程序结束：交易未完成")

        except KeyboardInterrupt:
            logger.info("用户中断程序")
        except Exception as e:
            logger.error(f"程序运行错误: {e}")
        finally:
            self.is_running = False
            # 取消所有挂单
            try:
                self.account1_client.cancel_all_orders()
                self.account2_client.cancel_all_orders()
            except:
                pass
            logger.info("现货自交易机器人已停止")


def main():
    parser = argparse.ArgumentParser(description='AsterDex现货自交易机器人')
    parser.add_argument('--config', type=str, required=True, help='账号配置文件路径')
    parser.add_argument('--check-balance', action='store_true', help='只检查余额不开始交易')

    args = parser.parse_args()

    logger.info(f"启动参数: config={args.config}, check-balance={args.check_balance}")

    try:
        bot = SpotSelfTradingBot(args.config)

        if args.check_balance:
            logger.info("执行账户检查模式")
            bot.check_balances()
        else:
            logger.info("执行交易模式")
            bot.run()

    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()