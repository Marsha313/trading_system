import requests
import hashlib
import hmac
import time
import json
import yaml
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
from datetime import datetime, timedelta
import os
import argparse
from dataclasses import dataclass

@dataclass
class AccountConfig:
    """账户配置类"""
    name: str
    api_key: str
    secret_key: str
    enabled: bool = True

class AsterDexMultiAccountSpotAnalytics:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.accounts = self._load_accounts_config()
        self.settings = self._load_settings()
        self.base_url = "https://sapi.asterdex.com"  # 改为现货API

    def _load_accounts_config(self) -> List[AccountConfig]:
        """从YAML配置文件加载多个账户配置"""
        if not os.path.exists(self.config_file):
            self._create_sample_config()
            raise ValueError(f"配置文件 {self.config_file} 不存在，已创建示例配置文件，请填写您的API密钥")

        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        accounts = []

        # 加载账户配置
        if 'accounts' in config:
            for account_config in config['accounts']:
                account = AccountConfig(
                    name=account_config.get('name', '未命名账户'),
                    api_key=account_config.get('api_key', ''),
                    secret_key=account_config.get('secret_key', ''),
                    enabled=account_config.get('enabled', True)
                )
                if account.api_key and account.secret_key and account.enabled:
                    accounts.append(account)

        # 向后兼容：如果使用旧的配置格式
        elif 'api' in config and config['api'].get('api_key') and config['api'].get('secret_key'):
            account = AccountConfig(
                name='默认账户',
                api_key=config['api']['api_key'],
                secret_key=config['api']['secret_key'],
                enabled=True
            )
            accounts.append(account)

        if not accounts:
            raise ValueError("没有找到有效的账户配置")

        print(f"加载了 {len(accounts)} 个账户配置")
        for account in accounts:
            print(f"  - {account.name} ({'启用' if account.enabled else '禁用'})")

        return accounts

    def _load_settings(self) -> Dict:
        """加载设置"""
        if not os.path.exists(self.config_file):
            return {}

        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config.get('settings', {})

    def _create_sample_config(self):
        """创建示例配置文件"""
        sample_config = {
            'accounts': [
                {
                    'name': '主账户',
                    'api_key': '您的API_KEY_1',
                    'secret_key': '您的SECRET_KEY_1',
                    'enabled': True
                },
                {
                    'name': '子账户1',
                    'api_key': '您的API_KEY_2',
                    'secret_key': '您的SECRET_KEY_2',
                    'enabled': True
                }
            ],
            'settings': {
                'default_period_days': 7,
                'max_trades_limit': 500,
                'show_account_balance': True,
                'compare_performance': True,
                'daily_volume_only': False,
                'daily_volume_timezone': 'UTC',
                'minutes_interval': None  # 新增：分钟间隔配置，None表示不使用
            }
        }

        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True, indent=2)

        print(f"已创建示例配置文件: {self.config_file}")
        print("请编辑配置文件并填写您的API密钥")

    def _get_time_range(self, days: int = None, minutes: int = None) -> Tuple[int, int]:
        """获取时间范围"""
        # 优先使用分钟间隔配置
        minutes_interval = minutes or self.settings.get('minutes_interval')
        
        if minutes_interval:
            # 使用分钟间隔统计
            current_time = int(time.time() * 1000)
            start_time = current_time - (minutes_interval * 60 * 1000)
            
            start_dt = datetime.fromtimestamp(start_time/1000)
            end_dt = datetime.fromtimestamp(current_time/1000)
            
            print(f"⏰ 统计最近{minutes_interval}分钟交易数据")
            print(f"   时间范围: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return start_time, current_time
        
        daily_volume_only = self.settings.get('daily_volume_only', False)

        if daily_volume_only:
            # 仅统计当天交易量
            current_time = int(time.time() * 1000)

            # 获取当天UTC 0点的时间戳
            utc_now = datetime.utcnow()
            utc_today_start = datetime(utc_now.year, utc_now.month, utc_now.day)
            today_start_time = int(utc_today_start.timestamp() * 1000)

            print(f"📅 仅统计当天交易量 (UTC时间)")
            print(f"   统计时间范围: {utc_today_start.strftime('%Y-%m-%d %H:%M:%S UTC')} 到 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

            return today_start_time, current_time
        else:
            # 正常统计指定天数的数据
            if days is None:
                days = self.settings.get('default_period_days', 7)

            current_time = int(time.time() * 1000)
            start_time = current_time - (days * 24 * 60 * 60 * 1000)

            start_dt = datetime.fromtimestamp(start_time/1000)
            end_dt = datetime.fromtimestamp(current_time/1000)

            print(f"📊 统计最近{days}天交易数据")
            print(f"   时间范围: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")

            return start_time, current_time

    def _generate_signature(self, secret_key: str, params: Dict) -> str:
        """生成HMAC SHA256签名"""
        query_string = urlencode(params)
        signature = hmac.new(
            secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _signed_request(self, account: AccountConfig, method: str, endpoint: str, params: Dict = None) -> Dict:
        """发送签名请求"""
        if params is None:
            params = {}

        # 创建会话并设置API密钥
        session = requests.Session()
        session.headers.update({
            'X-MBX-APIKEY': account.api_key
        })

        # 添加必要参数
        current_time = int(time.time() * 1000)
        params['timestamp'] = current_time
        params['recvWindow'] = 5000

        # 生成签名
        params['signature'] = self._generate_signature(account.secret_key, params)

        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == 'GET':
                response = session.get(url, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = session.post(url, data=params, timeout=30)
            elif method.upper() == 'DELETE':
                response = session.delete(url, data=params, timeout=30)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"账户 {account.name} 请求失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            raise

    def get_account_trades(self, account: AccountConfig, symbol: str = None, start_time: int = None,
                          end_time: int = None, limit: int = None) -> List[Dict]:
        """获取账户交易记录 - 现货版本"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        if limit:
            params['limit'] = min(limit, 1000)
        else:
            params['limit'] = min(self.settings.get('max_trades_limit', 500), 1000)

        return self._signed_request(account, 'GET', '/api/v1/userTrades', params)

    def get_account_info(self, account: AccountConfig) -> Dict:
        """获取账户信息 - 现货版本"""
        return self._signed_request(account, 'GET', '/api/v1/account')

    def get_open_orders(self, account: AccountConfig, symbol: str = None) -> List[Dict]:
        """获取当前委托订单 - 现货版本"""
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._signed_request(account, 'GET', '/api/v1/openOrders', params)

    def get_ticker_price(self, symbol: str = None) -> Dict:
        """获取最新价格"""
        params = {}
        if symbol:
            params['symbol'] = symbol

        session = requests.Session()
        url = f"{self.base_url}/api/v1/ticker/price"
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_all_ticker_prices(self) -> Dict:
        """获取所有交易对的最新价格"""
        session = requests.Session()
        url = f"{self.base_url}/api/v1/ticker/price"
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def test_account_connectivity(self, account: AccountConfig) -> bool:
        """测试账户连接性"""
        try:
            session = requests.Session()
            session.headers.update({'X-MBX-APIKEY': account.api_key})
            url = f"{self.base_url}/api/v1/ping"
            response = session.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False

    def calculate_account_performance(self, account: AccountConfig, days: int = None, minutes: int = None) -> Dict:
        """计算单个账户的交易表现 - 现货版本"""
        # 获取时间范围（支持分钟间隔）
        start_time, end_time = self._get_time_range(days, minutes)

        print(f"分析账户 {account.name} 的现货交易数据...")

        try:
            # 测试连接
            if not self.test_account_connectivity(account):
                print(f"账户 {account.name} 连接测试失败")
                return {}

            # 1. 获取交易记录
            trades = self.get_account_trades(
                account,
                start_time=start_time,
                end_time=end_time
            )

            print(f"账户 {account.name} 获取到 {len(trades)} 条交易记录")

            # 2. 获取账户余额和持仓信息
            account_info = self.get_account_info(account)
            
            # 计算总余额和持仓
            balance_analysis = self._analyze_account_balance(account_info)
            total_balance = balance_analysis['total_balance_usdt']

            # 3. 获取当前委托订单
            open_orders = self.get_open_orders(account)
            print(f"账户 {account.name} 获取到 {len(open_orders)} 个当前委托订单")

            # 分析交易表现
            analytics = self._analyze_trading_performance(trades, balance_analysis, open_orders, start_time, end_time)
            analytics['account_balance'] = total_balance
            analytics['account_name'] = account.name

            return analytics

        except Exception as e:
            print(f"分析账户 {account.name} 交易数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _analyze_account_balance(self, account_info: Dict) -> Dict:
        """分析账户余额和持仓"""
        balances = account_info.get('balances', [])
        
        # 获取所有交易对的最新价格
        try:
            ticker_prices = self.get_all_ticker_prices()
            price_dict = {}
            for ticker in ticker_prices:
                price_dict[ticker['symbol']] = float(ticker['price'])
        except Exception as e:
            print(f"获取价格数据失败: {e}")
            price_dict = {}

        total_balance_usdt = 0.0
        positions = []
        active_positions_count = 0

        for balance in balances:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked

            # 只统计有余额的资产
            if total > 0:
                # 计算USDT价值
                usdt_value = 0.0
                
                if asset == 'USDT':
                    usdt_value = total
                else:
                    # 查找对应的交易对价格
                    symbol = f"{asset}USDT"
                    if symbol in price_dict:
                        usdt_value = total * price_dict[symbol]
                    else:
                        # 尝试反向查找
                        symbol_reverse = f"USDT{asset}"
                        if symbol_reverse in price_dict:
                            usdt_value = total / price_dict[symbol_reverse]
                
                total_balance_usdt += usdt_value

                # 记录持仓信息
                if usdt_value > 1:  # 只记录价值超过1 USDT的持仓
                    position_data = {
                        'asset': asset,
                        'total_amount': total,
                        'free_amount': free,
                        'locked_amount': locked,
                        'usdt_value': usdt_value
                    }
                    positions.append(position_data)
                    active_positions_count += 1

        # 按价值排序
        positions.sort(key=lambda x: x['usdt_value'], reverse=True)

        return {
            'total_balance_usdt': total_balance_usdt,
            'active_positions_count': active_positions_count,
            'total_position_value': total_balance_usdt,
            'positions': positions,
            'balances': balances
        }

    def _analyze_trading_performance(self, trades: List[Dict], balance_analysis: Dict, open_orders: List[Dict], start_time: int, end_time: int) -> Dict:
        """分析交易表现 - 现货版本"""
        time_diff = end_time - start_time
        minutes_diff = time_diff / (60 * 1000)
        hours_diff = time_diff / (60 * 60 * 1000)
        days_diff = time_diff / (24 * 60 * 60 * 1000)

        # 根据时间间隔选择合适的单位
        if minutes_diff < 60:
            period_str = f"{minutes_diff:.1f}分钟"
        elif hours_diff < 24:
            period_str = f"{hours_diff:.1f}小时"
        else:
            period_str = f"{days_diff:.1f}天"

        # 初始化统计变量
        stats = {
            'period': period_str,
            'period_minutes': minutes_diff,
            'start_time': start_time,
            'end_time': end_time,
            'volume_analysis': self._analyze_volume_spot(trades),
            'commission_analysis': self._analyze_commission_spot(trades),
            'pnl_analysis': self._analyze_pnl_spot(trades),
            'position_analysis': balance_analysis,  # 使用余额分析作为持仓分析
            'order_analysis': self._analyze_open_orders_spot(open_orders),
            'efficiency_analysis': {}  # 新增：效率分析
        }

        # 获取关键数据
        total_turnover = stats['volume_analysis']['total_turnover']
        total_commission = stats['commission_analysis']['total_commission']
        realized_pnl = stats['pnl_analysis']['realized_pnl']
        
        # 计算效率：(盈亏 + 手续费) / 手续费
        if total_commission != 0:
            efficiency = (realized_pnl + total_commission) / total_commission
        else:
            efficiency = 0

        stats['efficiency_analysis'] = {
            'total_commission': total_commission,
            'realized_pnl': realized_pnl,
            'efficiency_ratio': efficiency,
            'cost_pnl_total': realized_pnl + total_commission
        }

        return stats

    def _analyze_volume_spot(self, trades: List[Dict]) -> Dict:
        """分析成交量 - 现货版本"""
        if not trades:
            return self._get_empty_volume_stats()

        total_volume = 0.0
        total_turnover = 0.0
        total_trades = len(trades)
        symbols_traded = set()
        trades_by_symbol = {}
        buy_volume = 0.0
        sell_volume = 0.0

        for trade in trades:
            symbol = trade['symbol']
            quantity = float(trade['qty'])
            quote_quantity = float(trade['quoteQty'])
            side = trade['side']

            # 所有交易都计入
            total_volume += quantity
            total_turnover += quote_quantity

            # 统计买卖方向
            if side == 'BUY':
                buy_volume += quantity
            elif side == 'SELL':
                sell_volume += quantity

            symbols_traded.add(symbol)

            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = self._get_empty_symbol_stats_spot()

            # 更新统计
            self._update_symbol_stats_spot(trades_by_symbol[symbol], quantity, quote_quantity, side)

        return {
            'total_volume': total_volume,
            'total_turnover': total_turnover,
            'total_trades': total_trades,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'net_volume': buy_volume - sell_volume,
            'avg_trade_size': total_volume / total_trades if total_trades > 0 else 0,
            'avg_trade_turnover': total_turnover / total_trades if total_trades > 0 else 0,
            'symbols_traded_count': len(symbols_traded),
            'symbols_traded': list(symbols_traded),
            'trades_by_symbol': trades_by_symbol,
            'buy_sell_ratio': buy_volume / sell_volume if sell_volume > 0 else 1
        }

    def _get_empty_volume_stats(self) -> Dict:
        """获取空的成交量统计 - 现货版本"""
        return {
            'total_volume': 0, 'total_turnover': 0, 'total_trades': 0,
            'buy_volume': 0, 'sell_volume': 0, 'net_volume': 0,
            'avg_trade_size': 0, 'avg_trade_turnover': 0,
            'symbols_traded_count': 0, 'symbols_traded': [],
            'trades_by_symbol': {}, 'buy_sell_ratio': 1
        }

    def _get_empty_symbol_stats_spot(self) -> Dict:
        """获取空的币对统计 - 现货版本"""
        return {
            'total_volume': 0.0, 'total_turnover': 0.0, 'trade_count': 0,
            'buy_volume': 0.0, 'sell_volume': 0.0, 'net_volume': 0.0,
            'realized_pnl': 0.0, 'total_commission': 0.0
        }

    def _update_symbol_stats_spot(self, stats: Dict, quantity: float, quote_quantity: float, side: str):
        """更新币对统计 - 现货版本"""
        stats['total_volume'] += quantity
        stats['total_turnover'] += quote_quantity
        stats['trade_count'] += 1

        if side == 'BUY':
            stats['buy_volume'] += quantity
            stats['net_volume'] += quantity
        elif side == 'SELL':
            stats['sell_volume'] += quantity
            stats['net_volume'] -= quantity

    def _analyze_commission_spot(self, trades: List[Dict]) -> Dict:
        """分析手续费 - 现货版本"""
        total_commission = 0.0
        commission_by_asset = {}
        commission_by_symbol = {}

        for trade in trades:
            commission = float(trade.get('commission', 0))
            commission_asset = trade.get('commissionAsset', '')
            symbol = trade['symbol']

            total_commission += commission

            if commission_asset:
                if commission_asset not in commission_by_asset:
                    commission_by_asset[commission_asset] = 0.0
                commission_by_asset[commission_asset] += commission

            if symbol not in commission_by_symbol:
                commission_by_symbol[symbol] = 0.0
            commission_by_symbol[symbol] += commission

        return {
            'total_commission': total_commission,
            'commission_by_asset': commission_by_asset,
            'commission_by_symbol': commission_by_symbol
        }

    def _analyze_pnl_spot(self, trades: List[Dict]) -> Dict:
        """分析盈亏 - 现货版本"""
        # 现货交易的盈亏计算比较复杂，需要跟踪成本基础
        # 这里简化处理：通过买卖价差计算近似盈亏
        realized_pnl = 0.0
        pnl_by_symbol = {}
        trade_pairs = {}

        # 按币对分组交易
        for trade in trades:
            symbol = trade['symbol']
            if symbol not in trade_pairs:
                trade_pairs[symbol] = []
            trade_pairs[symbol].append(trade)

        # 计算每个币对的盈亏
        for symbol, symbol_trades in trade_pairs.items():
            # 按时间排序
            symbol_trades.sort(key=lambda x: x['time'])
            
            # 简化计算：使用先进先出法
            buy_queue = []
            symbol_pnl = 0.0
            
            for trade in symbol_trades:
                if trade['side'] == 'BUY':
                    # 记录买入
                    buy_queue.append({
                        'quantity': float(trade['qty']),
                        'price': float(trade['price']),
                        'commission': float(trade.get('commission', 0))
                    })
                else:  # SELL
                    sell_quantity = float(trade['qty'])
                    sell_price = float(trade['price'])
                    sell_commission = float(trade.get('commission', 0))
                    
                    # 匹配买入记录
                    while sell_quantity > 0 and buy_queue:
                        buy_record = buy_queue[0]
                        if buy_record['quantity'] <= sell_quantity:
                            # 完全匹配这个买入记录
                            matched_quantity = buy_record['quantity']
                            cost = buy_record['price'] * matched_quantity
                            revenue = sell_price * matched_quantity
                            pnl = revenue - cost - buy_record['commission'] - sell_commission
                            symbol_pnl += pnl
                            
                            sell_quantity -= matched_quantity
                            buy_queue.pop(0)
                        else:
                            # 部分匹配
                            matched_quantity = sell_quantity
                            cost = buy_record['price'] * matched_quantity
                            revenue = sell_price * matched_quantity
                            pnl = revenue - cost - (buy_record['commission'] * (matched_quantity / buy_record['quantity'])) - sell_commission
                            symbol_pnl += pnl
                            
                            # 更新买入记录
                            buy_record['quantity'] -= matched_quantity
                            buy_record['commission'] *= (1 - matched_quantity / (buy_record['quantity'] + matched_quantity))
                            sell_quantity = 0
            
            realized_pnl += symbol_pnl
            pnl_by_symbol[symbol] = symbol_pnl

        # 计算胜率
        winning_symbols = len([pnl for pnl in pnl_by_symbol.values() if pnl > 0])
        losing_symbols = len([pnl for pnl in pnl_by_symbol.values() if pnl < 0])

        return {
            'realized_pnl': realized_pnl,
            'pnl_by_symbol': pnl_by_symbol,
            'winning_symbols': winning_symbols,
            'losing_symbols': losing_symbols,
            'win_rate': winning_symbols / len(pnl_by_symbol) if pnl_by_symbol else 0
        }

    def _analyze_open_orders_spot(self, open_orders: List[Dict]) -> Dict:
        """分析当前委托订单 - 现货版本"""
        if not open_orders:
            return {
                'total_orders': 0,
                'orders_by_symbol': {},
                'orders_by_type': {},
                'orders_by_side': {},
                'total_order_value': 0.0,
                'orders': []
            }

        orders_by_symbol = {}
        orders_by_type = {}
        orders_by_side = {}
        total_order_value = 0.0

        for order in open_orders:
            symbol = order['symbol']
            order_type = order['type']
            side = order['side']
            quantity = float(order.get('origQty', 0))
            price = float(order.get('price', 0))
            order_value = quantity * price if price > 0 else 0

            total_order_value += order_value

            # 按币对统计
            if symbol not in orders_by_symbol:
                orders_by_symbol[symbol] = 0
            orders_by_symbol[symbol] += 1

            # 按订单类型统计
            if order_type not in orders_by_type:
                orders_by_type[order_type] = 0
            orders_by_type[order_type] += 1

            # 按买卖方向统计
            if side not in orders_by_side:
                orders_by_side[side] = 0
            orders_by_side[side] += 1

        return {
            'total_orders': len(open_orders),
            'orders_by_symbol': orders_by_symbol,
            'orders_by_type': orders_by_type,
            'orders_by_side': orders_by_side,
            'total_order_value': total_order_value,
            'orders': open_orders
        }

    def generate_multi_account_report(self, days: int = None, minutes: int = None) -> Dict:
        """生成多账户综合报告 - 现货版本"""
        start_time, end_time = self._get_time_range(days, minutes)

        print(f"\n正在生成 {len(self.accounts)} 个账户的现货交易分析报告...")

        all_accounts_data = {}
        total_stats = {
            'total_turnover': 0,
            'total_commission': 0,
            'total_realized_pnl': 0,
            'total_net_profit': 0,
            'total_trades': 0,
            'total_open_orders': 0,
            'total_order_value': 0.0,
            'total_active_positions': 0,
            'total_position_value': 0.0,
            'total_efficiency': 0.0
        }

        # 分析每个账户
        for account in self.accounts:
            account_data = self.calculate_account_performance(account, days, minutes)
            if account_data:
                all_accounts_data[account.name] = account_data

                # 累计总统计
                total_stats['total_turnover'] += account_data['volume_analysis']['total_turnover']
                total_stats['total_commission'] += account_data['commission_analysis']['total_commission']
                total_stats['total_realized_pnl'] += account_data['pnl_analysis']['realized_pnl']
                total_stats['total_net_profit'] += (account_data['pnl_analysis']['realized_pnl'] -
                                                  account_data['commission_analysis']['total_commission'])
                total_stats['total_trades'] += account_data['volume_analysis']['total_trades']
                total_stats['total_open_orders'] += account_data['order_analysis']['total_orders']
                total_stats['total_order_value'] += account_data['order_analysis']['total_order_value']
                total_stats['total_active_positions'] += account_data['position_analysis']['active_positions_count']
                total_stats['total_position_value'] += account_data['position_analysis']['total_position_value']
                total_stats['total_efficiency'] += account_data['efficiency_analysis']['efficiency_ratio']

        # 计算平均效率
        if all_accounts_data:
            total_stats['avg_efficiency'] = total_stats['total_efficiency'] / len(all_accounts_data)

        report = {
            'report_period': self._get_report_period_description(start_time, end_time),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_accounts': len(all_accounts_data),
            'total_statistics': total_stats,
            'accounts_data': all_accounts_data,
            'config_used': {
                'period_days': days,
                'minutes_interval': minutes or self.settings.get('minutes_interval'),
                'max_trades_limit': self.settings.get('max_trades_limit', 500),
                'show_account_balance': self.settings.get('show_account_balance', True),
                'compare_performance': self.settings.get('compare_performance', True),
                'daily_volume_only': self.settings.get('daily_volume_only', False),
                'daily_volume_timezone': self.settings.get('daily_volume_timezone', 'UTC')
            }
        }

        return report

    def _get_report_period_description(self, start_time: int, end_time: int) -> str:
        """获取报告周期描述"""
        # 计算时间间隔
        time_diff = end_time - start_time
        minutes_diff = time_diff / (60 * 1000)
        hours_diff = time_diff / (60 * 60 * 1000)
        days_diff = time_diff / (24 * 60 * 60 * 1000)

        if minutes_diff < 60:
            return f"最近{minutes_diff:.0f}分钟"
        elif hours_diff < 24:
            return f"最近{hours_diff:.1f}小时"
        else:
            return f"最近{days_diff:.1f}天"

def display_multi_account_report(report: Dict):
    """显示多账户分析报告 - 现货版本"""
    if not report or not report['accounts_data']:
        print("没有可用的报告数据")
        return

    config_used = report['config_used']
    minutes_interval = config_used.get('minutes_interval')
    daily_volume_only = config_used.get('daily_volume_only', False)

    print("\n" + "="*120)
    if minutes_interval:
        print(f"📊 多账户现货交易分析报告 - 最近{minutes_interval}分钟")
    elif daily_volume_only:
        print("📊 多账户现货当天交易量统计报告")
    else:
        print("📊 多账户现货交易分析报告")
    print("="*120)
    print(f"报告周期: {report['report_period']}")
    print(f"生成时间: {report['generated_at']}")
    print(f"分析账户数量: {report['total_accounts']} 个")

    if minutes_interval:
        print(f"⏰ 统计模式: 最近{minutes_interval}分钟交易数据")
    elif daily_volume_only:
        print(f"📅 统计模式: 仅统计当天交易量 (UTC时间)")

    accounts_data = report['accounts_data']

    # 各账户详细分析
    if config_used.get('show_account_balance', True):
        print(f"\n💰 各账户余额:")
        for account_name, data in accounts_data.items():
            balance = data.get('account_balance', 0)
            print(f"  {account_name}: {balance:,.2f} USDT")

    # 显示每个账户的详细报告
    print(f"\n🔍 各账户详细分析:")
    for account_name, data in accounts_data.items():
        display_single_account_details_spot(account_name, data, minutes_interval, daily_volume_only)

    # 总体统计
    total_stats = report['total_statistics']
    print(f"\n🏆 总体统计:")
    print(f"总交易额: {total_stats['total_turnover']:,.2f} USDT")
    print(f"总手续费: {total_stats['total_commission']:,.2f} USDT")
    print(f"总已实现盈亏: {total_stats['total_realized_pnl']:+,.2f} USDT")
    print(f"总净收益: {total_stats['total_net_profit']:+,.2f} USDT")
    print(f"总交易次数: {total_stats['total_trades']:,} 次")
    print(f"总委托订单: {total_stats['total_open_orders']} 个")
    print(f"总委托价值: {total_stats['total_order_value']:,.2f} USDT")
    print(f"总持仓数量: {total_stats['total_active_positions']} 个")
    print(f"总持仓价值: {total_stats['total_position_value']:,.2f} USDT")
    print(f"平均效率比率: {total_stats.get('avg_efficiency', 0):.4f}")

    # 各账户详细统计
    print(f"\n📈 各账户表现对比:")
    print("-" * 160)
    print(f"{'账户名称':<15} {'交易额':<12} {'手续费':<8} {'盈亏':<9} {'净收益':<8} {'交易次数':<9} {'持仓数':<8} {'持仓价值':<12} {'委托数':<8} {'胜率':<9} {'效率':<10}")
    print("-" * 160)

    for account_name, data in accounts_data.items():
        volume = data['volume_analysis']
        commission = data['commission_analysis']
        pnl = data['pnl_analysis']
        position = data['position_analysis']
        orders = data['order_analysis']
        efficiency_data = data['efficiency_analysis']

        net_profit = pnl['realized_pnl'] - commission['total_commission']
        win_rate = pnl['win_rate']
        efficiency = efficiency_data['efficiency_ratio']

        print(f"{account_name:<14} {volume['total_turnover']:>11,.0f} {commission['total_commission']:>12,.0f} "
              f"{pnl['realized_pnl']:>12,.1f} {net_profit:>12,.1f} {volume['total_trades']:>12} "
              f"{position['active_positions_count']:>10} {position['total_position_value']:>15,.0f} "
              f"{orders['total_orders']:>10} "
              f"{win_rate:>12.1%} {efficiency:>12.4f}")

    print("-" * 160)

def display_single_account_details_spot(account_name: str, data: Dict, minutes_interval: int = None, daily_volume_only: bool = False):
    """显示单个账户的详细信息 - 现货版本"""
    volume = data['volume_analysis']
    commission = data['commission_analysis']
    pnl = data['pnl_analysis']
    position = data['position_analysis']
    orders = data['order_analysis']
    efficiency_data = data['efficiency_analysis']

    if minutes_interval:
        print(f"\n  📋 账户: {account_name} (最近{minutes_interval}分钟)")
    elif daily_volume_only:
        print(f"\n  📋 账户: {account_name} (当天交易量)")
    else:
        print(f"\n  📋 账户: {account_name}")

    print(f"    交易额: {volume['total_turnover']:,.2f} USDT")
    print(f"    买入量: {volume['buy_volume']:,.4f}")
    print(f"    卖出量: {volume['sell_volume']:,.4f}")
    print(f"    净买入: {volume['net_volume']:+,.4f}")
    print(f"    交易次数: {volume['total_trades']:,} 次")
    print(f"    交易币对: {volume['symbols_traded_count']} 个")

    # 显示持仓信息
    print(f"    当前持仓: {position['active_positions_count']} 个币种")
    print(f"    持仓价值: {position['total_position_value']:,.2f} USDT")
    
    # 显示主要持仓
    if position['positions']:
        print(f"    主要持仓:")
        for pos in position['positions'][:5]:  # 显示前5个持仓
            print(f"      {pos['asset']}: {pos['total_amount']:,.4f} (价值: {pos['usdt_value']:,.2f} USDT)")
        if len(position['positions']) > 5:
            print(f"      ... 还有 {len(position['positions']) - 5} 个持仓")

    # 显示委托订单信息
    print(f"    当前委托: {orders['total_orders']} 个活跃订单")
    print(f"    委托价值: {orders['total_order_value']:,.2f} USDT")
    
    # 显示委托订单类型分布
    if orders['orders_by_type']:
        type_distribution = ", ".join([f"{k}: {v}" for k, v in orders['orders_by_type'].items()])
        print(f"    订单类型: {type_distribution}")
    
    if orders['orders_by_side']:
        side_distribution = ", ".join([f"{k}: {v}" for k, v in orders['orders_by_side'].items()])
        print(f"    买卖方向: {side_distribution}")

    # 显示效率分析
    print(f"    效率分析:")
    print(f"      手续费: {commission['total_commission']:,.4f}")
    print(f"      盈亏: {pnl['realized_pnl']:+,.4f}")
    print(f"      成本+盈亏: {efficiency_data['cost_pnl_total']:+,.4f}")
    print(f"      效率比率: {efficiency_data['efficiency_ratio']:.4f}")

    # 显示主要交易币对
    trades_by_symbol = volume['trades_by_symbol']
    if trades_by_symbol:
        sorted_symbols = sorted(trades_by_symbol.keys(),
                              key=lambda x: trades_by_symbol[x]['total_turnover'], reverse=True)[:3]
        print(f"    主要交易币对:")
        for symbol in sorted_symbols:
            symbol_data = trades_by_symbol[symbol]
            symbol_pnl = pnl['pnl_by_symbol'].get(symbol, 0)
            symbol_commission = commission['commission_by_symbol'].get(symbol, 0)
            print(f"      {symbol}: {symbol_data['total_turnover']:,.0f} USDT, "
                  f"盈亏: {symbol_pnl:+,.1f}, 手续费: {symbol_commission:,.1f}")

    # 显示当前委托订单详情
    open_orders = orders['orders']
    if open_orders:
        print(f"    当前委托订单详情:")
        for order in open_orders[:5]:  # 显示前5个委托订单
            symbol = order['symbol']
            side = order['side']
            order_type = order['type']
            quantity = float(order.get('origQty', 0))
            price = float(order.get('price', 0))
            status = order.get('status', 'UNKNOWN')
            
            side_emoji = "🟢" if side == 'BUY' else "🔴"
            print(f"      {symbol}: {side_emoji} {side} {order_type} {quantity:.4f} @ {price:.4f} ({status})")
        
        if len(open_orders) > 5:
            print(f"      ... 还有 {len(open_orders) - 5} 个委托订单")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AsterDex多账户现货交易分析工具')
    parser.add_argument('--config', '-c', default='config.yaml', help='配置文件路径')
    parser.add_argument('--days', '-d', type=int, help='分析天数')
    parser.add_argument('--minutes', '-m', type=int, help='分析分钟数（优先于天数）')
    parser.add_argument('--export', '-e', action='store_true', help='导出报告到文件')
    parser.add_argument('--account', '-a', help='指定单个账户分析（默认分析所有账户）')
    parser.add_argument('--daily', action='store_true', help='仅统计当天交易量（覆盖配置文件设置）')

    args = parser.parse_args()

    try:
        # 创建分析实例
        analyzer = AsterDexMultiAccountSpotAnalytics(config_file=args.config)

        # 如果命令行指定了--daily，覆盖配置文件设置
        if args.daily:
            analyzer.settings['daily_volume_only'] = True
            print("🔔 使用命令行参数：仅统计当天交易量")

        # 如果命令行指定了--minutes，覆盖配置文件设置
        if args.minutes:
            analyzer.settings['minutes_interval'] = args.minutes
            print(f"🔔 使用命令行参数：统计最近{args.minutes}分钟交易数据")

        # 生成多账户报告
        report = analyzer.generate_multi_account_report(days=args.days, minutes=args.minutes)

        # 显示报告
        display_multi_account_report(report)

        # 可选：保存报告到文件
        if args.export and report:
            # 生成文件名
            if args.minutes:
                time_suffix = f"_{args.minutes}min"
            elif analyzer.settings.get('daily_volume_only', False):
                time_suffix = "_daily"
            else:
                time_suffix = f"_{args.days or 7}days"
                
            filename = f"multi_account_spot_trading_report{time_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n报告已保存到: {filename}")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()