import time
import hmac
import hashlib
import requests
import json
import logging
from typing import Dict, List, Optional
import argparse
import yaml
from decimal import Decimal, ROUND_DOWN

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('spot_position_closer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpotPositionCloser:
    def __init__(self, config_path: str):
        """初始化现货仓位清理器"""
        logger.info("🎯 初始化现货仓位清理器...")
        self.config_path = config_path
        self.load_config(config_path)
        self.base_url = "https://sapi.asterdex.com"
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.config['api_key']
        })
        
        # 加载启用的交易对列表
        self.enabled_symbols = self.get_enabled_symbols()
        logger.info(f"📋 已启用 {len(self.enabled_symbols)} 个交易对: {', '.join(self.enabled_symbols)}")
        
        logger.info("✅ 现货仓位清理器初始化完成")

    def load_config(self, config_path: str):
        """加载配置文件 - 适配新的配置文件格式"""
        logger.info(f"📁 加载配置文件: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # 适配新的配置文件格式
            api_key = config_data['api']['api_key']
            secret_key = config_data['api']['secret_key']

            self.config = {
                'api_key': api_key,
                'secret_key': secret_key,
                'symbols_config': config_data.get('symbols', [])
            }

            logger.info("✅ API密钥加载成功")
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise

    def get_enabled_symbols(self) -> List[str]:
        """获取启用的交易对列表"""
        enabled_symbols = []
        for symbol_config in self.config.get('symbols_config', []):
            symbol = symbol_config.get('symbol', '')
            enable = symbol_config.get('enable', True)  # 默认启用
            
            if enable:
                enabled_symbols.append(symbol)
                logger.debug(f"✅ 启用交易对: {symbol}")
            else:
                logger.debug(f"❌ 禁用交易对: {symbol}")
        
        return enabled_symbols

    def generate_signature(self, params: Dict) -> str:
        """生成HMAC SHA256签名"""
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        return hmac.new(
            self.config['secret_key'].encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def api_request(self, method: str, endpoint: str, signed: bool = False, **params) -> Dict:
        """发送API请求"""
        url = f"{self.base_url}{endpoint}"

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params['signature'] = self.generate_signature(params)

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = self.session.post(url, data=params, timeout=10)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, data=params, timeout=10)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API请求失败: {e}")
            raise

    def get_account_balances(self) -> List[Dict]:
        """获取账户余额信息，只返回有余额的资产"""
        try:
            logger.info("📊 获取账户余额信息...")
            response = self.api_request('GET', '/api/v1/account', signed=True)

            balances = []
            if response and 'balances' in response:
                logger.info(f"📋 收到 {len(response['balances'])} 个资产数据")
                
                for balance_data in response['balances']:
                    asset = balance_data.get('asset', '')
                    free = float(balance_data.get('free', 0))
                    locked = float(balance_data.get('locked', 0))
                    total = free + locked
                    
                    # 调试输出每个资产的信息
                    logger.debug(f"🔍 检查资产: {asset} = 可用:{free}, 冻结:{locked}, 总计:{total}")
                    
                    # 只要有余额就认为是有效资产
                    if total > 0.000001:
                        balances.append({
                            'asset': asset,
                            'free': free,
                            'locked': locked,
                            'total': total
                        })
                        logger.info(f"✅ 发现有余额资产: {asset} - 可用:{free}, 冻结:{locked}")

            logger.info(f"📊 总共发现 {len(balances)} 个有余额的资产")
            return balances

        except Exception as e:
            logger.error(f"❌ 获取账户余额失败: {e}")
            import traceback
            logger.error(f"❌ 详细错误: {traceback.format_exc()}")
            return []

    def get_symbol_for_asset(self, asset: str) -> Optional[str]:
        """为资产找到对应的交易对（使用USDT作为报价资产）"""
        # 优先在启用的交易对中查找
        for symbol in self.enabled_symbols:
            if symbol.endswith('USDT') and symbol.startswith(asset):
                return symbol
        
        # 如果没有找到，尝试构造标准的USDT交易对
        potential_symbol = f"{asset}USDT"
        
        # 检查这个交易对是否在交易所存在
        try:
            exchange_info = self.api_request('GET', '/api/v1/exchangeInfo')
            for symbol_info in exchange_info.get('symbols', []):
                if symbol_info['symbol'] == potential_symbol and symbol_info['status'] == 'TRADING':
                    logger.info(f"✅ 找到可交易对: {potential_symbol}")
                    return potential_symbol
        except Exception as e:
            logger.warning(f"⚠️ 检查交易对 {potential_symbol} 时出错: {e}")
        
        return None

    def get_all_open_orders(self) -> List[Dict]:
        """获取所有交易对的挂单信息"""
        try:
            response = self.api_request('GET', '/api/v1/openOrders', signed=True)

            if response and isinstance(response, list):
                logger.info(f"📋 收到 {len(response)} 个挂单数据")
                return response
            else:
                logger.info("📭 没有挂单数据")
                return []

        except Exception as e:
            logger.error(f"❌ 获取挂单信息失败: {e}")
            return []

    def format_quantity(self, symbol: str, quantity: float) -> str:
        """格式化数量到合适的精度"""
        try:
            # 先获取该交易对的精度信息
            exchange_info = self.api_request('GET', '/api/v1/exchangeInfo')
            symbol_info = None
            
            for info in exchange_info.get('symbols', []):
                if info['symbol'] == symbol:
                    symbol_info = info
                    break
            
            if symbol_info:
                # 找到LOT_SIZE过滤器
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_size_filter:
                    step_size = Decimal(lot_size_filter['stepSize'])
                    # 计算精度位数
                    step_str = format(step_size, 'f').rstrip('0').rstrip('.')
                    if '.' in step_str:
                        precision = len(step_str.split('.')[1])
                    else:
                        precision = 0
                    
                    # 格式化数量
                    formatted = f"{quantity:.{precision}f}"
                    result = formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted
                    logger.debug(f"🔧 {symbol} 数量格式化: {quantity} -> {result}")
                    return result

            # 默认处理
            result = f"{quantity:.8f}".rstrip('0').rstrip('.')
            logger.debug(f"🔧 {symbol} 使用默认格式化: {quantity} -> {result}")
            return result

        except Exception as e:
            logger.error(f"❌ {symbol} 数量格式化失败 {quantity}: {e}")
            # 备用方案
            result = f"{quantity:.6f}".rstrip('0').rstrip('.')
            return result

    def cancel_orders(self, symbols: List[str]):
        """取消指定交易对的挂单"""
        try:
            # 如果没有指定交易对，使用启用的交易对
            if not symbols:
                symbols = self.enabled_symbols
                logger.info("🔄 开始取消所有启用交易对的挂单...")
            else:
                # 过滤掉未启用的交易对
                symbols = [s for s in symbols if s in self.enabled_symbols]
                if not symbols:
                    logger.info("📭 指定的交易对都未启用")
                    return True
                logger.info(f"🔄 开始取消 {len(symbols)} 个启用交易对的挂单: {', '.join(symbols)}")

            # 获取所有挂单
            all_orders = self.get_all_open_orders()
            
            # 过滤指定交易对的挂单
            target_orders = [order for order in all_orders if order['symbol'] in symbols]

            if not target_orders:
                logger.info("📭 没有需要取消的订单")
                return True

            logger.info(f"🔄 发现 {len(target_orders)} 个挂单需要取消")

            # 按交易对分组
            symbol_orders = {}
            for order in target_orders:
                symbol = order['symbol']
                if symbol not in symbol_orders:
                    symbol_orders[symbol] = []
                symbol_orders[symbol].append(order)

            # 逐个取消订单
            success_count = 0
            total_count = len(target_orders)
            
            for symbol, orders in symbol_orders.items():
                logger.info(f"📊 处理 {symbol} 的 {len(orders)} 个挂单")
                for order in orders:
                    try:
                        result = self.api_request('DELETE', '/api/v1/order', signed=True,
                                       symbol=order['symbol'],
                                       orderId=order['orderId'])
                        logger.info(f"✅ 取消订单: {order['symbol']} - {order['orderId']} - {order['side']} {order['origQty']} @ {order['price']}")
                        success_count += 1
                        time.sleep(0.1)  # 避免频率限制
                    except Exception as e:
                        logger.error(f"❌ 取消订单失败 {order['symbol']} - {order['orderId']}: {e}")

            logger.info(f"✅ 成功取消 {success_count}/{total_count} 个订单")
            return success_count == total_count

        except Exception as e:
            logger.error(f"❌ 取消订单失败: {e}")
            return False

    def sell_assets(self, assets: List[str] = None):
        """市价卖出指定资产（转换为USDT）"""
        try:
            # 获取所有有余额的资产
            all_balances = self.get_account_balances()
            
            # 过滤要卖出的资产
            if assets:
                target_balances = [balance for balance in all_balances if balance['asset'] in assets]
                logger.info(f"🔄 开始卖出指定 {len(target_balances)} 个资产: {', '.join(assets)}")
            else:
                # 默认卖出所有非USDT资产
                target_balances = [balance for balance in all_balances if balance['asset'] != 'USDT']
                logger.info("🔄 开始卖出所有非USDT资产...")

            if not target_balances:
                logger.info("📭 当前没有需要卖出的资产")
                return True

            logger.info(f"🔄 发现 {len(target_balances)} 个有余额的资产需要处理")

            success_count = 0
            total_count = len(target_balances)

            for balance in target_balances:
                asset = balance['asset']
                free_amount = balance['free']
                
                # 跳过USDT本身
                if asset == 'USDT':
                    logger.info(f"⏭️ 跳过USDT资产")
                    continue
                
                # 找到对应的交易对
                symbol = self.get_symbol_for_asset(asset)
                if not symbol:
                    logger.warning(f"⚠️ 未找到 {asset} 对应的USDT交易对，跳过")
                    continue

                try:
                    # 格式化数量
                    quantity = self.format_quantity(symbol, free_amount)
                    logger.info(f"🔢 {asset} 格式化后数量: {quantity}")

                    # 下市价卖单
                    order_result = self.api_request('POST', '/api/v1/order', signed=True,
                        symbol=symbol,
                        side='SELL',
                        type='MARKET',
                        quantity=quantity
                    )

                    if order_result:
                        logger.info(f"✅ {asset} 卖出订单已提交: {order_result.get('orderId', 'N/A')}")
                        success_count += 1
                    else:
                        logger.error(f"❌ {asset} 卖出订单提交失败")

                    time.sleep(0.3)  # 避免频率限制

                except Exception as e:
                    logger.error(f"❌ {asset} 卖出失败: {e}")
                    import traceback
                    logger.error(f"❌ 详细错误: {traceback.format_exc()}")

            # 等待并确认资产已卖出
            logger.info("⏳ 等待卖出操作确认...")
            time.sleep(5)
            
            # 检查最终余额状态
            final_balances = self.get_account_balances()
            if assets:
                remaining_assets = [b for b in final_balances if b['asset'] in assets and b['total'] > 0.000001 and b['asset'] != 'USDT']
            else:
                remaining_assets = [b for b in final_balances if b['total'] > 0.000001 and b['asset'] != 'USDT']
            
            remaining_count = len(remaining_assets)
            if remaining_count == 0:
                logger.info("✅ 所有资产已成功卖出")
                return True
            else:
                logger.warning(f"⚠️ 仍有 {remaining_count} 个资产未完全卖出")
                for balance in remaining_assets:
                    logger.warning(f"⚠️ 剩余资产: {balance['asset']} - {balance['total']}")
                return False

        except Exception as e:
            logger.error(f"❌ 卖出资产失败: {e}")
            import traceback
            logger.error(f"❌ 详细错误: {traceback.format_exc()}")
            return False

    def run(self, cancel_orders: bool = True, sell_assets: bool = True, symbols: List[str] = None, assets: List[str] = None):
        """运行清理程序"""
        logger.info("🚀 开始执行现货清理操作...")
        
        # 如果没有指定交易对，使用启用的交易对
        if not symbols:
            symbols = self.enabled_symbols
            logger.info(f"🎯 处理所有启用交易对: {', '.join(symbols)}")
        else:
            # 过滤掉未启用的交易对
            symbols = [s for s in symbols if s in self.enabled_symbols]
            if not symbols:
                logger.info("📭 指定的交易对都未启用")
                return True
            logger.info(f"🎯 处理指定启用交易对: {', '.join(symbols)}")
        
        success = True
        
        # 取消挂单
        if cancel_orders:
            if not self.cancel_orders(symbols):
                success = False
                logger.error("❌ 取消挂单失败")
            else:
                logger.info("✅ 取消挂单完成")
            
            # 等待一下让取消操作完成
            time.sleep(2)
        
        # 卖出资产
        if sell_assets:
            if not self.sell_assets(assets):
                success = False
                logger.error("❌ 卖出资产失败")
            else:
                logger.info("✅ 卖出资产完成")
        
        if success:
            logger.info("🎉 所有清理操作完成!")
        else:
            logger.warning("⚠️ 部分操作失败，请检查日志")
        
        return success

def main():
    parser = argparse.ArgumentParser(description='Asterdex现货清理工具')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (例如: enhanced_market_maker.yaml)')
    parser.add_argument('--no-cancel', action='store_true', 
                       help='不取消挂单，仅卖出资产')
    parser.add_argument('--no-sell', action='store_true', 
                       help='不卖出资产，仅取消挂单')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='指定交易对 (例如: BTCUSDT ETHUSDT), 不指定则处理所有启用交易对')
    parser.add_argument('--assets', type=str, nargs='+',
                       help='指定要卖出的资产 (例如: BTC ETH), 不指定则卖出所有非USDT资产')
    parser.add_argument('--list-symbols', action='store_true',
                       help='列出所有有余额或挂单的交易对')
    parser.add_argument('--list-enabled', action='store_true',
                       help='列出所有启用的交易对')
    parser.add_argument('--debug', action='store_true',
                       help='开启调试模式，显示更多详细信息')
    
    args = parser.parse_args()

    # 设置调试级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 如果只是列出启用的交易对
    if args.list_enabled:
        try:
            closer = SpotPositionCloser(args.config)
            print("📋 启用的交易对:")
            for symbol in closer.enabled_symbols:
                print(f"   - {symbol}")
            return
        except Exception as e:
            print(f"❌ 列出启用交易对失败: {e}")
            return

    # 如果只是列出交易对状态
    if args.list_symbols:
        try:
            closer = SpotPositionCloser(args.config)
            balances = closer.get_account_balances()
            orders = closer.get_all_open_orders()
            
            print("📊 交易对状态汇总:")
            print(f"   有余额的资产: {len(balances)} 个")
            for balance in balances:
                print(f"     {balance['asset']}: 可用:{balance['free']}, 冻结:{balance['locked']}, 总计:{balance['total']}")
            
            print(f"   有挂单的交易对: {len(orders)} 个")
            order_symbols = set(order['symbol'] for order in orders)
            for symbol in order_symbols:
                symbol_orders = [o for o in orders if o['symbol'] == symbol]
                print(f"     {symbol}: {len(symbol_orders)} 个挂单")
            
            return
        except Exception as e:
            print(f"❌ 列出状态失败: {e}")
            import traceback
            print(f"❌ 详细错误: {traceback.format_exc()}")
            return

    # 确定执行的操作
    cancel_orders = not args.no_cancel
    sell_assets = not args.no_sell
    
    if not cancel_orders and not sell_assets:
        print("❌ 必须至少执行一个操作 (取消挂单或卖出资产)")
        return

    position_closer = None
    try:
        position_closer = SpotPositionCloser(args.config)

        print("🎯 Asterdex现货清理工具启动")
        print(f"   - 配置文件: {args.config}")
        print(f"   - 启用交易对: {', '.join(position_closer.enabled_symbols)}")
        
        if args.symbols:
            # 显示哪些交易对被过滤掉了
            filtered_symbols = [s for s in args.symbols if s not in position_closer.enabled_symbols]
            if filtered_symbols:
                print(f"   - 过滤掉的交易对: {', '.join(filtered_symbols)} (未启用)")
            print(f"   - 实际处理交易对: {', '.join([s for s in args.symbols if s in position_closer.enabled_symbols])}")
        else:
            print("   - 处理所有启用交易对")
        
        if args.assets:
            print(f"   - 卖出指定资产: {', '.join(args.assets)}")
        else:
            print("   - 卖出所有非USDT资产")
        
        if cancel_orders:
            print("   - 取消所有挂单")
        if sell_assets:
            print("   - 市价卖出资产")
        
        # 显示当前状态
        balances = position_closer.get_account_balances()
        orders = position_closer.get_all_open_orders()
        print(f"📊 当前状态: {len(balances)} 个有余额的资产, {len(orders)} 个挂单")
        
        print("   正在执行...")

        success = position_closer.run(cancel_orders, sell_assets, args.symbols, args.assets)
        
        if success:
            print("✅ 清理操作完成!")
        else:
            print("❌ 清理操作遇到问题，请检查日志")

    except KeyboardInterrupt:
        print("\n🛑 用户中断操作")
    except Exception as e:
        logger.error(f"❌ 程序运行错误: {e}")
        print(f"❌ 程序运行错误: {e}")
        import traceback
        print(f"❌ 详细错误: {traceback.format_exc()}")
    finally:
        if position_closer:
            print("🔚 程序结束")

if __name__ == "__main__":
    main()