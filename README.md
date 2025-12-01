-- vol.py 统计
# 基本使用 - 使用默认配置（7天数据）
python3 vol.py

# 指定配置文件
python3 vol.py --config my_config.yaml
python3 vol.py -c my_config.yaml

# 时间范围控制
python3 vol.py --days 3              # 统计最近3天
python3 vol.py -d 3                  # 统计最近3天（简写）
python3 vol.py --minutes 30          # 统计最近30分钟
python3 vol.py -m 30                 # 统计最近30分钟（简写）
python3 vol.py --daily               # 仅统计当天交易量
python3 vol.py --minutes 60 --daily  # 分钟数优先（统计60分钟）

# 账户筛选
python3 vol.py --account "主账户"     # 只分析指定账户
python3 vol.py -a "主账户"            # 只分析指定账户（简写）

# 导出功能
python3 vol.py --export              # 生成报告并保存为JSON文件
python3 vol.py -e                    # 生成报告并保存为JSON文件（简写）

# 组合使用示例
python3 vol.py --config prod.yaml --days 1 --export
python3 vol.py -c prod.yaml -m 60 -e
python3 vol.py --minutes 30 --account "交易账户" --export
python3 vol.py --daily --export
-- closer.py 清仓
# 取消所有挂单并卖出所有非USDT资产
python3 closer.py --config config.yaml

# 仅取消挂单，不卖出资产
python3 closer.py --config config.yaml --no-sell

# 仅卖出指定资产，不取消挂单
python3 closer.py --config config.yaml --no-cancel --assets BTC ETH

# 列出所有有余额的资产和挂单
python3 closer.py --config config.yaml --list-symbols