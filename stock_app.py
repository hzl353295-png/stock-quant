# ========== 导入与基础配置 ==========
from flask import Flask, request, render_template_string, jsonify, send_file
import pandas as pd
import numpy as np
import requests
import time
import re
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

app = Flask(__name__)

# 权重文件
WEIGHTS_FILE = 'weights.json'
WATCHLIST_FILE = 'watchlist.json'
LOG_FILE = 'trade_log.json'

DEFAULT_WEIGHTS = {
    'macd': 30, 'rsi': 20, 'kdj': 15, 'ma': 20, 'volume': 10, 'market': 10,
    'flow': 10, 'pattern': 5  # 新增资金流向和形态权重
}

# 缓存系统：{code: (timestamp, df)}
DATA_CACHE = {}
CACHE_TTL = 300  # 5分钟

# 线程锁
cache_lock = threading.Lock()

# ------------------- 自选股存储 -------------------
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as f:
            return json.load(f)
    return []

def save_watchlist(data):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(data, f)

# ------------------- 股票名称获取 -------------------
def get_stock_name(code):
    try:
        if code.startswith('6'):
            secid = f'1.{code}'
        else:
            secid = f'0.{code}'
        url = 'https://push2.eastmoney.com/api/qt/stock/get'
        params = {'secid': secid, 'fields': 'f57', 'ut': 'fa5fd1943c7b386f172d6893d3e0f15e'}
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        return data['data']['f57']
    except:
        return ''

# ------------------- 资金流向 -------------------
def get_money_flow(code):
    """获取主力资金净流入/流出（万元）"""
    try:
        if code.startswith('6'):
            secid = f'1.{code}'
        else:
            secid = f'0.{code}'
        url = 'https://push2.eastmoney.com/api/qt/stock/get'
        params = {
            'secid': secid,
            'fields': 'f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f64,f65,f70,f71,f76,f77,f82,f83,f88,f89',
            'ut': 'fa5fd1943c7b386f172d6893d3e0f15e'
        }
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()['data']
        # 主力净流入
        main_in = float(data.get('f62', 0))  # 今日主力净流入
        return main_in
    except:
        return 0

# ------------------- 数据获取（缓存加速） -------------------
def get_stock_data(code):
    with cache_lock:
        if code in DATA_CACHE:
            ts, df = DATA_CACHE[code]
            if time.time() - ts < CACHE_TTL:
                return df.copy()
    # 否则获取新数据
    df = _fetch_data(code)
    with cache_lock:
        DATA_CACHE[code] = (time.time(), df.copy())
    return df

def _fetch_data(code):
    """实际获取数据，优先东方财富，失败用新浪"""
    try:
        return get_data_eastmoney(code)
    except:
        return get_data_sina(code)

def get_data_eastmoney(code, max_retries=3):
    if code.startswith('6'):
        secid = f'1.{code}'
    else:
        secid = f'0.{code}'
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': secid,
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101', 'fqt': '1', 'end': '20500101', 'lmt': '500',
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    for i in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            data = resp.json()
            if data.get('data') and data['data'].get('klines'):
                break
        except:
            if i == max_retries - 1:
                raise
            time.sleep(1)
    lines = data['data']['klines']
    records = []
    for line in lines:
        parts = line.split(',')
        records.append({
            'date': parts[0],
            'open': float(parts[1]), 'close': float(parts[2]),
            'high': float(parts[3]), 'low': float(parts[4]), 'volume': float(parts[5])
        })
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]

def get_data_sina(code):
    market = 'sh' if code.startswith('6') else 'sz'
    url = f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={market}{code}&scale=240&ma=no&datalen=500'
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    data = resp.json()
    records = []
    for item in data:
        records.append({
            'date': item['day'], 'open': float(item['open']),
            'high': float(item['high']), 'low': float(item['low']),
            'close': float(item['close']), 'volume': float(item['volume'])
        })
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]

# ------------------- 指标计算 -------------------
def compute_all_indicators(df):
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['signal']

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    df['ma20'] = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = df['ma20'] + 2 * std20
    df['bb_lower'] = df['ma20'] - 2 * std20

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    return df

# ------------------- K线形态识别 -------------------
def detect_pattern(df):
    """检测最后三天K线形态，返回形态名称和方向（1看涨，-1看跌，0无）"""
    if len(df) < 3:
        return None, 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    o1, c1, h1, l1 = last['open'], last['close'], last['high'], last['low']
    o2, c2, h2, l2 = prev['open'], prev['close'], prev['high'], prev['low']
    o3, c3, h3, l3 = prev2['open'], prev2['close'], prev2['high'], prev2['low']

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    body3 = abs(c3 - o3)

    # 锤子线 (下影线 > 2倍实体，上影线极短)
    if body1 > 0 and (l1 == min(l1, l2)) and ((o1 - l1) > 2 * body1) and ((h1 - c1) < body1 * 0.3) and c1 > o1:
        return "锤子线（看涨）", 1
    if body1 > 0 and ((h1 - o1) > 2 * body1) and ((c1 - l1) < body1 * 0.3) and c1 < o1:
        return "上吊线（看跌）", -1

    # 吞没形态
    if c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2:  # 阳包阴
        return "看涨吞没", 1
    if c2 > o2 and c1 < o1 and c1 < o2 and o1 > c2:  # 阴包阳
        return "看跌吞没", -1

    # 早晨之星 / 黄昏之星
    if (c3 < o3) and (abs(c2 - o2) < body3 * 0.3) and (c1 > o1) and (c1 > (o3 + c3) / 2):
        return "早晨之星（看涨）", 1
    if (c3 > o3) and (abs(c2 - o2) < body3 * 0.3) and (c1 < o1) and (c1 < (o3 + c3) / 2):
        return "黄昏之星（看跌）", -1

    # 十字星
    if body1 < (h1 - l1) * 0.1:
        return "十字星（反转可能）", 0

    return None, 0

# ------------------- 大盘环境 -------------------
def get_market_env():
    try:
        df = get_stock_data('000001')
        df = compute_all_indicators(df)
        last = df.iloc[-1]
        score = 0
        if last['close'] > last['ma20']: score += 1
        if last['macd_diff'] > 0: score += 1
        if last['rsi'] > 40: score += 1
        if score >= 2: return '强'
        if score == 1: return '中'
        return '弱'
    except:
        return '中'

# ------------------- 综合评分（加入资金和形态） -------------------
def deep_score(df, market_env='中', weights=None, flow=0, pattern_dir=0):
    if weights is None:
        weights = DEFAULT_WEIGHTS
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close, macd, diff, prev_diff = last['close'], last['macd'], last['macd_diff'], prev['macd_diff']
    rsi, k, d, j, vol_ratio = last['rsi'], last['k'], last['d'], last['j'], last['vol_ratio']
    ma5, ma10, ma20 = last['ma5'], last['ma10'], last['ma20']
    bb_upper, bb_lower = last['bb_upper'], last['bb_lower']

    # 原有评分
    score_macd = 0
    if prev_diff <= 0 and diff > 0: score_macd = 30
    elif diff > 0 and diff > prev_diff: score_macd = 18
    elif diff > 0: score_macd = 8
    elif diff < 0 and diff > prev_diff: score_macd = -5
    else: score_macd = -25

    score_rsi = 0
    if 30 <= rsi <= 40: score_rsi = 15
    elif 40 <= rsi <= 60: score_rsi = 10
    elif 60 < rsi <= 70: score_rsi = 5
    elif rsi > 70: score_rsi = -15
    elif rsi < 30: score_rsi = 20

    score_kdj = 0
    if k > 80 and d > 80: score_kdj = -10
    elif k < 20 and d < 20: score_kdj = 12
    if j < 0 or j > 100: score_kdj += 5

    score_ma = 0
    if close > ma5 > ma10 > ma20: score_ma = 20
    elif close > ma5: score_ma = 10
    elif close < ma5: score_ma = -10
    if close > bb_upper: score_ma += 8
    elif close < bb_lower: score_ma += 12

    score_vol = 0
    if vol_ratio > 1.5 and diff > 0: score_vol = 18
    elif vol_ratio > 1.5 and diff < 0: score_vol = -15
    elif vol_ratio < 0.5: score_vol = -5

    score_market = 0
    if market_env == '弱': score_market = -15
    elif market_env == '强': score_market = 10

    # 资金流向评分 (单位万元)
    if flow > 5000: score_flow = 15
    elif flow > 1000: score_flow = 10
    elif flow > 0: score_flow = 5
    elif flow < -5000: score_flow = -15
    elif flow < -1000: score_flow = -10
    else: score_flow = 0

    # 形态评分
    score_pattern = 0
    if pattern_dir == 1: score_pattern = 10
    elif pattern_dir == -1: score_pattern = -10

    total = (score_macd * weights['macd'] / 30 +
             score_rsi * weights['rsi'] / 20 +
             score_kdj * weights['kdj'] / 15 +
             score_ma * weights['ma'] / 20 +
             score_vol * weights['volume'] / 10 +
             score_market * weights['market'] / 10 +
             score_flow * weights.get('flow', 10) / 10 +
             score_pattern * weights.get('pattern', 5) / 5)
    total = round(total)
    prob = max(0, min(100, round((total + 90) / 180 * 100)))
    return total, prob

# ------------------- 多策略对比 -------------------
def multi_strategy(df):
    """返回4种不同策略的评分和建议"""
    market_env = get_market_env()
    last = df.iloc[-1]

    # 策略1：标准MACD+RSI
    from_strategy1 = deep_score(df, market_env, DEFAULT_WEIGHTS)
    # 策略2：纯MACD（提高macd权重）
    w2 = DEFAULT_WEIGHTS.copy()
    w2['macd'] = 50; w2['rsi'] = 10; w2['kdj'] = 5; w2['ma'] = 10; w2['volume'] = 5
    from_strategy2 = deep_score(df, market_env, w2)
    # 策略3：纯KDJ+布林带
    w3 = DEFAULT_WEIGHTS.copy()
    w3['macd'] = 5; w3['rsi'] = 5; w3['kdj'] = 40; w3['ma'] = 30; w3['volume'] = 10
    from_strategy3 = deep_score(df, market_env, w3)
    # 策略4：量价结合（量能权重最高）
    w4 = DEFAULT_WEIGHTS.copy()
    w4['macd'] = 20; w4['rsi'] = 10; w4['kdj'] = 5; w4['ma'] = 15; w4['volume'] = 40
    from_strategy4 = deep_score(df, market_env, w4)

    def advice(score):
        if score >= 35: return "强烈买入"
        if score >= 15: return "可考虑买入"
        if score >= -10: return "观望"
        return "卖出/规避"
    return [
        {'name': 'MACD+RSI', 'score': from_strategy1[0], 'prob': from_strategy1[1], 'advice': advice(from_strategy1[0])},
        {'name': '纯MACD', 'score': from_strategy2[0], 'prob': from_strategy2[1], 'advice': advice(from_strategy2[0])},
        {'name': 'KDJ+布林', 'score': from_strategy3[0], 'prob': from_strategy3[1], 'advice': advice(from_strategy3[0])},
        {'name': '量价优先', 'score': from_strategy4[0], 'prob': from_strategy4[1], 'advice': advice(from_strategy4[0])},
    ]

# ------------------- 回测 -------------------
def backtest_strategy(df, start_date=None, end_date=None, init_capital=100000, commission=0.001):
    df = df.copy()
    df['buy_signal'] = 0
    df['sell_signal'] = 0
    macd_diff = df['macd_diff'].values
    rsi = df['rsi'].values
    for i in range(1, len(df)):
        if macd_diff[i-1] <= 0 and macd_diff[i] > 0 and rsi[i] < 70:
            df.iloc[i, df.columns.get_loc('buy_signal')] = 1
        if (macd_diff[i-1] >= 0 and macd_diff[i] < 0) or (rsi[i] > 80):
            df.iloc[i, df.columns.get_loc('sell_signal')] = 1

    if start_date: df = df[df.index >= start_date]
    if end_date: df = df[df.index <= end_date]

    capital = init_capital
    position = 0
    buy_price = 0
    trades = []
    for i in range(len(df)):
        date = df.index[i]
        close = df.iloc[i]['close']
        if df.iloc[i]['buy_signal'] and position == 0:
            position = int(capital * (1 - commission) / close)
            capital -= position * close * (1 + commission)
            buy_price = close
            trades.append({'date': date, 'type': 'buy', 'price': close})
        elif df.iloc[i]['sell_signal'] and position > 0:
            revenue = position * close * (1 - commission)
            capital += revenue
            profit_pct = (revenue - position * buy_price * (1 + commission)) / (position * buy_price * (1 + commission))
            trades.append({'date': date, 'type': 'sell', 'price': close, 'profit_pct': profit_pct})
            position = 0
    if position > 0:
        capital += position * df.iloc[-1]['close'] * (1 - commission)
    total_return = (capital - init_capital) / init_capital * 100
    win = [t for t in trades if t['type'] == 'sell' and t.get('profit_pct', 0) > 0]
    win_rate = len(win) / max(1, len([t for t in trades if t['type'] == 'sell'])) * 100
    return {'total_return': round(total_return, 2), 'final_capital': round(capital, 2),
            'trade_count': len([t for t in trades if t['type'] == 'sell']), 'win_rate': round(win_rate, 2)}

# ------------------- 板块热度 -------------------
def get_sector_heat():
    try:
        url = 'https://push2.eastmoney.com/api/qt/clist/get'
        params = {'pn': '1', 'pz': '500', 'po': '1', 'np': '1', 'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                  'fltt': '2', 'invt': '2', 'fid': 'f3', 'fs': 'm:90+t:2', 'fields': 'f2,f3,f4,f12,f14'}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return [{'code': i['f12'], 'name': i['f14'], 'pct': i['f3']} for i in data['data']['diff'][:5]]
    except:
        return []

def get_top_stocks_in_sector(sector_code):
    try:
        url = 'https://push2.eastmoney.com/api/qt/clist/get'
        params = {'pn': '1', 'pz': '10', 'po': '1', 'np': '1', 'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                  'fltt': '2', 'invt': '2', 'fid': 'f3', 'fs': f'b:{sector_code}', 'fields': 'f2,f3,f12,f14'}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return [{'code': i['f12'], 'name': i['f14'], 'pct': i['f3']} for i in data['data']['diff'][:5]]
    except:
        return []

# ================== 前端 HTML（响应式） ==================
HTML = '''
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化交易终端 Pro Max</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:"Microsoft YaHei",sans-serif; background:#1e1e2f; color:#eee; display:flex; flex-direction:column; }
        .sidebar { width:100%; background:#2a2a3d; display:flex; flex-wrap:wrap; gap:5px; padding:10px; position:sticky; top:0; z-index:100; }
        .sidebar button { background:#3a3a5a; color:white; border:none; padding:10px 8px; border-radius:6px; cursor:pointer; font-size:12px; flex:1 0 auto; }
        .sidebar button:hover { background:#5050a0; }
        .main { flex:1; padding:15px; overflow-y:auto; }
        .tab { display:none; }
        .tab.active { display:block; }
        .card { background:#2a2a3d; border-radius:12px; padding:15px; margin-bottom:15px; }
        h2 { font-size:18px; margin-bottom:10px; }
        input,textarea,select { background:#3a3a5a; border:none; color:white; padding:10px; border-radius:6px; width:100%; margin:5px 0; font-size:14px; }
        button { background:#5050a0; color:white; border:none; padding:10px 15px; border-radius:6px; cursor:pointer; margin:2px; font-size:14px; }
        button:hover { background:#6060c0; }
        table { width:100%; border-collapse:collapse; margin-top:10px; font-size:13px; }
        th { background:#5050a0; padding:8px; }
        td { padding:6px; border-bottom:1px solid #444; }
        .green { color:#00b894; } .red { color:#d63031; } .yellow { color:#fdcb6e; }
        .signal-box { padding:15px; border-radius:10px; margin:10px 0; }
        .strong-buy { background:#00b89433; border:1px solid #00b894; }
        .buy { background:#55efc433; border:1px solid #55efc4; }
        .wait { background:#0984e333; border:1px solid #0984e3; }
        .sell { background:#d6303133; border:1px solid #d63031; }
        @media (min-width: 768px) {
            body { flex-direction:row; }
            .sidebar { width:200px; flex-direction:column; position:static; }
            .sidebar button { font-size:14px; }
        }
    </style>
</head>
<body>
<div class="sidebar">
    <button onclick="showTab('single')">📈 单股分析</button>
    <button onclick="showTab('batch')">📊 智能选股</button>
    <button onclick="showTab('watchlist')">⭐ 自选监控</button>
    <button onclick="showTab('alert')">🔔 提醒</button>
    <button onclick="showTab('backtest')">📉 回测</button>
    <button onclick="showTab('sector')">🏭 板块</button>
    <button onclick="showTab('weights')">⚙️ 权重</button>
    <button onclick="showTab('log')">📓 操作日志</button>
</div>
<div class="main">
    <!-- 单股分析 -->
    <div id="single" class="tab active">
        <div class="card">
            <h2>单股深度分析</h2>
            <input type="text" id="s_code" placeholder="股票代码"><br>
            <input type="number" id="s_shares" placeholder="持仓股数（0空仓）"><br>
            <input type="number" id="s_cost" placeholder="成本价"><br>
            <button onclick="analyzeSingle()">分析</button>
            <div id="single_result"></div>
        </div>
    </div>
    <!-- 智能选股 -->
    <div id="batch" class="tab">
        <div class="card">
            <h2>智能筛选批量选股</h2>
            <textarea id="b_codes" placeholder="输入代码，逗号/空格分隔"></textarea>
            <div style="display:flex; flex-wrap:wrap; gap:10px;">
                <label>最低评分 <input type="number" id="min_score" value="15" style="width:80px;"></label>
                <label>最低概率% <input type="number" id="min_prob" value="60" style="width:80px;"></label>
                <label>RSI上限 <input type="number" id="max_rsi" value="70" style="width:80px;"></label>
                <label>MACD状态 <select id="macd_filter"><option value="">不限</option><option value="long">多头</option><option value="golden">金叉</option></select></label>
            </div>
            <button onclick="analyzeBatch()">智能筛选</button>
            <div id="batch_result"></div>
        </div>
    </div>
    <!-- 自选 -->
    <div id="watchlist" class="tab">
        <div class="card">
            <h2>自选股</h2>
            <input type="text" id="wl_code" placeholder="代码"><button onclick="addWatchlist()">添加</button>
            <button onclick="refreshWatchlist()">刷新</button>
            <div id="watchlist_result"></div>
        </div>
    </div>
    <!-- 提醒 -->
    <div id="alert" class="tab">
        <div class="card">
            <h2>提醒</h2>
            <button id="alert_toggle" onclick="toggleAlert()">开启</button>
            <div id="alert_status"></div>
        </div>
    </div>
    <!-- 回测 -->
    <div id="backtest" class="tab">
        <div class="card">
            <h2>回测</h2>
            <input type="text" id="bt_code" placeholder="代码"><br>
            <input type="date" id="bt_start" value="2023-01-01"><br>
            <input type="date" id="bt_end"><br>
            <button onclick="runBacktest()">回测</button>
            <div id="backtest_result"></div>
        </div>
    </div>
    <!-- 板块 -->
    <div id="sector" class="tab">
        <div class="card">
            <h2>板块热度</h2>
            <button onclick="loadSectorHeat()">刷新</button>
            <div id="sector_result"></div>
        </div>
    </div>
    <!-- 权重 -->
    <div id="weights" class="tab">
        <div class="card">
            <h2>权重</h2>
            <div id="weights_inputs"></div>
            <button onclick="saveWeights()">保存</button>
        </div>
    </div>
    <!-- 操作日志 -->
    <div id="log" class="tab">
        <div class="card">
            <h2>操作日志</h2>
            <input type="text" id="log_code" placeholder="股票代码"><br>
            <input type="text" id="log_action" placeholder="操作（买入/卖出）"><br>
            <input type="number" id="log_price" placeholder="价格"><br>
            <input type="number" id="log_shares" placeholder="数量"><br>
            <button onclick="addLog()">添加记录</button>
            <button onclick="loadLogs()">刷新日志</button>
            <div id="log_list"></div>
        </div>
    </div>
</div>

<script>
let alertInterval, alertActive = false;

function showTab(id) {
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    if (id === 'log') loadLogs();
}

// 单股
async function analyzeSingle() {
    let code = document.getElementById('s_code').value;
    let shares = document.getElementById('s_shares').value;
    let cost = document.getElementById('s_cost').value;
    let resp = await fetch('/single', {
        method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body:`stock_code=${code}&shares=${shares}&cost=${cost}`
    });
    document.getElementById('single_result').innerHTML = await resp.text();
}

// 批量
async function analyzeBatch() {
    let codes = document.getElementById('b_codes').value;
    let minScore = document.getElementById('min_score').value || 0;
    let minProb = document.getElementById('min_prob').value || 0;
    let maxRsi = document.getElementById('max_rsi').value || 100;
    let macdFilter = document.getElementById('macd_filter').value;
    let resp = await fetch('/batch_filter', {
        method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body:`codes=${codes}&min_score=${minScore}&min_prob=${minProb}&max_rsi=${maxRsi}&macd_filter=${macdFilter}`
    });
    document.getElementById('batch_result').innerHTML = await resp.text();
}

// 自选
async function addWatchlist() {
    let code = document.getElementById('wl_code').value;
    await fetch('/watchlist/add', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:`code=${code}`});
    refreshWatchlist();
}
async function removeWatchlist(code) {
    await fetch('/watchlist/remove', {method:'POST', body:`code=${code}`, headers:{'Content-Type':'application/x-www-form-urlencoded'}});
    refreshWatchlist();
}
async function refreshWatchlist() {
    let resp = await fetch('/watchlist/refresh');
    document.getElementById('watchlist_result').innerHTML = await resp.text();
}
refreshWatchlist();

// 提醒
async function checkSignals() {
    let resp = await fetch('/alert/check');
    let data = await resp.json();
    data.alerts.forEach(a => {
        if (Notification.permission === "granted") new Notification(a.code, {body: a.msg});
        document.getElementById('alert_status').innerHTML += `<div>${a.code}: ${a.msg}</div>`;
    });
}
function toggleAlert() {
    if (alertActive) {
        clearInterval(alertInterval); alertActive = false;
        document.getElementById('alert_toggle').innerText = "开启";
    } else {
        if (Notification.permission !== "granted") Notification.requestPermission();
        alertInterval = setInterval(checkSignals, 300000);
        alertActive = true;
        document.getElementById('alert_toggle').innerText = "关闭";
    }
}

async function runBacktest() {
    let code = document.getElementById('bt_code').value;
    let start = document.getElementById('bt_start').value;
    let end = document.getElementById('bt_end').value;
    let resp = await fetch('/backtest', {
        method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body:`code=${code}&start=${start}&end=${end}`
    });
    document.getElementById('backtest_result').innerHTML = await resp.text();
}

async function loadSectorHeat() {
    let resp = await fetch('/sector');
    document.getElementById('sector_result').innerHTML = await resp.text();
}

// 权重
async function loadWeights() {
    let resp = await fetch('/weights');
    let data = await resp.json();
    let html = '';
    for (let k in data) html += `<label>${k} <input type="number" id="w_${k}" value="${data[k]}" min="0" max="100" style="width:70px;"></label> `;
    document.getElementById('weights_inputs').innerHTML = html;
}
async function saveWeights() {
    let weights = {};
    ['macd','rsi','kdj','ma','volume','market','flow','pattern'].forEach(k => {
        weights[k] = parseInt(document.getElementById('w_'+k).value) || 0;
    });
    await fetch('/weights', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(weights)});
    alert('已保存');
}
loadWeights();

// 操作日志 (localStorage)
function loadLogs() {
    let logs = JSON.parse(localStorage.getItem('trade_logs') || '[]');
    let html = '<table><tr><th>时间</th><th>代码</th><th>操作</th><th>价格</th><th>数量</th></tr>';
    logs.forEach(l => {
        html += `<tr><td>${l.time}</td><td>${l.code}</td><td>${l.action}</td><td>${l.price}</td><td>${l.shares}</td></tr>`;
    });
    html += '</table>';
    document.getElementById('log_list').innerHTML = html;
}
function addLog() {
    let code = document.getElementById('log_code').value;
    let action = document.getElementById('log_action').value;
    let price = document.getElementById('log_price').value;
    let shares = document.getElementById('log_shares').value;
    let logs = JSON.parse(localStorage.getItem('trade_logs') || '[]');
    logs.push({time: new Date().toLocaleString(), code, action, price, shares});
    localStorage.setItem('trade_logs', JSON.stringify(logs));
    loadLogs();
}
</script>
</body>
</html>
'''

# ================== 后端路由 ==================
@app.route('/')
def index():
    return render_template_string(HTML)

# 单股分析（含形态、资金、多策略）
@app.route('/single', methods=['POST'])
def single():
    code = request.form.get('stock_code')
    shares = int(request.form.get('shares', 0) or 0)
    cost = float(request.form.get('cost', 0) or 0)
    try:
        df = get_stock_data(code)
        df = compute_all_indicators(df)
        market_env = get_market_env()
        # 资金流向
        flow = get_money_flow(code)
        # 形态
        pattern_name, pattern_dir = detect_pattern(df)
        # 评分
        weights = load_weights()
        score, prob = deep_score(df, market_env, weights, flow, pattern_dir)
        last = df.iloc[-1]
        if score >= 35: advice="强烈买入"; cls="strong-buy"
        elif score >= 15: advice="可考虑买入"; cls="buy"
        elif score >= -10: advice="观望"; cls="wait"
        else: advice="卖出/规避"; cls="sell"
        # 止损止盈
        atr = last['atr']
        stop_loss = round(last['close'] - 2*atr, 2)
        take_profit = round(last['close'] + 3*atr, 2)
        # 持仓
        pos_html = ''
        if shares > 0 and cost > 0:
            profit = shares * (last['close'] - cost)
            pct = profit / (shares * cost) * 100
            pos_html += f"浮动盈亏: {profit:+.2f} ({pct:+.2f}%)<br>"
        # 多策略
        multi = multi_strategy(df)
        multi_html = '<b>多策略对比：</b><table><tr><th>策略</th><th>评分</th><th>概率</th><th>建议</th></tr>'
        for m in multi:
            multi_html += f'<tr><td>{m["name"]}</td><td>{m["score"]}</td><td>{m["prob"]}%</td><td>{m["advice"]}</td></tr>'
        multi_html += '</table>'
        # K线图
        fig, ax = plt.subplots(figsize=(8,4))
        df['close'].plot(ax=ax, label='Close')
        ax.set_title(code)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        html = f'''
        <div class="signal-box {cls}"><b>{advice}</b> 概率{prob}%</div>
        {pos_html}
        <p>止损: {stop_loss} | 止盈: {take_profit}</p>
        <p>资金流向: {flow:.0f}万元 | 形态: {pattern_name or "无典型形态"}</p>
        {multi_html}
        <img src="data:image/png;base64,{img}" style="width:100%">
        <a href="data:image/png;base64,{img}" download="{code}.png">下载图表</a>
        '''
        return html
    except Exception as e:
        return f"错误: {e}"

# 批量选股（带筛选）
@app.route('/batch_filter', methods=['POST'])
def batch_filter():
    codes = re.split(r'[,，\s]+', request.form.get('codes', ''))
    codes = [c.strip() for c in codes if c.strip()][:50]
    min_score = int(request.form.get('min_score', 0))
    min_prob = int(request.form.get('min_prob', 0))
    max_rsi = float(request.form.get('max_rsi', 100))
    macd_filter = request.form.get('macd_filter', '')

    results = []
    def process(code):
        try:
            df = get_stock_data(code)
            df = compute_all_indicators(df)
            last = df.iloc[-1]
            market_env = get_market_env()
            flow = get_money_flow(code)
            pattern_name, pattern_dir = detect_pattern(df)
            weights = load_weights()
            score, prob = deep_score(df, market_env, weights, flow, pattern_dir)
            if score < min_score or prob < min_prob or last['rsi'] > max_rsi:
                return None
            if macd_filter == 'long' and last['macd_diff'] <= 0:
                return None
            if macd_filter == 'golden' and (df.iloc[-2]['macd_diff'] > 0 or last['macd_diff'] <= 0):
                return None
            if score >= 35: advice="强烈买入"
            elif score >= 15: advice="可考虑买入"
            elif score >= -10: advice="观望"
            else: advice="卖出/规避"
            return {'code': code, 'price': last['close'], 'prob': prob, 'score': score, 'advice': advice,
                    'rsi': last['rsi'], 'macd': "多头" if last['macd_diff']>0 else "空头", 'flow': flow}
        except:
            return None

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process, code) for code in codes]
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    results.sort(key=lambda x: x['score'], reverse=True)
    table = '<table><tr><th>代码</th><th>现价</th><th>概率</th><th>评分</th><th>建议</th><th>RSI</th><th>MACD</th><th>主力净流入(万)</th></tr>'
    for r in results:
        table += f'<tr><td>{r["code"]}</td><td>{r["price"]:.2f}</td><td>{r["prob"]}%</td><td>{r["score"]}</td><td>{r["advice"]}</td><td>{r["rsi"]:.1f}</td><td>{r["macd"]}</td><td>{r["flow"]:.0f}</td></tr>'
    table += '</table>'
    return table if results else '没有符合条件的结果'

# 自选股相关（同前，略加修改以显示名称）
@app.route('/watchlist/add', methods=['POST'])
def add_watchlist():
    code = request.form.get('code')
    wl = load_watchlist()
    if code not in wl:
        wl.append(code)
        save_watchlist(wl)
    return 'ok'

@app.route('/watchlist/remove', methods=['POST'])
def remove_watchlist():
    code = request.form.get('code')
    wl = load_watchlist()
    if code in wl:
        wl.remove(code)
        save_watchlist(wl)
    return 'ok'

@app.route('/watchlist/refresh')
def refresh_watchlist():
    wl = load_watchlist()
    if not wl:
        return '<p>暂无自选</p>'
    rows = ''
    for code in wl:
        try:
            df = get_stock_data(code)
            df = compute_all_indicators(df)
            score, prob = deep_score(df, get_market_env(), load_weights())
            last = df.iloc[-1]
            name = get_stock_name(code) or ''
            rows += f'<tr><td>{code} {name}</td><td>{last["close"]:.2f}</td><td>{prob}%</td><td>{score}</td><td><button onclick="removeWatchlist(\'{code}\')">删除</button></td></tr>'
        except:
            rows += f'<tr><td>{code}</td><td colspan="3">失败</td><td><button onclick="removeWatchlist(\'{code}\')">删除</button></td></tr>'
    return f'<table><tr><th>代码/名称</th><th>现价</th><th>概率</th><th>评分</th><th>操作</th></tr>{rows}</table>'

# 提醒
@app.route('/alert/check')
def alert_check():
    wl = load_watchlist()
    alerts = []
    for code in wl:
        try:
            df = get_stock_data(code)
            df = compute_all_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            if prev['macd_diff'] <= 0 and last['macd_diff'] > 0:
                alerts.append({'code': code, 'msg': 'MACD金叉'})
            if prev['macd_diff'] >= 0 and last['macd_diff'] < 0:
                alerts.append({'code': code, 'msg': 'MACD死叉'})
            if last['rsi'] > 80:
                alerts.append({'code': code, 'msg': 'RSI超买'})
            if last['rsi'] < 20:
                alerts.append({'code': code, 'msg': 'RSI超卖'})
        except:
            pass
    return jsonify({'alerts': alerts})

# 回测
@app.route('/backtest', methods=['POST'])
def backtest_route():
    code = request.form.get('code')
    start = request.form.get('start')
    end = request.form.get('end')
    try:
        df = get_stock_data(code)
        df = compute_all_indicators(df)
        result = backtest_strategy(df, start, end)
        return f"<p>总收益: {result['total_return']}% | 资金: {result['final_capital']} | 交易次数: {result['trade_count']} | 胜率: {result['win_rate']}%</p>"
    except Exception as e:
        return f"回测失败: {e}"

@app.route('/sector')
def sector():
    sectors = get_sector_heat()
    html = ''
    for sec in sectors:
        html += f'<h3>{sec["name"]} ({sec["pct"]}%)</h3>'
        stocks = get_top_stocks_in_sector(sec['code'])
        for s in stocks:
            html += f'<span>{s["code"]} {s["name"]} {s["pct"]}%</span><br>'
    return html

def load_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_WEIGHTS.copy()

@app.route('/weights', methods=['GET', 'POST'])
def weights():
    if request.method == 'POST':
        data = request.get_json()
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(data, f)
        return 'ok'
    else:
        return jsonify(load_weights())

if __name__ == '__main__':
    print("量化终端 Pro 已启动 → http://0.0.0.0:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)