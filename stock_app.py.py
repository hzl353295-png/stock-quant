# ========== 导入与基础配置 ==========
from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
import requests
import time
import re
import json
import os
import threading
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------- 优化点1：统一日志系统，错误分级可排查 --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

WEIGHTS_FILE = 'weights.json'
WATCHLIST_FILE = 'watchlist.json'
EXPERIMENT_LOG_FILE = 'experiments.json'

DEFAULT_WEIGHTS = {
    'macd': 30, 'rsi': 20, 'kdj': 15, 'ma': 20, 'volume': 10, 'market': 10,
    'flow': 10, 'pattern': 5
}

# 数据缓存
DATA_CACHE = {}
CACHE_TTL = 300
cache_lock = threading.Lock()

# -------------------------- 优化点4：股票名称字典加锁，修复多线程竞态问题 --------------------------
STOCK_NAME_DICT = {
    '600584': '长电科技', '600854': '春兰股份', '600519': '贵州茅台', 
    '000001': '平安银行', '300750': '宁德时代', '002475': '立讯精密',
    '002298': '中电兴发', '002218': '拓日新能', '000595': '宝塔实业',
    '000021': '深科技', '002156': '通富微电',
    '301583': '北方长龙', '688146': '中船特气', '688072': '拓荆科技',
    '001258': '立新能源', '688237': '超卓航科',
    '600000': '浦发银行', '600036': '招商银行', '601318': '中国平安',
    '000858': '五粮液', '002594': '比亚迪'
}
name_dict_lock = threading.Lock()

# 大盘环境缓存（批量选股时避免重复计算）
MARKET_ENV_CACHE = {'value': '中', 'timestamp': 0}
MARKET_ENV_TTL = 300

# -------------------------- 优化点1：统一请求头，伪装浏览器，降低反爬拦截概率 --------------------------
COMMON_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Referer': 'http://quote.eastmoney.com/',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9'
}

# ------------------- 文件操作 -------------------
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"自选股文件读取失败: {e}")
    return []

def save_watchlist(data):
    try:
        with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"自选股文件保存失败: {e}")

def load_experiments():
    if os.path.exists(EXPERIMENT_LOG_FILE):
        try:
            with open(EXPERIMENT_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"实验记录文件读取失败: {e}")
    return []

def save_experiment(data):
    experiments = load_experiments()
    experiments.append(data)
    try:
        with open(EXPERIMENT_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(experiments, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"实验记录保存失败: {e}")

def load_weights():
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"权重文件读取失败，使用默认权重: {e}")
    return DEFAULT_WEIGHTS.copy()

# ------------------- 股票名称查询 -------------------
def get_stock_name(code):
    # 先查本地缓存
    if code in STOCK_NAME_DICT:
        return STOCK_NAME_DICT[code]
    
    try:
        market = '1' if code.startswith('6') else '0'
        secid = f'{market}.{code}'
        url = 'https://push2.eastmoney.com/api/qt/stock/get'
        params = {
            'secid': secid, 
            'fields': 'f57', 
            'ut': 'fa5fd1943c7b386f172d6893d3e0f15e'
        }
        resp = requests.get(url, params=params, headers=COMMON_HEADERS, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        name = data.get('data', {}).get('f57', '')
        
        if name:
            # 多线程写入加锁
            with name_dict_lock:
                STOCK_NAME_DICT[code] = name
            return name
        logger.warning(f"股票{code}名称查询返回为空")
        return ""
    
    except requests.exceptions.RequestException as e:
        logger.error(f"股票{code}名称查询接口异常（可能被反爬）: {e}")
        return ""
    except Exception as e:
        logger.error(f"股票{code}名称查询未知错误: {e}")
        return ""

# ------------------- 实时行情 -------------------
def get_realtime_summary(code):
    try:
        market = '1' if code.startswith('6') else '0'
        secid = f'{market}.{code}'
        url = 'https://push2.eastmoney.com/api/qt/stock/get'
        params = {
            'secid': secid,
            'fields': 'f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f14,f15,f16,f17,f18',
            'ut': 'fa5fd1943c7b386f172d6893d3e0f15e'
        }
        resp = requests.get(url, params=params, headers=COMMON_HEADERS, timeout=5)
        resp.raise_for_status()
        data = resp.json().get('data', {})
        
        if not data:
            logger.warning(f"股票{code}实时行情返回空数据")
            return {'change': 0, 'change_pct': 0, 'turnover': 0, 'vol_ratio': 0.0}
        
        return {
            'price': float(data.get('f2', 0)),
            'change': float(data.get('f3', 0)),
            'change_pct': float(data.get('f4', 0)),
            'volume': float(data.get('f5', 0)),
            'turnover': float(data.get('f6', 0)),
            'high': float(data.get('f15', 0)),
            'low': float(data.get('f16', 0)),
            'vol_ratio': float(data.get('f17', 0)),
            'open': float(data.get('f14', 0))
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"股票{code}实时行情接口请求失败（疑似反爬拦截）: {e}")
        return {'change': 0, 'change_pct': 0, 'turnover': 0, 'vol_ratio': 0.0}
    except Exception as e:
        logger.error(f"股票{code}实时行情解析失败: {e}")
        return {'change': 0, 'change_pct': 0, 'turnover': 0, 'vol_ratio': 0.0}

# ------------------- 历史K线数据获取 -------------------
def get_stock_data(code):
    with cache_lock:
        if code in DATA_CACHE:
            ts, df = DATA_CACHE[code]
            if time.time() - ts < CACHE_TTL:
                return df.copy()
    
    df = _fetch_data(code)
    with cache_lock:
        DATA_CACHE[code] = (time.time(), df.copy())
    return df

def _fetch_data(code):
    try:
        return get_data_eastmoney(code)
    except Exception as e:
        logger.warning(f"东财数据源获取失败，切换新浪备用源: {e}")
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
        'ut': 'fa5fd1943c7b386f172d6893d3e0f15e'
    }
    
    data = None
    for i in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=COMMON_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('data') and data['data'].get('klines'):
                break
            logger.warning(f"东财K线第{i+1}次请求返回空数据，重试中...")
            time.sleep(1)
        except requests.exceptions.ConnectionError as e:
            logger.error(f"东财K线接口连接被断开（反爬拦截），第{i+1}次重试: {e}")
            if i == max_retries - 1:
                raise RuntimeError("东财接口反爬拦截，连接频繁断开") from e
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                raise RuntimeError(f"东财接口请求失败: {e}") from e
            time.sleep(1)
    
    if not data or not data.get('data') or not data['data'].get('klines'):
        raise ValueError("东财数据接口返回为空")
    
    lines = data['data']['klines']
    records = []
    for line in lines:
        parts = line.split(',')
        records.append({
            'date': parts[0],
            'open': float(parts[1]), 'close': float(parts[2]),
            'high': float(parts[3]), 'low': float(parts[4]),
            'volume': float(parts[5])
        })
    
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]

def get_data_sina(code):
    market = 'sh' if code.startswith('6') else 'sz'
    url = f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
    params = {
        'symbol': f'{market}{code}',
        'scale': '240',
        'ma': 'no',
        'datalen': '500'
    }
    
    try:
        resp = requests.get(url, params=params, headers=COMMON_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if not data or (isinstance(data, dict) and data.get('__ERROR')):
            raise ValueError("新浪接口返回错误或空数据")
        
        records = []
        for item in data:
            records.append({
                'date': item['day'],
                'open': float(item['open']), 'high': float(item['high']),
                'low': float(item['low']), 'close': float(item['close']),
                'volume': float(item['volume'])
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        logger.error(f"新浪备用数据源也获取失败: {e}")
        return pd.DataFrame()

# ------------------- 技术指标计算 -------------------
def compute_all_indicators(df):
    if df.empty:
        return df
        
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['signal']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    # 均线 + 布林带
    df['ma20'] = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = df['ma20'] + 2 * std20
    df['bb_lower'] = df['ma20'] - 2 * std20

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    # ✅ 核心：通过K线数据计算真实的量比，作为 0 值的兜底
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    return df

# ------------------- K线形态识别 -------------------
def detect_pattern(df):
    if len(df) < 3:
        return None, 0
    last = df.iloc[-1]; prev = df.iloc[-2]; prev2 = df.iloc[-3]
    o1, c1, h1, l1 = last['open'], last['close'], last['high'], last['low']
    o2, c2 = prev['open'], prev['close']
    o3, c3 = prev2['open'], prev2['close']
    body1 = abs(c1 - o1); body2 = abs(c2 - o2); body3 = abs(c3 - o3)

    if body1 > 0 and (o1 - l1) > 2*body1 and (h1 - c1) < body1*0.3 and c1 > o1:
        return "锤子线（看涨）", 1
    if body1 > 0 and (h1 - o1) > 2*body1 and (c1 - l1) < body1*0.3 and c1 < o1:
        return "上吊线（看跌）", -1
    if c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2:
        return "看涨吞没", 1
    if c2 > o2 and c1 < o1 and c1 < o2 and o1 > c2:
        return "看跌吞没", -1
    if (c3 < o3) and (abs(c2-o2) < body3*0.3) and (c1 > o1) and (c1 > (o3+c3)/2):
        return "早晨之星（看涨）", 1
    if (c3 > o3) and (abs(c2-o2) < body3*0.3) and (c1 < o1) and (c1 < (o3+c3)/2):
        return "黄昏之星（看跌）", -1
    if body1 < (h1 - l1) * 0.1:
        return "十字星（反转可能）", 0
    return None, 0

# ------------------- 大盘环境 -------------------
def get_market_env(df=None):
    if df is None:
        now = time.time()
        if now - MARKET_ENV_CACHE['timestamp'] < MARKET_ENV_TTL:
            return MARKET_ENV_CACHE['value']
        try:
            df = get_stock_data('000001')
            df = compute_all_indicators(df)
        except Exception as e:
            logger.error(f"大盘数据获取失败，默认中性: {e}")
            return '中'
    
    if df.empty or len(df) < 20:
        return '中'
    
    last = df.iloc[-1]
    score = 0
    if last['close'] > last['ma20']: score += 1
    if last['macd_diff'] > 0: score += 1
    if last['rsi'] > 40: score += 1
    
    result = '强' if score >= 2 else ('中' if score == 1 else '弱')
    
    if df is None:
        MARKET_ENV_CACHE['value'] = result
        MARKET_ENV_CACHE['timestamp'] = time.time()
    
    return result

# ------------------- 综合评分 -------------------
def deep_score(df, market_env='中', weights=None, flow=0, pattern_dir=0):
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    last = df.iloc[-1]; prev = df.iloc[-2]
    close, diff, prev_diff = last['close'], last['macd_diff'], prev['macd_diff']
    rsi, k, d, j, vol_ratio = last['rsi'], last['k'], last['d'], last['j'], last['vol_ratio']
    ma5, ma10, ma20 = last['ma5'], last['ma10'], last['ma20']

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
    if close > last['bb_upper']: score_ma += 8
    elif close < last['bb_lower']: score_ma += 12

    score_vol = 0
    if vol_ratio > 1.5 and diff > 0: score_vol = 18
    elif vol_ratio > 1.5 and diff < 0: score_vol = -15
    elif vol_ratio < 0.5: score_vol = -5

    score_market = 0
    if market_env == '弱': score_market = -15
    elif market_env == '强': score_market = 10

    if flow > 5000: score_flow = 15
    elif flow > 1000: score_flow = 10
    elif flow > 0: score_flow = 5
    elif flow < -5000: score_flow = -15
    elif flow < -1000: score_flow = -10
    else: score_flow = 0

    score_pattern = 10 if pattern_dir == 1 else (-10 if pattern_dir == -1 else 0)

    total = (
        score_macd * weights['macd']/30 +
        score_rsi * weights['rsi']/20 +
        score_kdj * weights['kdj']/15 +
        score_ma * weights['ma']/20 +
        score_vol * weights['volume']/10 +
        score_market * weights['market']/10 +
        score_flow * weights.get('flow',10)/10 +
        score_pattern * weights.get('pattern',5)/5
    )
    total = round(total)
    prob = max(0, min(100, round((total+90)/180*100)))
    return total, prob

# ------------------- 多策略对比 -------------------
def multi_strategy(df):
    market_env = get_market_env()
    s1 = deep_score(df, market_env, DEFAULT_WEIGHTS)
    
    w2 = DEFAULT_WEIGHTS.copy()
    w2['macd']=50; w2['rsi']=10; w2['kdj']=5; w2['ma']=10; w2['volume']=5
    s2 = deep_score(df, market_env, w2)
    
    w3 = DEFAULT_WEIGHTS.copy()
    w3['macd']=5; w3['rsi']=5; w3['kdj']=40; w3['ma']=30; w3['volume']=10
    s3 = deep_score(df, market_env, w3)
    
    w4 = DEFAULT_WEIGHTS.copy()
    w4['macd']=20; w4['rsi']=10; w4['kdj']=5; w4['ma']=15; w4['volume']=40
    s4 = deep_score(df, market_env, w4)
    
    def adv(score):
        if score>=35: return "强烈买入"
        if score>=15: return "可考虑买入"
        if score>=-10: return "观望"
        return "卖出/规避"
    
    return [
        {'name':'综合策略','score':s1[0],'prob':s1[1],'advice':adv(s1[0])},
        {'name':'纯MACD','score':s2[0],'prob':s2[1],'advice':adv(s2[0])},
        {'name':'KDJ+布林','score':s3[0],'prob':s3[1],'advice':adv(s3[0])},
        {'name':'量价优先','score':s4[0],'prob':s4[1],'advice':adv(s4[0])}
    ]

# ------------------- 回测引擎 -------------------
def backtest_strategy(df, start_date=None, end_date=None, init_capital=100000):
    commission_rate = 0.0003
    stamp_tax_rate = 0.001
    transfer_fee_rate = 0.00001
    
    df = df.copy()
    df = compute_all_indicators(df)
    
    try:
        market_df = get_stock_data('000001')
        market_df = compute_all_indicators(market_df)
    except Exception as e:
        logger.warning(f"回测大盘数据获取失败，默认中性环境: {e}")
        market_df = None

    if start_date: df = df[df.index >= pd.to_datetime(start_date)]
    if end_date: df = df[df.index <= pd.to_datetime(end_date)]
    
    if len(df) < 30:
        raise ValueError("可用数据量不足，无法进行有效回测")

    capital = init_capital
    position = 0
    buy_price = 0
    trades = []
    daily_values = []
    holding = False
    weights = load_weights()

    for i in range(len(df)-1):
        date_t = df.index[i]
        date_next = df.index[i+1]
        close_t = df.iloc[i]['close']
        open_next = df.iloc[i+1]['open']
        
        df_t = df.iloc[:i+1]
        
        if market_df is not None:
            market_t = market_df[market_df.index <= date_t]
            market_env_t = get_market_env(market_t)
        else:
            market_env_t = '中'
        
        _, pattern_dir = detect_pattern(df_t)
        score, _ = deep_score(df_t, market_env_t, weights, flow=0, pattern_dir=pattern_dir)
        
        buy_signal = (score >= 15) and (not holding)
        sell_signal = (score <= -10) and holding

        if buy_signal:
            buy_amount = capital * (1 - commission_rate - transfer_fee_rate)
            position = int(buy_amount / open_next / 100) * 100
            if position <= 0:
                continue
            total_cost = position * open_next * (1 + commission_rate + transfer_fee_rate)
            capital -= total_cost
            buy_price = open_next
            trades.append({'date': date_next, 'type': 'buy', 'price': open_next, 'shares': position})
            holding = True
        
        elif sell_signal:
            sell_amount = position * open_next * (1 - commission_rate - stamp_tax_rate - transfer_fee_rate)
            capital += sell_amount
            profit_pct = (sell_amount - position * buy_price * (1 + commission_rate + transfer_fee_rate)) / (position * buy_price * (1 + commission_rate + transfer_fee_rate))
            trades.append({'date': date_next, 'type': 'sell', 'price': open_next, 'profit_pct': profit_pct})
            position = 0
            holding = False
        
        daily_values.append(capital + position * close_t)
    
    if holding and len(df) > 0:
        last_close = df.iloc[-1]['close']
        capital += position * last_close * (1 - commission_rate - stamp_tax_rate - transfer_fee_rate)
        daily_values[-1] = capital
        holding = False

    total_return = (capital - init_capital)/init_capital * 100
    sell_trades = [t for t in trades if t['type']=='sell']
    trade_count = len(sell_trades)
    win_rate = len([t for t in sell_trades if t.get('profit_pct',0)>0]) / max(1,trade_count) * 100

    daily_values = np.array(daily_values)
    peak = np.maximum.accumulate(daily_values)
    drawdown = (peak - daily_values)/peak * 100
    max_drawdown = np.max(drawdown) if len(drawdown)>0 else 0

    if trade_count > 0:
        wins = [t['profit_pct'] for t in sell_trades if t.get('profit_pct',0)>0]
        losses = [abs(t['profit_pct']) for t in sell_trades if t.get('profit_pct',0)<0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 1
        profit_factor = round(avg_win/avg_loss, 2) if avg_loss>0 else 0
    else:
        profit_factor = 0

    if len(daily_values) > 1:
        daily_returns = np.diff(daily_values) / daily_values[:-1]
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe = round((mean_ret/std_ret)*np.sqrt(252), 2) if std_ret>0 else 0
    else:
        sharpe = 0

    return {
        'total_return': round(total_return,2),
        'final_capital': round(capital,2),
        'trade_count': trade_count,
        'win_rate': round(win_rate,2),
        'max_drawdown': round(max_drawdown,2),
        'profit_factor': profit_factor,
        'sharpe': sharpe
    }

# ------------------- 板块热度 -------------------
def get_sector_heat():
    try:
        url = 'https://push2.eastmoney.com/api/qt/clist/get'
        params = {
            'pn':'1','pz':'500','po':'1','np':'1',
            'ut':'bd1d9ddb04089700cf9c27f6f7426281',
            'fltt':'2','invt':'2','fid':'f3','fs':'m:90+t:2',
            'fields':'f2,f3,f4,f12,f14'
        }
        resp = requests.get(url, params=params, headers=COMMON_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [{'code':i['f12'],'name':i['f14'],'pct':i['f3']} for i in data['data']['diff'][:5]]
    except Exception as e:
        logger.error(f"板块热度获取失败: {e}")
        return []

def get_top_stocks_in_sector(sector_code):
    try:
        url = 'https://push2.eastmoney.com/api/qt/clist/get'
        params = {
            'pn':'1','pz':'10','po':'1','np':'1',
            'ut':'bd1d9ddb04089700cf9c27f6f7426281',
            'fltt':'2','invt':'2','fid':'f3',
            'fs':f'b:{sector_code}',
            'fields':'f2,f3,f12,f14'
        }
        resp = requests.get(url, params=params, headers=COMMON_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [{'code':i['f12'],'name':i['f14'],'pct':i['f3']} for i in data['data']['diff'][:5]]
    except Exception as e:
        logger.error(f"板块个股获取失败: {e}")
        return []

# ================== 前端 HTML ==================
HTML = '''
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>量化终端 Pro Max</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:"Microsoft YaHei",sans-serif; background:#1e1e2f; color:#eee; display:flex; flex-direction:column; }
        .sidebar { width:100%; background:#2a2a3d; display:flex; flex-wrap:wrap; gap:5px; padding:10px; position:sticky; top:0; z-index:100; }
        .sidebar button { background:#3a3a5a; color:white; border:none; padding:10px 8px; border-radius:6px; cursor:pointer; font-size:12px; flex:1 0 auto; transition: all 0.2s; }
        .sidebar button:hover { background:#5050a0; transform: translateY(-2px); }
        .main { flex:1; padding:15px; overflow-y:auto; }
        .tab { display:none; }
        .tab.active { display:block; }
        .card { background:#2a2a3d; border-radius:12px; padding:15px; margin-bottom:15px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
        h2 { font-size:18px; margin-bottom:15px; border-left: 4px solid #5050a0; padding-left: 10px; }
        input,textarea,select { background:#3a3a5a; border:none; color:white; padding:12px 15px; border-radius:8px; width:100%; margin:6px 0; font-size:14px; transition: all 0.2s; }
        input:focus, textarea:focus, select:focus { outline: 2px solid #5050a0; background:#40406b; }
        textarea { min-height: 60px; resize: vertical; }
        button.action-btn { background:#5050a0; color:white; border:none; padding:12px 20px; border-radius:8px; cursor:pointer; margin-top:10px; font-size:14px; font-weight:bold; transition: 0.2s; }
        button.action-btn:hover { background:#6060c0; transform: translateY(-2px); }
        button.small-btn { background:#3a3a5a; padding:6px 12px; margin:2px; }
        button.small-btn:hover { background:#5050a0; }
        .filter-group { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-top:12px; }
        .filter-item { display:inline-flex; align-items:center; background:#3a3a5a; padding:0 12px 0 16px; border-radius:8px; height:42px; font-size:13px; color:#ccc; gap:8px; }
        .filter-item input { background:transparent; border:none; color:white; width:45px; text-align:center; padding:0; margin:0; font-size:14px; }
        .filter-item input:focus { outline:none; }
        .filter-item select { background:transparent; border:none; color:white; padding:0; margin:0; font-size:14px; cursor:pointer; }
        .filter-item select:focus { outline:none; }
        table { width:100%; border-collapse:collapse; margin-top:15px; font-size:13px; }
        th { background:#40406b; padding:10px; text-align:left; font-weight:bold; color:#d0d0f0; }
        td { padding:10px; border-bottom:1px solid #3a3a5a; }
        tr:hover { background:#3a3a5a; }
        .green { color:#00b894; } .red { color:#d63031; } .yellow { color:#fdcb6e; }
        .signal-box { padding:15px; border-radius:8px; margin:15px 0; text-align:center; font-size:16px; font-weight:bold; }
        .strong-buy { background:#00b89433; border:1px solid #00b894; color:#00b894; }
        .buy { background:#55efc433; border:1px solid #55efc4; color:#55efc4; }
        .wait { background:#0984e333; border:1px solid #0984e3; color:#0984e3; }
        .sell { background:#d6303133; border:1px solid #d63031; color:#d63031; }
        .stock-name { font-size:18px; font-weight:bold; color:#f0a030; margin-bottom:5px; }
        .pos-info { color:#eee; font-size:14px; margin:10px 0; background:#36364f; padding:10px; border-radius:6px; }
        .result-meta { display:flex; flex-wrap:wrap; gap:10px 25px; padding:10px 0 5px 0; color:#b0b0d0; font-size:14px; align-items:center; }
        .result-meta span { background:#2a2a3d; padding:4px 12px; border-radius:4px; border:1px solid #444; }
        .ai-tip { background:#36364f; border-left: 4px solid #f0a030; padding:10px 15px; border-radius:4px; margin:10px 0; font-size:14px; line-height:1.6; color:#e0e0f0; }
        .info-tag { display:inline-block; background:#3a3a5a; padding:4px 12px; border-radius:4px; margin:3px 3px; font-size:13px; }
        @media (min-width: 768px) {
            body { flex-direction:row; }
            .sidebar { width:200px; flex-direction:column; position:static; height:100vh; }
            .sidebar button { font-size:14px; flex:none; padding:15px 10px; }
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
    <button onclick="showTab('exp')">🧪 实验记录</button>
</div>
<div class="main">
    <div id="single" class="tab active">
        <div class="card">
            <h2>单股深度分析</h2>
            <input type="text" id="s_code" placeholder="股票代码（如 600584）"><br>
            <input type="number" id="s_shares" placeholder="持仓股数（0为空仓）"><br>
            <input type="number" id="s_cost" placeholder="成本价"><br>
            <button class="action-btn" onclick="analyzeSingle()">开始分析</button>
            <div id="single_result"></div>
        </div>
    </div>
    <div id="batch" class="tab">
        <div class="card">
            <h2>智能选股</h2>
            <textarea id="b_codes" placeholder="输入代码，用逗号或空格分隔（如：600584,000001）"></textarea>
            <div class="filter-group">
                <div class="filter-item">最低评分 <input type="number" id="min_score" value="15"></div>
                <div class="filter-item">最低概率% <input type="number" id="min_prob" value="60"></div>
                <div class="filter-item">RSI上限 <input type="number" id="max_rsi" value="70"></div>
                <div class="filter-item">MACD <select id="macd_filter"><option value="">不限</option><option value="long">多头</option><option value="golden">金叉</option></select></div>
            </div>
            <button class="action-btn" onclick="analyzeBatch()">智能筛选</button>
            <div id="batch_result"></div>
        </div>
    </div>
    <div id="watchlist" class="tab">
        <div class="card">
            <h2>自选股</h2>
            <div style="display:flex; gap:10px; align-items:center;">
                <input type="text" id="wl_code" placeholder="输入代码" style="flex:1;">
                <button class="action-btn" onclick="addWatchlist()" style="width:auto; margin:0;">添加</button>
                <button class="action-btn" onclick="refreshWatchlist()" style="width:auto; margin:0;">刷新</button>
            </div>
            <div id="watchlist_result"></div>
        </div>
    </div>
    <div id="alert" class="tab">
        <div class="card">
            <h2>盘中提醒</h2>
            <button id="alert_toggle" class="action-btn" onclick="toggleAlert()">开启提醒</button>
            <div id="alert_status" style="margin-top:10px;"></div>
        </div>
    </div>
    <div id="backtest" class="tab">
        <div class="card">
            <h2>历史回测（逻辑统一版）</h2>
            <p style="color:#b0b0d0; font-size:13px; margin-bottom:15px;">✅ 信号与前端选股完全一致：综合评分≥15买入，≤-10卖出；T日收盘出信号，T+1日开盘成交；含印花税+佣金+过户费真实成本。</p>
            <input type="text" id="bt_code" placeholder="股票代码"><br>
            <input type="date" id="bt_start" value="2023-01-01"><br>
            <input type="date" id="bt_end"><br>
            <button class="action-btn" onclick="runBacktest()">运行回测</button>
            <div id="backtest_result"></div>
        </div>
    </div>
    <div id="sector" class="tab">
        <div class="card">
            <h2>板块热度</h2>
            <button class="action-btn" onclick="loadSectorHeat()">刷新板块</button>
            <div id="sector_result"></div>
        </div>
    </div>
    <div id="weights" class="tab">
        <div class="card">
            <h2>权重设置</h2>
            <div id="weights_inputs"></div>
            <button class="action-btn" onclick="saveWeights()">保存</button>
        </div>
    </div>
    
    <div id="exp" class="tab">
        <div class="card">
            <h2>🧪 量化实验日志 & 敌对审计</h2>
            <div style="display:flex; gap:15px; flex-wrap:wrap;">
                <button class="action-btn" style="flex:1;" onclick="generateAuditPrompt()">🕵️ 生成敌对审计 Prompt</button>
            </div>
            <div style="margin-top:15px; padding:10px; background:#2a2a3d; border-radius:6px;">
                <p style="color:#b0b0d0; font-size:13px; margin-bottom:8px;">复制上面生成的提示词，发给另外的AI模型帮您找策略漏洞。</p>
                <textarea id="audit_prompt_result" placeholder="点上方按钮一键生成审计 Prompt..." rows="4" readonly></textarea>
            </div>
            <hr style="border-color:#444; margin:20px 0;">
            <h3>记录本次实验</h3>
            <label>研究假设/观察：</label>
            <input type="text" id="exp_hypothesis" placeholder="例如：加入大盘过滤因子后能否降低回撤">
            <label>本次修改内容：</label>
            <input type="text" id="exp_change" placeholder="修改了什么参数/代码">
            <label>预期目标：</label>
            <input type="text" id="exp_goal" placeholder="年化提至20%，回撤降至15%">
            <label>实际结果（回测后自行填写）：</label>
            <input type="text" id="exp_result" placeholder="年化19%，回撤16%，未达预期...">
            <button class="action-btn" onclick="saveExperiment()">📝 保存实验记录</button>
            <div id="exp_list" style="margin-top:15px;"></div>
        </div>
    </div>
</div>

<script>
let alertInterval, alertActive = false;

function showTab(id) {
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    if (id === 'exp') loadExperiments();
}

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
        document.getElementById('alert_toggle').innerText = "开启提醒";
    } else {
        if (Notification.permission !== "granted") Notification.requestPermission();
        alertInterval = setInterval(checkSignals, 300000);
        alertActive = true;
        document.getElementById('alert_toggle').innerText = "关闭提醒";
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

async function loadWeights() {
    let resp = await fetch('/weights');
    let data = await resp.json();
    let html = '';
    for (let k in data) html += `<div class="filter-item" style="margin:5px;">${k} <input type="number" id="w_${k}" value="${data[k]}" style="width:50px;"></div> `;
    document.getElementById('weights_inputs').innerHTML = `<div class="filter-group">${html}</div>`;
}
async function saveWeights() {
    let weights = {};
    ['macd','rsi','kdj','ma','volume','market','flow','pattern'].forEach(k => {
        weights[k] = parseInt(document.getElementById('w_'+k).value) || 0;
    });
    await fetch('/weights', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(weights)});
    alert('权重已保存');
}
loadWeights();

function generateAuditPrompt() {
    let prompt = `我现在需要你对我的量化选股策略进行一次【敌对审计】。你的任务是专门找这个策略的致命漏洞和造假可能性，以证明这个回测收益可能是假的。
策略逻辑：基于MACD、RSI、KDJ、均线等多因子综合评分，多头市场选股，T日收盘发出信号，T+1日以开盘价成交。
因子表：MACD金叉/死叉、RSI超买超卖、KDJ、布林带、量价关系、形态识别。
请你重点检查以下问题：
1. 是否存在未来函数？（信号产生时间和数据可用时间是否匹配，有没有偷看未来数据）
2. 代码中是否有shift写反、跨股票滚动、未来一天数据滥用的问题？
3. 我的回测是否使用了完全真实、不考虑涨跌停/停牌/滑点的成交价？
4. 历史回测的高收益是否集中在特定的市场环境、大盘年份或特定行业中？
5. 策略容量和换手率成本在实盘中是否能落地？
6. 有没有多重检验导致的参数过拟合（例如参数19日、20日、21日差异巨大）？
7. 财务和行情数据是否严格使用了后复权？
请输出：风险等级、具体漏洞证据、需要追加的测试、可能的修复方式。请你像一个铁面无私的审计员一样挑刺，不要给我信心，专门给我找推翻的策略的理由。`;
    document.getElementById('audit_prompt_result').value = prompt;
}

async function saveExperiment() {
    let data = {
        'hypothesis': document.getElementById('exp_hypothesis').value,
        'change': document.getElementById('exp_change').value,
        'goal': document.getElementById('exp_goal').value,
        'result': document.getElementById('exp_result').value,
        'time': new Date().toLocaleString()
    };
    await fetch('/exp/save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
    loadExperiments();
    alert('实验记录已保存！');
}

async function loadExperiments() {
    let resp = await fetch('/exp/list');
    let data = await resp.json();
    if (data.length === 0) {
        document.getElementById('exp_list').innerHTML = '<p style="color:#666;">暂无记录，开始你的第一个量化实验吧！</p>';
        return;
    }
    let html = `<h3>过往实验记录（${data.length}次）</h3><table><tr><th>假设</th><th>修改内容</th><th>预期</th><th>结果</th><th>时间</th></tr>`;
    data.slice().reverse().forEach(d => {
        html += `<tr><td>${d.hypothesis}</td><td>${d.change}</td><td>${d.goal}</td><td>${d.result}</td><td>${d.time}</td></tr>`;
    });
    html += '</table>';
    document.getElementById('exp_list').innerHTML = html;
}
</script>
</body>
</html>
'''

# ================== 路由 ==================
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/single', methods=['POST'])
def single():
    code = request.form.get('stock_code')
    shares = int(request.form.get('shares', 0) or 0)
    cost = float(request.form.get('cost', 0) or 0)
    try:
        df = get_stock_data(code)
        if df.empty:
            return '<div class="ai-tip">❌ 数据获取失败，请检查股票代码或稍后重试（接口可能被反爬）</div>'
        
        df = compute_all_indicators(df)
        market_env = get_market_env()
        pattern_name, pattern_dir = detect_pattern(df)
        weights = load_weights()
        score, prob = deep_score(df, market_env, weights, 0, pattern_dir)
        last = df.iloc[-1]
        
        summary = get_realtime_summary(code)
        
        diag_msg = []
        if last['close'] > last['ma20']: diag_msg.append("当前股价处于20日均线之上，中短期趋势偏强")
        else: diag_msg.append("当前股价位于20日均线之下，处于弱势震荡区间")
        if last['macd_diff'] > 0: diag_msg.append("MACD处于多头区间")
        else: diag_msg.append("MACD处于空头区间")
        if last['rsi'] > 70: diag_msg.append("RSI接近超买区，有回调风险")
        elif last['rsi'] < 30: diag_msg.append("RSI处于超卖区，可能反弹")
        else: diag_msg.append("RSI处于合理震荡区")
        
        # ✅ 修复量比：如果实时接口返回0，则用计算得到的K线量比兜底
        display_vol_ratio = summary['vol_ratio'] if summary['vol_ratio'] > 0 else round(last['vol_ratio'], 2)
        if display_vol_ratio > 1.5: diag_msg.append("当前量比大于1.5，成交活跃度提升")
        
        resistance = round(max(last['bb_upper'], df['high'].rolling(60).max().iloc[-1]), 2)
        support = round(min(last['bb_lower'], df['low'].rolling(60).min().iloc[-1]), 2)

        if score >= 35: advice="强烈买入"; cls="strong-buy"
        elif score >= 15: advice="可考虑买入"; cls="buy"
        elif score >= -10: advice="观望"; cls="wait"
        else: advice="卖出/规避"; cls="sell"
        
        atr = last['atr']
        stop_loss = round(last['close'] - 2*atr, 2)
        take_profit = round(last['close'] + 3*atr, 2)
        
        pos_html = ''
        if shares > 0 and cost > 0:
            profit = shares * (last['close'] - cost)
            pct = profit / (shares * cost) * 100
            pos_html += f"浮动盈亏: {profit:+.2f} ({pct:+.2f}%)<br>"
            
        multi = multi_strategy(df)
        multi_html = '<b>多策略对比：</b><table><tr><th>策略</th><th>评分</th><th>概率</th><th>建议</th></tr>'
        for m in multi:
            multi_html += f'<tr><td>{m["name"]}</td><td>{m["score"]}</td><td>{m["prob"]}%</td><td>{m["advice"]}</td></tr>'
        multi_html += '</table>'
        
        name = get_stock_name(code)
        sign_color = "green" if summary['change_pct'] > 0 else "red"
        
        html = f'''
        <div class="stock-name">{name} ({code})</div>
        <div class="signal-box {cls}"><b>{advice}</b> 概率{prob}%</div>
        <div class="pos-info">
            {pos_html}
            <span class="info-tag">今日涨跌幅: <span style="color: {sign_color}; font-weight:bold;">{summary['change_pct']:.2f}%</span></span>
            <span class="info-tag">换手率: {summary['turnover']:.2f}%</span>
            <span class="info-tag">量比: {display_vol_ratio:.2f}</span>
        </div>
        <div class="result-meta">
            <span>止损: {stop_loss}</span>
            <span>止盈: {take_profit}</span>
            <span>压力位: {resistance}</span>
            <span>支撑位: {support}</span>
            <span>形态: {pattern_name or "无典型"}</span>
        </div>
        <div class="ai-tip">💡 诊断建议：{"，".join(diag_msg)}</div>
        {multi_html}
        '''
        return html
    except Exception as e:
        logger.error(f"单股分析异常: {e}")
        return f'<div class="ai-tip" style="border-left-color:#d63031;">❌ 分析失败：{str(e)}</div>'

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
            if df.empty:
                return None
            df = compute_all_indicators(df)
            last = df.iloc[-1]
            market_env = get_market_env()
            pattern_name, pattern_dir = detect_pattern(df)
            weights = load_weights()
            score, prob = deep_score(df, market_env, weights, 0, pattern_dir)
            
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
            
            name = get_stock_name(code)
            summary = get_realtime_summary(code)
            
            # ✅ 修复量比：如果接口返回0，则用计算得到的K线量比兜底
            display_vol_ratio = summary['vol_ratio'] if summary['vol_ratio'] > 0 else round(last['vol_ratio'], 2)
            
            return {
                'code': code, 'name': name, 'price': last['close'],
                'prob': prob, 'score': score, 'advice': advice,
                'rsi': last['rsi'], 'macd': "多头" if last['macd_diff']>0 else "空头",
                'change': summary['change_pct'], 'vol_ratio': display_vol_ratio
            }
        except Exception as e:
            logger.warning(f"批量分析股票{code}失败: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process, code) for code in codes]
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    if not results:
        return '<p style="color:#b0b0d0;">没有符合条件的结果，或全部数据获取失败</p>'
    
    table = '<table><tr><th>代码/名称</th><th>现价</th><th>涨跌幅</th><th>评分</th><th>建议</th><th>RSI</th><th>MACD</th><th>量比</th></tr>'
    for r in results:
        display_name = r["name"] if r["name"] != r["code"] else ""
        sign_color = "green" if r['change'] >= 0 else "red"
        table += f'<tr><td>{r["code"]} {display_name}</td><td>{r["price"]:.2f}</td><td style="color:{sign_color};">{r["change"]:.2f}%</td><td>{r["score"]}</td><td>{r["advice"]}</td><td>{r["rsi"]:.1f}</td><td>{r["macd"]}</td><td>{r["vol_ratio"]:.2f}</td></tr>'
    table += '</table>'
    return table

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
            name = get_stock_name(code)
            summary = get_realtime_summary(code)
            sign_color = "green" if summary['change_pct'] >= 0 else "red"
            rows += f'<tr><td>{code} {name}</td><td>{last["close"]:.2f}</td><td style="color:{sign_color};">{summary["change_pct"]:.2f}%</td><td>{score}</td><td><button class="small-btn" onclick="removeWatchlist(\'{code}\')">删除</button></td></tr>'
        except Exception as e:
            logger.warning(f"自选股{code}刷新失败: {e}")
            rows += f'<tr><td>{code}</td><td colspan="3">获取失败</td><td><button class="small-btn" onclick="removeWatchlist(\'{code}\')">删除</button></td></tr>'
    return f'<table><tr><th>代码/名称</th><th>现价</th><th>涨跌幅</th><th>评分</th><th>操作</th></tr>{rows}</table>'

@app.route('/alert/check')
def alert_check():
    wl = load_watchlist()
    alerts = []
    for code in wl:
        try:
            df = get_stock_data(code)
            df = compute_all_indicators(df)
            last = df.iloc[-1]; prev = df.iloc[-2]
            if prev['macd_diff'] <= 0 and last['macd_diff'] > 0:
                alerts.append({'code': code, 'msg': 'MACD金叉'})
            if prev['macd_diff'] >= 0 and last['macd_diff'] < 0:
                alerts.append({'code': code, 'msg': 'MACD死叉'})
            if last['rsi'] > 80:
                alerts.append({'code': code, 'msg': 'RSI超买'})
            if last['rsi'] < 20:
                alerts.append({'code': code, 'msg': 'RSI超卖'})
        except Exception as e:
            logger.warning(f"信号监控{code}异常: {e}")
    return jsonify({'alerts': alerts})

@app.route('/backtest', methods=['POST'])
def backtest_route():
    code = request.form.get('code')
    start = request.form.get('start')
    end = request.form.get('end')
    try:
        df = get_stock_data(code)
        if df.empty:
            return '<p style="color:#d63031;">回测失败：无法获取股票K线数据</p>'
        
        result = backtest_strategy(df, start, end)
        return f"""
        <p>✅ 回测完成（信号与选股逻辑完全统一，含印花税+佣金+过户费真实成本）</p>
        <div class="result-meta">
            <span>总收益: {result['total_return']}%</span>
            <span>最终资金: {result['final_capital']}元</span>
            <span>交易次数: {result['trade_count']}次</span>
            <span>胜率: {result['win_rate']}%</span>
            <span>最大回撤: {result['max_drawdown']}%</span>
            <span>盈亏比: {result['profit_factor']}</span>
            <span>夏普比率: {result['sharpe']}</span>
        </div>
        """
    except Exception as e:
        logger.error(f"回测异常: {e}")
        return f'<p style="color:#d63031;">回测失败: {str(e)}</p>'

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

@app.route('/weights', methods=['GET', 'POST'])
def weights():
    if request.method == 'POST':
        data = request.get_json()
        with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return 'ok'
    return jsonify(load_weights())

@app.route('/exp/save', methods=['POST'])
def save_exp():
    data = request.get_json()
    save_experiment(data)
    return 'ok'

@app.route('/exp/list')
def list_exp():
    return jsonify(load_experiments())

if __name__ == '__main__':
    # ✅ 适配云端 Render (自动识别环境端口，本地则仍用 5001)
    port = int(os.environ.get('PORT', 5001))
    print("="*50)
    print(f"量化终端 Pro Max 已启动，监听端口: {port}")
    print("="*50)
    app.run(host='0.0.0.0', port=port, debug=False)