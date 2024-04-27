import numpy as np

def rsi(data, window=14):
    diff = np.diff(data)
    gain = (diff + np.abs(diff)) / 2
    loss = (-diff + np.abs(diff)) / 2

    avg_gain = np.convolve(gain, np.ones(window) / window, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window) / window, mode='valid')
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def macd(data, short_window=12, long_window=26):
    short_avg = np.convolve(data, np.ones(short_window) / short_window, mode='valid')
    long_avg = np.convolve(data, np.ones(long_window) / long_window, mode='valid')
    macd = short_avg[long_window - 1:] - long_avg
    
    return macd

def cci(high, low, close, window):
    typical_price = (high + low + close) / 3
    sma = np.convolve(typical_price, np.ones(window) / window, mode='valid')
    mean_dev = np.convolve(np.abs(typical_price - sma), np.ones(window) / window, mode='valid')
    
    cci = (typical_price[window - 1:] - sma) / (0.015 * mean_dev)
    
    return cci

def adx(high, low, close, window):
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr_sma = np.convolve(tr, np.ones(window) / window, mode='valid')
    
    pdm = np.maximum(high[1:] - high[:-1], 0)
    ndm = np.maximum(low[:-1] - low[1:], 0)
    
    pdm_sma = np.convolve(pdm, np.ones(window) / window, mode='valid')
    ndm_sma = np.convolve(ndm, np.ones(window) / window, mode='valid')
    
    pdi = (pdm_sma / tr_sma) * 100
    ndi = (ndm_sma / tr_sma) * 100
    
    dx = np.abs(pdi - ndi) / (pdi + ndi) * 100
    adx = np.convolve(dx, np.ones(window) / window, mode='valid')
    
    return adx


###########
def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def calculate_volatility(data, window_size):
    return data.rolling(window=window_size).std()

def calculate_momentum(data, window_size):
    return data - data.shift(window_size)

def calculate_rsi(data, window_size):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=window_size, min_periods=1).mean()
    avg_loss = -down.rolling(window=window_size, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))