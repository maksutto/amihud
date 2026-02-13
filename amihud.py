import math
import os

import multiprocess as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d

from filter_outliers import filter_outliers
from set_split import set_split
def hodges_tompkins(price_data, window=20, trading_periods=252, clean=True):

    log_return = (price_data["Close"] / price_data["Close"].shift(1)).apply(np.log)

    vol = log_return.rolling(window=window, center=False).std() * math.sqrt(
        trading_periods
    )

    h = window
    n = (log_return.count() - h) + 1

    adj_factor = 1.0 / (1.0 - (h / n) + ((h ** 2 - 1) / (3 * n ** 2)))

    result = vol * adj_factor

    if clean:
        return result.dropna()
    return result



# Функция для расчета ILLIQ
def calculate_illiq(ticker, start_date, end_date, DIX):
    """
    Расчет показателя ILLIQ для указанного тикера за период
    
    Parameters:
    ticker (str): Тикер акции
    start_date (str): Начальная дата в формате 'YYYY-MM-DD'
    end_date (str): Конечная дата в формате 'YYYY-MM-DD'
    
    Returns:
    DataFrame: Данные с расчетом ILLIQ по месяцам
    """
    # Загрузка данных
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    stock.columns = stock.columns.droplevel(1)
   
    if stock.empty:
        print(f"Не удалось загрузить данные для {ticker}")
        return pd.DataFrame()
    
    # Расчет дневной доходности
    stock['Return'] = stock['Close'].pct_change()
    
    # Расчет долларового объема (Volume * Adj Close)
    stock['Dollar_Volume'] = stock['Volume'] * stock['Close']
    
    # Создание месячного индекса
    stock['Year_Month'] = stock.index.to_period('M')
    DIX['Year_Month'] = DIX.index.to_period('M')
    stock = pd.concat([stock,DIX['4'],DIX['2']],axis=1)
    stock['Dix_volume'] = (stock['4']-stock['2']) * stock['Close']
    stock['Dix_volume_sell'] = stock['2'] * stock['Close']
    

    # Группировка по месяцу и расчет показателей
    monthly_data = stock.groupby('Year_Month').apply(
        lambda x: pd.Series({
            'Monthly_Return': (1 + x['Return']).prod() - 1,
            'Monthly_Dollar_Volume': x['Dollar_Volume'].sum(),'Monthly_Dollar_Volume_dix': x['Dix_volume'].sum(),'Monthly_Dollar_Volume_dix_sell': x['Dix_volume_sell'].sum(),
            'Trading_Days': len(x)
        })
    ).reset_index()
    
    # Фильтрация месяцев с менее чем 15 торговыми днями
    monthly_data = monthly_data[monthly_data['Trading_Days'] >= 5]
    
    # Расчет ILLIQ
    monthly_data['ILLIQ'] = np.abs(monthly_data['Monthly_Return']) / monthly_data['Monthly_Dollar_Volume']
    monthly_data['ILLIQ_dix'] = np.abs(monthly_data['Monthly_Return']) / monthly_data['Monthly_Dollar_Volume_dix']
    monthly_data['ILLIQ_dix_sell'] = np.abs(monthly_data['Monthly_Return']) / monthly_data['Monthly_Dollar_Volume_dix_sell']
    # Добавление тикера
    monthly_data['Ticker'] = ticker

    return monthly_data


# Функция для расчета ILLIQ для нескольких тикеров
def calculate_illiq_multiple(tickers, start_date, end_date, DIX):
    """
    Расчет ILLIQ для нескольких тикеров

    Parameters:
    tickers (list): Список тикеров
    start_date (str): Начальная дата
    end_date (str): Конечная дата

    Returns:
    DataFrame: Данные с расчетом ILLIQ для всех тикеров
    """
    all_data = []

    for ticker in tickers:
        print(f"Обрабатывается {ticker}...")
        data = calculate_illiq(ticker, start_date, end_date, DIX)
        if not data.empty:
            all_data.append(data)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()



    

def rank_iv(data, window=252):
    data['min_iv'] = data['vix'].rolling(window=window, min_periods=1).min()
    data['max_iv'] = data['vix'].rolling(window=window, min_periods=1).max()
    data['iv_rank'] = (data['vix'] - data['min_iv']) / (data['max_iv'] - data['min_iv']) * 100
    data['iv_rank'] = data['iv_rank'].clip(0, 100)

    def calculate_iv_percentile(series):
        if len(series) < window:
            return np.nan
        current_iv = series.iloc[-1]
        historical_iv = series.iloc[:-1]
        count_below = (historical_iv < current_iv).sum()
        return (count_below / len(historical_iv)) * 100

    data['iv_percentile'] = data['vix'].ffill().rolling(window=window + 1, min_periods=window + 1).apply(
        calculate_iv_percentile,
        raw=False,
    )
    return data


def load_ticker_market_data(path, tick, start_date, date_col='date', date_format='%d %b %Y'):
    df = pd.read_csv(path, index_col=0)
    df = df.loc[df['tiker'] == tick]
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df.index = df[date_col]
    df = df[~df.index.duplicated()]
    return df.iloc[df.index > start_date]


def apply_figure_ranges(fig, stock, start_date, max_xaxis=17, max_yaxis=22):
    last_date = stock.index.tolist()[-1]
    xaxis_updates = {f'xaxis{idx if idx > 1 else ""}': dict(range=[start_date, last_date]) for idx in range(1, max_xaxis + 1)}
    yaxis_updates = {f'yaxis{idx}': dict(zerolinecolor='black') for idx in range(2, max_yaxis + 1)}
    fig.update_layout(xaxis_rangeslider_visible=False, **xaxis_updates, **yaxis_updates)


def build_suffix(ultraverse_df, temp_sig_z, speed, vanna_low, hard_sig):
    checks = [
        ((ultraverse_df['ultraverse'].iloc[-10:] > 0).any(), '_CYAN_'),
        ((ultraverse_df['ultraverse8'].iloc[-10:] > 0).any(), '_CYAN_'),
        ((ultraverse_df['ultraverse_3'].iloc[-10:] > 0).any(), '_BLUE_'),
        ((ultraverse_df['ultraverse_4'].iloc[-10:] > 0).any(), '_PURPLE_'),
        ((temp_sig_z['z_sig'].iloc[-10:] > 0).any(), '_RED_'),
        ((ultraverse_df['ultraverse7'].iloc[-10:] > 0).any(), '_CRIMSON_'),
        (((speed['calls'].ffill() > -speed['puts'].ffill()).iloc[-10:] > 0).any(), '_SPEED_GREEN_'),
        (((-(vanna_low['calls'] + vanna_low['puts'])/(vanna_low['calls'].abs() + vanna_low['puts'].abs())).iloc[-10:] < -0.8).any(), '_LOW_VANNA_'),
        ((hard_sig['hard_sig'].iloc[-10:] > 0.8).any(), '_hard_sig+++_'),
        ((hard_sig['hard_sig++++'].iloc[-10:] > 0.8).any(), '_hard_sig++++++++++++++++++_'),
    ]
    return ''.join(tag for matched, tag in checks if matched)



def Draw(tick,start_date = '2024-01-01'):
    # try:
        if tick == 'SPX':
            tick2 = '^SPX'
        else:
            tick2 = tick
        if tick == 'VIX':
            tick2 = '^VIX'
        
            
        try:
            loaded_DIX =pd.read_csv("C://Users//maksut//Dropbox//Market_stats//FINRA//"+tick+".csv",index_col=0)
        except:
            loaded_DIX =pd.DataFrame(columns=['2', '4'])
        loaded_DIX.index =pd.to_datetime(loaded_DIX.index,format='mixed')
        loaded_DIX = loaded_DIX.sort_index()
        loaded_DIX = loaded_DIX.iloc[loaded_DIX.index > start_date] 
        loaded_DIX = loaded_DIX .ffill()

        loaded_DIX['2'] = loaded_DIX['2']# * loaded_DIX['Close']
        loaded_DIX['4'] = loaded_DIX['4']# * loaded_DIX['Close']
        buy_vol = loaded_DIX['4'].ffill() - loaded_DIX['2']
        loaded_DIX['ratio'] = (buy_vol - loaded_DIX['2'])/loaded_DIX['4']*10
        buy_vol = buy_vol.rolling(30, min_periods=1).sum()
        sell_vol = loaded_DIX['2'].rolling(30, min_periods=1).sum()
        # dix = loaded_DIX['2'] / loaded_DIX['4']*100
        # dixB = (loaded_DIX['4'].ffill() - loaded_DIX['2']) / loaded_DIX['4']*100
        market_files = {
            'gamma_low': r'C:\Users\maksut\Dropbox\Market_stats\gamma_low_orig_range_summ_result_test_all.csv',
            'vanna_low': r'C:\Users\maksut\Dropbox\Market_stats\vanna_low_orig_range_summ_result_test_all.csv',
            'vanna_low2': r'C:\Users\maksut\Dropbox\Market_stats\gamma2_summ_result.csv',
            'gamma2_low': r'C:\Users\maksut\Dropbox\Market_stats\gamma_summ_result.csv',
            'gamma3_low': r'C:\Users\maksut\Dropbox\Market_stats\charm_range_summ_result_test.csv',
            'charm': r'C:\Users\maksut\Dropbox\Market_stats\charm3_range_summ_result_test.csv',
            'delta': r'C:\Users\maksut\Dropbox\Market_stats\delta2_range_summ_result_test.csv',
            'color': r'C:\Users\maksut\Dropbox\Market_stats\color_range_summ_result_test4.csv',
            'vanna_summ_': r'C:\Users\maksut\Dropbox\Market_stats\vanna2b_range_summ_result_test.csv',
            'zomma': r'C:\Users\maksut\Dropbox\Market_stats\zomma_range_summ_result_orig.csv',
            'vomma': r'C:\Users\maksut\Dropbox\Market_stats\vomma_original_range_summ_result_test.csv',
            'speed60': r'C:\Users\maksut\Dropbox\Market_stats\speed_range_summ_result_test30.csv',
            'speed': r'C:\Users\maksut\Dropbox\Market_stats\speed_range_summ_result_test.csv',
            'ultima': r'C:\Users\maksut\Dropbox\Market_stats\ultima_range_summ_result_test.csv',
        }
        market_data = {
            name: load_ticker_market_data(path, tick, start_date)
            for name, path in market_files.items()
        }

        gamma_low = market_data['gamma_low']
        vanna_low = market_data['vanna_low']
        vanna_low2 = market_data['vanna_low2']
        gamma2_low = market_data['gamma2_low']
        gamma3_low = market_data['gamma3_low']
        charm = market_data['charm']
        delta = market_data['delta']
        color = market_data['color']
        vanna_summ_ = market_data['vanna_summ_']
        zomma = market_data['zomma']
        vomma = market_data['vomma']
        speed60 = market_data['speed60']
        speed = market_data['speed']
        ultima = market_data['ultima']

        gamma_range = load_ticker_market_data(
            r'C:\Users\maksut\Dropbox\Market_stats\gamma2_range_summ_result_test.csv',
            tick,
            '2019-01-01',
        )
        gamma_range['calls'] = gamma_range['calls'].bfill().ffill()
        gamma_range['puts'] = gamma_range['puts'].bfill().ffill()

        vix = load_ticker_market_data(
            r'C:\Users\maksut\Dropbox\Market_stats\vix_result.csv',
            tick,
            '1900-01-01',
            date_col='gamma_date',
        )
        vix = rank_iv(vix)

        zomma = filter_outliers(zomma, 'calls', method='iqr', mode='remove_strong')
        zomma = filter_outliers(zomma, 'puts', method='iqr', mode='remove_strong')

        for frame in (speed60, speed):
            frame['puts'] = frame['puts'].bfill()
            frame['calls'] = frame['calls'].bfill()
            frame['puts'] = uniform_filter1d(frame['puts'], size=3)
            frame['calls'] = uniform_filter1d(frame['calls'], size=3)


        try:
            vb2_sig_conc = pd.read_csv(r'C:\\Users\\maksut\\Downloads\\signals\\'+ tick +"_vb2_sig.csv",index_col=0)   
            vb2_sig_conc.index = pd.to_datetime(vb2_sig_conc.index,format='mixed')
        except:
            vb2_sig_conc = pd.DataFrame()


        vex_voll = pd.read_csv("C:\\Users\\maksut\\Dropbox\\Market_stats\\vex_voll\\" + tick + "_vex_voll.csv",index_col=0)
        vex_voll["date"]=pd.to_datetime(vex_voll["date"])
        vex_voll.index = vex_voll["date"]
        vex_voll = vex_voll.ffill()
        vex_voll = vex_voll[~vex_voll.index.duplicated()]
        vex_voll = vex_voll.iloc[vex_voll.index > start_date]
        
        
        vex_voll2 = pd.read_csv("C:\\Users\\maksut\\Dropbox\\Market_stats\\vex_voll\\" + tick + "_vex_voll2.csv",index_col=0)
        vex_voll2["date"]=pd.to_datetime(vex_voll2["date"])
        vex_voll2.index = vex_voll2["date"]
        vex_voll2 = vex_voll2[~vex_voll2.index.duplicated()]
        vex_voll2 = vex_voll2.loc[vex_voll2.index > start_date]

    
    
 
        
        stock = yf.download(tick2, start=start_date, progress=False)
        stock.columns = stock.columns.droplevel(1)
        
        stock['Volume']= stock['Volume'] * stock['Close']

        
        temp_df = pd.concat([(gamma_low['puts'].ffill()-gamma_low['calls'].ffill()),stock['Volume']],axis=1,keys = ['gamma','vol']).ffill()
        temp_df['temp'] = temp_df['gamma'].abs() - (temp_df['vol']*10)
        
        stock['vix'] = hodges_tompkins(stock.ffill(), window=180, trading_periods=252, clean=True)*100
        stock = rank_iv(stock)
    
        delta['diff'] = (delta['calls'] + delta['puts'])
        delta['diff_z'] = delta['diff'].sub(delta['diff'].rolling(60).mean()).div(delta['diff'].rolling(60).std())
        stock['Close_z'] = stock['Close'].sub(stock['Close'].rolling(30).mean()).div(stock['Close'].rolling(30).std())
        delta['test'] = (-delta['puts'] - (delta['calls']+delta['puts']))/-delta['puts']
        vanna_low2['calls_z'] = vanna_low2['calls'].sub(vanna_low2['calls'].rolling(30).mean()).div(vanna_low2['calls'].rolling(30).std())
        vanna_low2['puts_z'] = vanna_low2['puts'].sub(vanna_low2['puts'].rolling(30).mean()).div(vanna_low2['puts'].rolling(30).std())   
        vanna_low2['summ_ratio'] = (vanna_low2['calls'].bfill().ffill() + vanna_low2['puts'].bfill().ffill())/(vanna_low2['calls'].abs().bfill().ffill() + vanna_low2['puts'].abs().bfill().ffill())

        vanna_low['ratio_calls'] = -(vanna_low['calls'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs())
        vanna_low['ratio_puts'] = -(vanna_low['puts'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs())

    
        vanna_low['calls_z'] = vanna_low['ratio_calls'].sub(vanna_low['ratio_calls'].rolling(30).mean()).div(vanna_low['ratio_calls'].rolling(30).std())
        vanna_low['puts_z'] = vanna_low['ratio_puts'].sub(vanna_low['ratio_puts'].rolling(30).mean()).div(vanna_low['ratio_puts'].rolling(30).std())   


        gamma2_low['diff'] = -(gamma2_low['puts'].bfill()+gamma2_low['calls'].bfill())
        gamma2_low['gl_z'] = gamma2_low['diff'].sub(gamma2_low['diff'].rolling(60).mean()).div(gamma2_low['diff'].rolling(60).std())   
        vanna_low['diff'] = -(vanna_low['puts'] + vanna_low['calls'])     
        vanna_low['vl_z'] = vanna_low['diff'].sub(vanna_low['diff'].rolling(60).mean()).div(vanna_low['diff'].rolling(60).std())   
        vanna_low['calls_z'] = vanna_low['calls'].sub(vanna_low['calls'].rolling(30).mean()).div(vanna_low['calls'].rolling(30).std())   
    
    
        delta['puts_z'] = delta['puts'].sub(delta['puts'].rolling(30).mean()).div(delta['puts'].rolling(30).std())
        delta['calls_z'] = delta['diff'].sub(delta['diff'].rolling(30).mean()).div(delta['diff'].rolling(30).std())
        vanna_low['diff_cadet'] = -(vanna_low['puts'] - vanna_low['calls'])
        vanna_low['diff_cadet_z'] = vanna_low['diff_cadet'].sub(vanna_low['diff_cadet'].rolling(30).mean()).div(vanna_low['diff_cadet'].rolling(30).std())   
        vanna_low['diff_brown'] = -(vanna_low['puts'] + vanna_low['calls'])
        vanna_low['diff_brown_z'] = vanna_low['diff_brown'].sub(vanna_low['diff_brown'].rolling(30).mean()).div(vanna_low['diff_brown'].rolling(30).std())   
        vanna_low.loc[(vanna_low['diff_cadet_z'] > 2) & (-vanna_low['puts'] > 0) & (-vanna_low['diff_brown_z'] > 1.4),'ultraverse_z'] = 2


        if tick == "SPX":
            splits = yf.download('^SPX',start = '2018-01-01',actions=True)
            splits = splits.xs('^SPX', axis=1, level=1)
        elif tick == "VIX":
            splits = yf.download("^VIX",start = '2018-01-01',actions=True)
            splits = splits.xs("^VIX", axis=1, level=1)
        else:
            splits = yf.download(tick,start = '2018-01-01',actions=True)
            splits = splits.xs(tick, axis=1, level=1)

                
        # splits = yf.download(tick,period="10y",actions=True)
        # splits = splits.xs(tick, axis=1, level=1)
        splits.index = splits.index.strftime('%Y-%m-%d')
        split_date = splits.loc[splits['Stock Splits']>0]['Stock Splits']
        split_size = split_date.values
        split_date = split_date.index
        k = 0
        for s in split_size:
            vanna_low = set_split("vanna_old", vanna_low, s, split_date[k], stock, "/")
            delta = set_split("delta", delta , s, split_date[k], stock, "/")     
            gamma_range = set_split("gamma_range", gamma_range, s, split_date[k], stock,"/")
            k += 1
        
        gamma_range['calls_z'] = gamma_range['calls'].sub(gamma_range['calls'].rolling(30).mean()).div(gamma_range['calls'].rolling(30).std())   
        
        ultraverse_df = pd.concat([delta['calls']+delta['puts'],(vanna_low['puts'] + vanna_low['calls'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs()), -vanna_low['puts'],charm['puts'], delta['zero'],stock['Close'],(delta['calls'] + delta['puts'])/(delta['calls'].abs()+delta['puts'].abs()),vanna_low['L1'],delta['diff_z'],stock['Close_z'],-(vanna_low['puts'] - vanna_low['calls']),
                                   delta['test'],vanna_low['calls_z'],vanna_low['puts_z'],vanna_low['L6']],
                                  axis=1, keys = ['delta','vanna','vanna_p','charm_p','delta_z','stock','delta_ratio','vanna_L1', 'delta_diff_z', 'Close_z','vanna_low_diff','delta_test','calls_z','puts_z','vanna_low_L6'])

        ultraverse_df['delta_z_diff'] = (ultraverse_df['stock']- ultraverse_df['delta_z'])/ultraverse_df['stock'].abs()
        ultraverse_df['vanna_L1_diff'] = (ultraverse_df['stock']- ultraverse_df['vanna_L1'])/ultraverse_df['stock'].abs()
        # ultraverse_df = ultraverse_df.bfill()
        ultraverse_df.loc[(ultraverse_df['delta'] < 0) & (ultraverse_df['vanna'] > 0) & (ultraverse_df['vanna_p'] < 0) & (ultraverse_df['charm_p'] > 0) & (ultraverse_df['calls_z'] > 0),'ultraverse'] = 2
        ultraverse_df.loc[(ultraverse_df['delta_ratio'] > 0.1) & (ultraverse_df['vanna'] > 0) & (ultraverse_df['vanna_p'] < 0) & (ultraverse_df['charm_p'] > 0)& (ultraverse_df['delta_z_diff'] - ultraverse_df['vanna_L1_diff'] < 0),'ultraverse_weak'] = 2
        ultraverse_df.loc[(ultraverse_df['delta_z_diff'] < 0.008) & (ultraverse_df['delta_z_diff'] > 0) & (ultraverse_df['vanna'] > 0) & (ultraverse_df['charm_p'] > 0),'ultraverse_2'] = 2
        ultraverse_df.loc[(ultraverse_df['delta_z_diff'] < -0.07) & (ultraverse_df['delta_ratio'] > 0.5) & (ultraverse_df['charm_p'] > 0) & (ultraverse_df['delta_diff_z'] < -1.5)& (ultraverse_df['vanna_p'] < 0),'ultraverse_3'] = 2
        ultraverse_df.loc[(ultraverse_df['delta_z_diff'] < 0) & (ultraverse_df['delta_ratio'] < 0.15) & (ultraverse_df['charm_p'] > 0) & (ultraverse_df['delta_diff_z'] < -1.3) ,'ultraverse_4'] = 2

        ultraverse_df.loc[ (ultraverse_df['delta'] > 0) & (ultraverse_df['puts_z'] > 0)& (ultraverse_df['vanna'] > 0) & (ultraverse_df['charm_p'] > 0)& ((ultraverse_df['puts_z'] - ultraverse_df['calls_z']) > 1),'ultraverse6'] = 2
        ultraverse_df.loc[  (ultraverse_df['puts_z'] < -3) & (ultraverse_df['calls_z'] > 3),'ultraverse7'] = 2
        ultraverse_df.loc[ (ultraverse_df['calls_z'] > 0) & (ultraverse_df['vanna'] > 0.95) & (ultraverse_df['delta_ratio'] > 0),'ultraverse8'] = 2
        ultraverse_df.loc[ (ultraverse_df['calls_z'] > 2.8) & (ultraverse_df['vanna_low_L6'] < ultraverse_df['stock']),'ultraverse9'] = 3

    
        fig = make_subplots(rows=9, cols=1,vertical_spacing = 0.01,specs=[[{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}] , [{"secondary_y": True}]                        
                                               ])
        
        #ROW 1
        # fig.append_trace(go.Scatter(x = stock.index, y=stock['Close'], mode='lines',line=dict(color='black')), row=1, col=1,)         
        fig.append_trace(go.Candlestick(x=stock.index,
                                         open = stock['Open'],
                                         high = stock['High'],
                                         low = stock['Low'],
                                         close = stock['Close']), row=1, col=1,)
    
        fig.add_trace(go.Scatter(x =  sell_vol.index, y = sell_vol, mode='lines',line_width=0.4,line=dict(color='red'),name='vex_voll2'),secondary_y=True, row=1, col=1,)
        fig.add_trace(go.Scatter(x =  buy_vol.index, y = buy_vol, mode='lines',line_width=0.4,line=dict(color='green'),name='vex_voll2'),secondary_y=True, row=1, col=1,)
        
        # fig.add_trace(go.Scatter(x =  vntest.index, y = vntest['max_strike'], mode='lines',line=dict(color='orange'),name='vex_voll2'), row=1, col=1,)


        # fig.add_trace(go.Scatter(x =  gamma_low.index, y = gamma_low['L1'], mode='lines',line=dict(color='cyan'),name='vex_voll2'), row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  gamma_low.index, y = gamma_low['L2'],opacity = 1, mode='lines',line=dict(color='cyan'),name='vex_voll2'), row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  gamma_low.index, y = gamma_low['zero'],opacity = 1, mode='lines',line=dict(color='cyan'),name='vex_voll2'), row=1, col=1,)
    
        fig.add_trace(go.Scatter(x =  vanna_low.index, y = vanna_low['L1'], mode='lines',line=dict(color='gray'),name='vex_voll2'), row=1, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.index, y = vanna_low['L2'],opacity = 0.6, mode='lines',line=dict(color='gray'),name='vex_voll2'), row=1, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.index, y = vanna_low['L3'], mode='lines',line=dict(color='green'),name='vex_voll2'), row=1, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.index, y = vanna_low['L4'], mode='lines',line=dict(color='green'),name='vex_voll2'), row=1, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.index, y = vanna_low['L5'], mode='lines',line=dict(color='crimson'),name='vex_voll2'), row=1, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.index, y = vanna_low['L6'], mode='lines',line=dict(color='crimson'),name='vex_voll2'), row=1, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.index, y = vanna_low['zero'], mode='lines',line=dict(color='violet'),name='vex_voll2'), row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  charm.index, y = charm['zero'], mode='lines',line=dict(color='crimson'),line_width=0.6,name='vex_voll2'), row=1, col=1,)
        fig.add_trace(go.Scatter(x =  delta.index, y = delta['zero'], mode='lines',line=dict(color='orange'),name='vex_voll2'), row=1, col=1,)
        fig.add_trace(go.Bar(x = ultraverse_df.index, y = ultraverse_df['ultraverse'],opacity = 0.6, marker = dict(color='cyan')),secondary_y=True, row=2, col=1)
        fig.add_trace(go.Bar(x = ultraverse_df.index, y = ultraverse_df['ultraverse_2'],opacity = 0.6, marker = dict(color='brown')),secondary_y=True, row=2, col=1)
        fig.add_trace(go.Bar(x = ultraverse_df.index, y = ultraverse_df['ultraverse_weak'],opacity = 0.6, marker = dict(color='violet')),secondary_y=True, row=2, col=1)
        fig.add_trace(go.Bar(x = ultraverse_df.index, y = ultraverse_df['ultraverse_3'],opacity = 0.6, marker = dict(color='blue')),secondary_y=True, row=2, col=1)
        fig.add_trace(go.Bar(x = ultraverse_df.index, y = ultraverse_df['ultraverse_4'],opacity = 0.6, marker = dict(color='#7700ff')),secondary_y=True, row=2, col=1)

        fig.add_trace(go.Bar(x = ultraverse_df.index, y = ultraverse_df['ultraverse6'],opacity = 0.6, marker = dict(color='#fa0771')),secondary_y=True, row=2, col=1)
        fig.add_trace(go.Bar(x = ultraverse_df.index, y = ultraverse_df['ultraverse7'],opacity = 0.6, marker = dict(color='#fa0771')),secondary_y=True, row=4, col=1)
        fig.add_trace(go.Bar(x = ultraverse_df.index, y = ultraverse_df['ultraverse8'],opacity = 0.6, marker = dict(color='cyan')),secondary_y=True, row=4, col=1)

        
        
        
        # fig.add_trace(go.Scatter(x =  zomma.index, y = zomma['bottom_L1'], mode='lines',line=dict(color='cyan'),name='vex_voll2'), row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  zomma.index, y = zomma['bottom_L2'], mode='lines',line=dict(color='purple'),line_width=0.8,name='vex_voll2'), row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  zomma.index, y = zomma['top_L1'], mode='lines',line=dict(color='purple'),line_width=0.8,name='vex_voll2'), row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  zomma.index, y = zomma['top_L2'], mode='lines',line=dict(color='purple'),line_width=0.8,name='vex_voll2'), row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  zomma.index, y = zomma['top_L3'], mode='lines',line=dict(color='purple'),line_width=0.8,name='vex_voll2'), row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  zomma.index, y = zomma['zero'], mode='lines',line=dict(color='cyan'),name='vex_voll2'), row=1, col=1,)

    
        fig.add_trace(go.Scatter(x =  charm.index, y = charm['puts'], mode='lines',line=dict(color='red'),name='vex_voll2'), row=2, col=1,)
        fig.add_trace(go.Scatter(x =  charm.index, y = charm['calls'], mode='lines',line=dict(color='blue'),name='vex_voll2'), row=2, col=1,)

        
        # fig.add_trace(go.Scatter(x =  checking.ffill().index, y = checking['ILLIQ_dix'].rolling(100).sum(), mode='lines',line=dict(color='green'),name='vex_voll2'),secondary_y=True, row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  checking.ffill().index, y = checking['ILLIQ'].rolling(100).sum(), mode='lines',line=dict(color='blue'),name='vex_voll2'),secondary_y=True, row=1, col=1,)
        # fig.add_trace(go.Scatter(x =  checking.ffill().index, y = checking['ILLIQ_dix_sell'].rolling(100).sum(), mode='lines',line=dict(color='red'),name='vex_voll2'),secondary_y=True, row=1, col=1,)
        
        fig.append_trace(go.Scatter(x = stock.index, y=stock['Close']/stock['Close'],opacity=0, mode='lines',line=dict(color='black')), row=2, col=1,) 
        # fig.add_trace(go.Scatter(x =  gamma_low.ffill().index, y = -(gamma_low['puts'].ffill()-gamma_low['calls'].ffill())/((gamma_low['puts'].ffill()+gamma_low['calls'].ffill())), mode='lines',line=dict(color='navy'),name='gamma_low_ratio'),secondary_y=True, row=2, col=1,)

        # fig.add_trace(go.Scatter(x =  gamma2_low.ffill().index, y = -(gamma2_low['puts'].ffill()+gamma2_low['calls'].ffill())/(gamma2_low['puts'].ffill().abs()+gamma2_low['calls'].ffill().abs()), mode='lines',line=dict(color='violet'),name='gamma2_low'),secondary_y=True, row=2, col=1,)
        # fig.add_trace(go.Scatter(x =  vanna_summ_.ffill().index, y = (vanna_summ_['puts'].ffill()+vanna_summ_['calls'].ffill())/(vanna_summ_['puts'].ffill().abs()+vanna_summ_['calls'].ffill().abs()), mode='lines',line=dict(color='purple'),name='vanna_summ_'),secondary_y=True, row=2, col=1,)



    

    
        '''SHORT''' #vanna_low['calls'] < (vanna_low['puts'] - vanna_low['calls']), vanna_low['calls_z'] < 0,  vanna_low['puts_z'] < 0
    
        # fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = -vanna_low['puts'], mode='lines',line=dict(color='red'),name='vex_voll2'), row=3, col=1,)
        # fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = -vanna_low['calls'], mode='lines',line=dict(color='green'),name='vex_voll2'), row=3, col=1,)
        # fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = -(vanna_low['puts'] - vanna_low['calls']), mode='lines',line=dict(color='cadetblue'),name='vex_voll2'), row=3, col=1,)
        # fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = (vanna_low['puts'] + vanna_low['calls']), mode='lines',line=dict(color='brown'),name='vex_voll2'), row=3, col=1,)
        # fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = (vanna_low['puts'] - vanna_low['calls']), mode='lines',line=dict(color='cadetblue'),opacity = 0.5,name='vex_voll2'), row=3, col=1,)
        # fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = (vanna_low['puts'] + vanna_low['calls'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs()), mode='lines',line=dict(color='navy'),name='vex_voll2'),secondary_y=True, row=3, col=1,)
        


    
        fig.append_trace(go.Scatter(x = stock.index, y=stock['Close']/stock['Close'],opacity=0, mode='lines',line=dict(color='black')), row=4, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = -vanna_low['puts'], mode='lines',line=dict(color='red'),name='vex_voll2'), row=4, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = vanna_low['calls'], mode='lines',line=dict(color='green'),line_width=0.5,name='vex_voll2'), row=4, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = -vanna_low['calls'], mode='lines',line=dict(color='green'),name='vex_voll2'), row=4, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = -(vanna_low['puts'] - vanna_low['calls']), mode='lines',line=dict(color='cadetblue'),name='vex_voll2'), row=4, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = -(vanna_low['puts'] + vanna_low['calls']), mode='lines',line=dict(color='brown'),name='vex_voll2'), row=4, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = (vanna_low['puts'] - vanna_low['calls']), mode='lines',line=dict(color='cadetblue'),opacity = 0.5,name='vex_voll2'), row=4, col=1,)
    
        fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = (vanna_low['calls'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs()), mode='lines',line=dict(color='navy'),name='vex_voll2'),secondary_y=True, row=2, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = -(vanna_low['puts'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs()), mode='lines',line=dict(color='crimson'),name='vex_voll2'),secondary_y=True, row=2, col=1,)   
        fig.add_trace(go.Scatter(x =  vanna_low2.ffill().index, y = (vanna_low2['calls'])/(vanna_low2['puts'].abs() + vanna_low2['calls'].abs()),line_width=0.5, mode='lines',line=dict(color='navy'),name='vex_voll2'),secondary_y=True, row=2, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low2.ffill().index, y = -(vanna_low2['puts'])/(vanna_low2['puts'].abs() + vanna_low2['calls'].abs()),line_width=0.5, mode='lines',line=dict(color='crimson'),name='vex_voll2'),secondary_y=True, row=2, col=1,) 
        fig.append_trace(go.Scatter(x = stock.index, y=stock['Close']/stock['Close'],opacity=0, mode='lines',line=dict(color='black')), row=5, col=1,)  
        # fig.add_trace(go.Scatter(x =  delta.ffill().index, y = -delta['puts'].ffill(),line_width=0.3, mode='lines',line=dict(color='crimson'),name='vex_voll2'), row=5, col=1,)
        # fig.add_trace(go.Scatter(x =  delta.ffill().index, y = delta['calls'].ffill(),line_width=0.3, mode='lines',line=dict(color='blue'),name='vex_voll2'), row=5, col=1,)
        # fig.add_trace(go.Scatter(x =  delta.ffill().index, y = delta['calls']+delta['puts'],line_width=0.3, mode='lines',line=dict(color='cadetblue'),name='vex_voll2'), row=5, col=1,)
        # fig.add_trace(go.Scatter(x =  delta.ffill().index, y = delta['puts'] + (delta['calls']+delta['puts']),line_width=0.3, mode='lines',line=dict(color='cadetblue'),name='vex_voll2'), row=5, col=1,)

        fig.add_trace(go.Scatter(x =  vanna_low2.ffill().index, y = (vanna_low2['calls'])/(vanna_low2['puts'].abs() + vanna_low2['calls'].abs()) - (vanna_low2['puts'])/(vanna_low2['puts'].abs() + vanna_low2['calls'].abs()),line_width=0.5, mode='lines',line=dict(color='navy'),name='vex_voll2'),secondary_y=True, row=4, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.ffill().index, y = ((vanna_low['calls'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs())) - ((vanna_low['puts'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs())), mode='lines',line=dict(color='navy'),name='vex_voll2'),secondary_y=True, row=4, col=1,)


        fig.append_trace(go.Scatter(x = stock.index, y=stock['Close']/stock['Close'],opacity=0, mode='lines',line=dict(color='black')), row=3, col=1,)
        fig.add_trace(go.Scatter(x =  stock.index, y = stock['iv_rank'], mode='lines',opacity=0.3,line=dict(color='green'),name='vex_voll2'),secondary_y=True, row=7, col=1,)
        fig.add_trace(go.Scatter(x =  vix.index, y = vix['iv_rank'], mode='lines',opacity=0.3,line=dict(color='orange'),name='vex_voll2'),secondary_y=True, row=7, col=1,)
        # fig.add_trace(go.Scatter(x =  vix.index, y = vix['iv_percentile'], mode='lines',line=dict(color='blue'),name='vex_voll2'), row=3, col=1,)

        vanna_low_sigs = pd.concat([vanna_low['calls'],-vanna_low['puts'],-(ultima['puts'].ffill()-ultima['calls'].ffill())/(ultima['puts'].ffill().abs() + ultima['calls'].ffill().abs()), (-speed['puts'].ffill() - speed['calls'].ffill())/(speed['puts'].ffill().abs() + speed['calls'].ffill().abs()),
                                   (delta['calls'] + delta['puts'])/(delta['calls'].abs() + delta['puts'].abs())],axis=1, keys = ['calls','puts',   'ultima_ratio','speed','delta'])
        vanna_low_sigs.loc[(vanna_low_sigs['calls'] > 0)  & (vanna_low_sigs['ultima_ratio'] > 0.1) & (vanna_low_sigs['calls'] > vanna_low_sigs['puts']) & (vanna_low_sigs['speed'] < 0) & (vanna_low_sigs['ultima_ratio'] > vanna_low_sigs['delta']),'z_sig'] = 4
        vanna_low_sigs.loc[(vanna_low_sigs['calls'] > 0)  & (vanna_low_sigs['ultima_ratio'] > 0.1) & (vanna_low_sigs['puts'] < 0) ,'z_sig'] = 6


        temp_sig_z = pd.concat([vanna_low['calls_z'],(vanna_low['calls'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs())-(vanna_low['puts'])/(vanna_low['puts'].abs() + vanna_low['calls'].abs()), (-speed['puts'].ffill() - speed['calls'].ffill())/(speed['puts'].ffill().abs() + speed['calls'].ffill().abs()),-(vanna_low['puts'] + vanna_low['calls']),-vanna_low['puts'],
                               -(ultima['puts'].ffill()-ultima['calls'].ffill())/(ultima['puts'].ffill().abs() + ultima['calls'].ffill().abs())],
                                  axis=1, keys = ['calls_z','vanna_low_ratio',   'speed_ratio',   'vanna_low_brown',   'vanna_low_puts',   'ultima_ratio'])
        temp_sig_z.loc[(temp_sig_z['calls_z'] > 2.5) & (temp_sig_z['vanna_low_ratio'] > 0.4) & (temp_sig_z['speed_ratio'] < 0) & (temp_sig_z['vanna_low_brown'] > 0) & (temp_sig_z['vanna_low_puts'] > 0) & (temp_sig_z['ultima_ratio'] < 0.04) ,'z_sig'] = 6

        fig.add_trace(go.Bar(x = temp_sig_z.index, y = temp_sig_z['z_sig'],opacity = 1, marker = dict(color='red')),secondary_y=True, row=3, col=1)   
          
    
        fig.add_trace(go.Bar(x = vanna_low_sigs.index, y = vanna_low_sigs['z_sig'],opacity = 1, marker = dict(color='orange')),secondary_y=True, row=3, col=1) 
        try:
            fig.add_trace(go.Bar(x = vb2_sig_conc.index, y = vb2_sig_conc['vb2_sig'],opacity = 0.6, marker = dict(color='#3399ff')),secondary_y=True, row=3, col=1)   
            fig.add_trace(go.Bar(x = vb2_sig_conc.index, y = vb2_sig_conc['strong_sig'],opacity = 1, marker = dict(color='crimson')),secondary_y=True, row=3, col=1)      
        except:
            pass



        fig.add_trace(go.Scatter(x =  vanna_low['calls_z'].index, y = -vanna_low['calls_z'], mode='lines',line=dict(color='cyan'),name='vex_voll2'),secondary_y=True, row=3, col=1,)
    
        fig.add_trace(go.Scatter(x =  vanna_low2['calls_z'].index, y = -vanna_low2['calls_z'], mode='lines',line=dict(color='brown'),name='vex_voll2'),secondary_y=True, row=3, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low2['puts_z'].index, y = -vanna_low2['puts_z'], mode='lines',line=dict(color='pink'),name='vex_voll2'),secondary_y=True, row=3, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low2.index, y = -vanna_low2['summ_ratio'],line_width=0.5, mode='lines',line=dict(color='blue'),name='vex_voll2'), row=3, col=1,)



        fig.add_trace(go.Scatter(x =  delta['calls_z'].index, y = delta['calls_z'], mode='lines',line=dict(color='green'),name='vex_voll2'),secondary_y=True, row=3, col=1,)
        fig.add_trace(go.Scatter(x =  stock['Close_z'].index, y = stock['Close_z'], mode='lines',line=dict(color='black'),name='vex_voll2'),secondary_y=True, row=3, col=1,)    


    
        
        
        # fig.add_trace(go.Scatter(x =  delta.index, y = delta['diff_z'], mode='lines',line=dict(color='violet'),name='vex_voll2'), row=6, col=1,)
        # fig.add_trace(go.Scatter(x =  stock.index, y = stock['Close_z'], mode='lines',line=dict(color='black'),name='vex_voll2'), row=6, col=1,)
        # fig.add_trace(go.Scatter(x =  gamma2_low.index, y = gamma2_low['gl_z'], mode='lines',line=dict(color='blue'),name='vex_voll2'), row=6, col=1,)
        # fig.add_trace(go.Scatter(x =  vanna_low.index, y = vanna_low['vl_z'], mode='lines',line=dict(color='red'),name='vex_voll2'), row=6, col=1,)
    

        fig.add_trace(go.Scatter(x =  speed.index, y = (speed['calls'].ffill() + speed['puts'].ffill())/(speed['puts'].ffill().abs() + speed['calls'].ffill().abs()), mode='lines',line=dict(color='#00ff04'),name='vex_voll2'), row=5, col=1,)
        fig.add_trace(go.Scatter(x =  speed60.index, y = (speed60['calls'].ffill() + speed60['puts'].ffill())/(speed60['puts'].ffill().abs() + speed60['calls'].ffill().abs()), mode='lines',line_width=0.3,line=dict(color='#00ff04'),name='vex_voll2'), row=5, col=1,)




        fig.add_trace(go.Scatter(x =  gamma2_low.ffill().index, y = -(gamma2_low['puts'].ffill()-gamma2_low['calls'].ffill())/(gamma2_low['puts'].ffill().abs()+gamma2_low['calls'].ffill().abs()), mode='lines',line=dict(color='violet'),name='gamma2_low'), row=5, col=1,)
        # fig.add_trace(go.Scatter(x =  gamma2_low.ffill().index, y = (gamma2_low['puts'].ffill()-gamma2_low['calls'].ffill())/(gamma2_low['puts'].ffill().abs()+gamma2_low['calls'].ffill().abs()), mode='lines',line=dict(color='violet'),line_width=0.4,name='gamma2_low'), row=5, col=1,)
        fig.add_trace(go.Scatter(x =  zomma.index, y = (zomma['puts'] - zomma['calls']) / (zomma['puts'].abs() + zomma['calls'].abs()), mode='lines',line=dict(color='#ff4d00'),name='vex_voll2'), row=5, col=1,)
        fig.add_trace(go.Scatter(x =  ultima.index, y = -(ultima['puts'].ffill() - ultima['calls'].ffill())/(ultima['puts'].ffill().abs() + ultima['calls'].ffill().abs()), mode='lines',line=dict(color='purple'),name='vex_voll2'), row=5, col=1,)
        fig.add_trace(go.Scatter(x =  delta.ffill().index, y = (delta['calls'] + delta['puts'])/(delta['calls'].abs() + delta['puts'].abs()), mode='lines',line=dict(color='navy'),name='vex_voll2'), row=5, col=1,)
        fig.add_trace(go.Scatter(x = vomma.index, y = (vomma['calls']- vomma['puts']) / (vomma['calls'].abs()+vomma['puts'].abs()),opacity = 1, mode='lines',line=dict(color='cyan'),name='vex_voll2'), row=5, col=1,) 
    
        fig.add_trace(go.Scatter(x = vanna_low.index, y = (vanna_low['calls'] - vanna_low['puts'])/(vanna_low['calls'].abs() + vanna_low['puts'].abs()),opacity = 1, mode='lines',line=dict(color='blue'),name='vex_voll2'), row=5, col=1,) 
        fig.add_trace(go.Scatter(x = vanna_low.index, y = -(vanna_low['calls'] + vanna_low['puts'])/(vanna_low['calls'].abs() + vanna_low['puts'].abs()),opacity = 0.5, mode='lines',line=dict(color='blue'),name='vex_voll2'), row=5, col=1,) 


    
        fig.add_trace(go.Scatter(x =  zomma.index, y = ((zomma['puts'] - zomma['calls'])/(zomma['puts'].abs() + zomma['calls'].abs())).rolling(30).sum(), mode='lines',line=dict(color='#ff4d00'),name='vex_voll2'),secondary_y=True, row=6, col=1,)

    
        fig.add_trace(go.Scatter(x =  delta.ffill().index, y = ((delta['calls'] + delta['puts'])/(delta['calls'].abs() + delta['puts'].abs())).rolling(30).sum(), mode='lines',line=dict(color='navy'),name='vex_voll2'),secondary_y=True, row=6, col=1,)
    


        fig.add_trace(go.Scatter(x =  zomma.index, y = -zomma['puts'], mode='lines',line=dict(color='#ff4d00'),name='vex_voll2'),secondary_y=True, row=6, col=1,)
        fig.add_trace(go.Scatter(x =  zomma.index, y = zomma['puts'], mode='lines',line=dict(color='#ff4d00'),line_width=.5,name='vex_voll2'),secondary_y=True, row=6, col=1,)
        fig.add_trace(go.Scatter(x =  zomma.index, y = -zomma['calls'], mode='lines',line=dict(color='blue'),name='vex_voll2'),secondary_y=True, row=6, col=1,)
        fig.add_trace(go.Scatter(x =  zomma.index, y = (zomma['puts'] - zomma['calls']), mode='lines',name='zomma p-c'),secondary_y=True, row=6, col=1,)
        fig.add_trace(go.Scatter(x =  zomma.index, y = (zomma['puts'] + zomma['calls']),line=dict(color='#85a318'), mode='lines',name='zomma p+c'),secondary_y=True, row=6, col=1,)




    
        # Опционально: если интерполяция "перегибает", можно заменить на smoothed в этих точках:
        # df_clean.loc[outlier_mask, 'value'] = smoothed[outlier_mask]
        fig.add_trace(go.Scatter(x =  speed60.index, y = speed60['calls']/(speed60['calls'].abs()+speed60['puts'].abs()),line_width=.5, mode='lines',line=dict(color='green')), row=6, col=1,)
        fig.add_trace(go.Scatter(x =  speed60.index, y = -speed60['puts']/(speed60['calls'].abs()+speed60['puts'].abs()) ,line_width=.5, mode='lines',line=dict(color='red')), row=6, col=1,)

    



        fig.add_trace(go.Scatter(x =  vanna_low2.index, y = -((vanna_low2['calls'])/(vanna_low2['puts'].abs() + vanna_low2['calls'].abs())).diff(23), mode='lines',line=dict(color='brown'),name='vex_voll2'), row=7, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low2.index, y = -((vanna_low2['puts'])/(vanna_low2['puts'].abs() + vanna_low2['calls'].abs())).diff(23), mode='lines',line=dict(color='red'),name='vex_voll2'), row=7, col=1,)
        fig.add_trace(go.Scatter(x =  delta.index, y = ((delta['calls'] + delta['puts'])/(delta['calls'].abs() + delta['puts'].abs())).diff(23), mode='lines',line=dict(color='blue'),name='vex_voll2'), row=7, col=1,)
        fig.add_trace(go.Scatter(x =  stock.index, y = stock['Close'].diff(23)/stock['Close'].diff(23).abs().max(), mode='lines',line=dict(color='black'),name='vex_voll2'), row=7, col=1,)



    
        fig.add_trace(go.Scatter(x =  color.index, y = ((color['puts']-color['calls'])/(color['puts'].abs() + color['calls'].abs())), mode='lines',line=dict(color='cyan'),name='vex_voll2'), row=7, col=1,)
        # fig.add_trace(go.Scatter(x =  color.index, y = ((color['puts']+color['calls'])/(color['puts'].abs() + color['calls'].abs())),line_width=0.5, mode='lines',line=dict(color='cyan'),name='vex_voll2'), row=7, col=1,)


    
        fig.add_trace(go.Scatter(x =  vanna_low.index, y = -vanna_low['diff_cadet_z'], mode='lines',line=dict(color='#00ff04'),name='vex_voll2'), row=8, col=1,)
        fig.add_trace(go.Scatter(x =  vanna_low.index, y = vanna_low['diff_brown_z'], mode='lines',line=dict(color='brown'),name='vex_voll2'), row=8, col=1,)
        fig.add_trace(go.Bar(x = vanna_low.index, y = vanna_low['ultraverse_z'],opacity = 1, marker = dict(color='cyan')),secondary_y=True, row=8, col=1)
        fig.add_trace(go.Scatter(x =  stock['Close_z'].index, y = stock['Close_z'], mode='lines',line=dict(color='black'),name='vex_voll2'), row=8, col=1,)    

        summ_p_c = pd.DataFrame()
        summ_p_c_2 = pd.DataFrame()
        summ_p_c["summ_p_c"] = (vex_voll["puts"] - vex_voll["calls"]) / (vex_voll["puts"].abs() + vex_voll["calls"].abs())
        summ_p_c_2["summ_p_c"] = (vex_voll2["calls"] - vex_voll2["puts"]) / (vex_voll2["puts"].abs() + vex_voll2["calls"].abs())
        summ_p_c_2["summ_p_c3"] = (vex_voll2["calls"] + vex_voll2["puts"]) / (vex_voll2["puts"].abs() + vex_voll2["calls"].abs())
        summ_p_c.index = vex_voll["date"]
        summ_p_c_2.index = vex_voll2["date"]
        summ_p_c["summ_p_c_2"] = (vex_voll["calls"]+vex_voll["puts"]) / (vex_voll["puts"].abs() + vex_voll["calls"].abs())


        fig.append_trace(go.Scatter(x = stock.index, y=stock['Close']/stock['Close'],opacity=0, mode='lines',line=dict(color='black')), row=9, col=1,)
        fig.add_trace(go.Scatter(x=summ_p_c.index,y=summ_p_c["summ_p_c_2"], mode='lines',line=dict(color='blue'),line_width=1), row=9, col=1)
        fig.add_trace(go.Scatter(x=summ_p_c_2.index,y=summ_p_c_2["summ_p_c"]*(-1), line_width=2, mode='lines',line=dict(color='orange')), row=9, col=1)
        fig.add_trace(go.Scatter(x=summ_p_c_2.index,y=summ_p_c_2["summ_p_c3"]*(-1), line_width=1, mode='lines',line=dict(color='orange')), row=9, col=1)
        fig.add_trace(go.Scatter(x=summ_p_c.index,y = summ_p_c["summ_p_c"]*(-1), mode='lines',line=dict(color='blue'),line_width=2), row=9, col=1)       

      

        hard_sig = pd.concat([summ_p_c_2["summ_p_c3"]*(-1), summ_p_c["summ_p_c"]*(-1),-(vanna_low['calls'] + vanna_low['puts'])/(vanna_low['calls'].abs() + vanna_low['puts'].abs()),(speed['calls'].ffill() + speed['puts'].ffill())/(speed['puts'].ffill().abs() + speed['calls'].ffill().abs()),
                             (vanna_low['calls'] - vanna_low['puts'])/(vanna_low['calls'].abs() + vanna_low['puts'].abs())],axis=1, keys = ['thin_yellow','bold_blue','vanna_low','speed','vanna_low2'])
        hard_sig.loc[(hard_sig['thin_yellow'] < -0.8)  & (hard_sig['bold_blue'] > 0.8) & (hard_sig['vanna_low'] < -0.8) & (hard_sig['speed'] > -0.2) ,'hard_sig'] = 1
        hard_sig.loc[(hard_sig['thin_yellow'] < -0.8)  & (hard_sig['bold_blue'] > 0.8) & (hard_sig['vanna_low'] < -0.8) & (hard_sig['speed'] > -0.2) & (hard_sig['speed'] > hard_sig['vanna_low2']) ,'hard_sig++++'] = 1
        fig.add_trace(go.Bar(x = hard_sig.index, y = hard_sig['hard_sig'],opacity = 1, marker = dict(color='cyan')),secondary_y=True, row=9, col=1)
        fig.add_trace(go.Bar(x = hard_sig.index, y = hard_sig['hard_sig++++'],opacity = 1, marker = dict(color='green')),secondary_y=True, row=9, col=1)
    
        fig.update_layout(height=1900, width=2900,
                         yaxis2=dict(zerolinecolor="black"),
                          yaxis3=dict(zerolinecolor="black"),
                         )
        apply_figure_ranges(fig, stock, start_date)


        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_yaxes(zeroline=True,zerolinewidth=1,automargin='top')
        fig.update_layout({"barmode":"stack"},bargroupgap = 0,bargap=0)

        # fig.show()
        suffix = build_suffix(ultraverse_df, temp_sig_z, speed, vanna_low, hard_sig)
        fig.write_image("F:/illiq/" + suffix + tick + '_' + '.png', engine='kaleido', width=2300, height=1900)

    # except:
    #     pass

tkrs=[]
with open('C://Users//maksut//Dropbox//Market_stats//tikers.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(',')]
        tkrs.append(inner_list)
tkrs = tkrs[0]

# Draw('SMH','2023-01-01')

for g in os.listdir(r"F:/illiq/"):
    os.remove(r"F:/illiq/"+g)


with mp.Pool(15) as pool:
    pool.map(Draw, tkrs, chunksize=1)
    pool.close() 
