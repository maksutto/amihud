import yfinance as yf
import pandas as pd
import numpy as np
import math

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



    


def Draw(tick,start_date = '2024-01-01'):    
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import math
        from set_split import set_split
        from scipy.ndimage import uniform_filter1d    
        from filter_outliers import filter_outliers
        def rank_iv(data):
            window = 252
            
            # Расчет минимального и максимального IV за период
            data['min_iv'] = data['vix'].rolling(window=window, min_periods=1).min()
            data['max_iv'] = data['vix'].rolling(window=window, min_periods=1).max()
            # Расчет IV Rank
            data['iv_rank'] = (data['vix'] - data['min_iv']) / (data['max_iv'] - data['min_iv']) * 100
            data['iv_rank'] = data['iv_rank'].clip(0, 100)

            # IV Percentile
            def calculate_iv_percentile(series):
                if len(series) < window:
                    return np.nan
                current_iv = series.iloc[-1]
                historical_iv = series.iloc[:-1]
                count_below = (historical_iv < current_iv).sum()
                return (count_below / len(historical_iv)) * 100
            
            data['iv_percentile'] = data['vix'].ffill().rolling(window=window + 1, min_periods=window + 1).apply(
                calculate_iv_percentile, raw=False
            )
            return data


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
        def converting(df,tick,start_date):
                df = df.loc[df['tiker'] == tick]
                df["date"]=pd.to_datetime(df["date"],format='%d %b %Y')
                df.index = df["date"]
                df = df[~df.index.duplicated()]
                df = df.iloc[df.index > start_date]
                return df
        
        gamma_low = pd.read_csv(r'C:\Users\maksut\Dropbox\Market_stats\gamma_low_orig_range_summ_result_test_all.csv',index_col=0)
        gamma_low = converting(gamma_low,tick,start_date)

        vanna_low = pd.read_csv(r'C:\Users\maksut\Dropbox\Market_stats\vanna_low_orig_range_summ_result_test_all.csv',index_col=0)#
        vanna_low = converting(vanna_low,tick,start_date)
   
        vanna_low2 = pd.read_csv(r'C:\Users\maksut\Dropbox\Market_stats\gamma2_summ_result.csv',index_col=0)#
        vanna_low2 = converting(vanna_low2,tick,start_date)

        gamma2_low = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\gamma_summ_result.csv',index_col=0) #gamma_summ_result.csv,gamma2_summ_result.csv
        gamma2_low = converting(gamma2_low,tick,start_date)
    
        gamma3_low = pd.read_csv(r'C:\Users\maksut\Dropbox\Market_stats\charm_range_summ_result_test.csv',index_col=0)
        gamma3_low = converting(gamma3_low,tick,start_date)
    
        charm = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\charm3_range_summ_result_test.csv',index_col=0)
        charm = converting(charm,tick,start_date)

        delta = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\delta2_range_summ_result_test.csv',index_col=0)
        delta = converting(delta,tick,start_date)      

        gamma_range=pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\gamma2_range_summ_result_test.csv',index_col=0)
        gamma_range = gamma_range.loc[gamma_range['tiker'] == tick]
        gamma_range["date"]=pd.to_datetime(gamma_range["date"],format='%d %b %Y')
        gamma_range.index = gamma_range["date"]
        gamma_range = gamma_range[~gamma_range.index.duplicated()]
        gamma_range['calls'] = gamma_range['calls'].bfill().ffill()
        gamma_range['puts'] = gamma_range['puts'].bfill().ffill()
        gamma_range = gamma_range.iloc[gamma_range.index > '2019-01-01']
      
        color = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\color_range_summ_result_test4.csv',index_col=0)
        color = color.loc[color['tiker'] == tick]
        color["date"]=pd.to_datetime(color["date"],format='%d %b %Y')
        color.index = color["date"]
        color = color[~color.index.duplicated()]           
        color = color.iloc[color.index > start_date]      



        # vntest = pd.read_csv(r'C:\Users\maksut\Dropbox\Market_stats\test_of_test.csv',index_col=0)
        # vntest = vntest.loc[vntest['ticker'] == tick]
        # vntest["date"]=pd.to_datetime(vntest["date"],format='mixed')
        # vntest.index = vntest["date"]


        vix = pd.read_csv(r'C:\Users\maksut\Dropbox\Market_stats\vix_result.csv',index_col=0)
        vix = vix.loc[vix['tiker'] == tick]
        vix["gamma_date"]=pd.to_datetime(vix["gamma_date"],format='%d %b %Y')
        vix.index = vix["gamma_date"]
        # vix = vix.iloc[vix.index > start_date]
        vix = rank_iv(vix)
    
        vanna_summ_ = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\vanna2b_range_summ_result_test.csv',index_col=0)
        vanna_summ_ = converting(vanna_summ_,tick,start_date)   
    
        zomma = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\zomma_range_summ_result_orig.csv',index_col=0)
        zomma = zomma.loc[zomma['tiker'] == tick]
        zomma["date"]=pd.to_datetime(zomma["date"],format='%d %b %Y')
        zomma.index = zomma["date"]
        zomma = zomma[~zomma.index.duplicated()]           
        zomma = filter_outliers(zomma, 'calls', method = 'iqr',mode = 'remove_strong')
        zomma = filter_outliers(zomma, 'puts', method = 'iqr',mode = 'remove_strong')


        zomma = zomma.iloc[zomma.index > start_date]

        vomma = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\vomma_original_range_summ_result_test.csv',index_col=0)
        vomma = vomma.loc[vomma['tiker'] == tick]
        vomma["date"]=pd.to_datetime(vomma["date"],format='%d %b %Y')
        vomma.index = vomma["date"]
        vomma = vomma[~vomma.index.duplicated()]           
        vomma = vomma.iloc[vomma.index > start_date]

        speed60 = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\speed_range_summ_result_test30.csv',index_col=0)
        speed60 = speed60.loc[speed60['tiker'] == tick]
        speed60["date"]=pd.to_datetime(speed60["date"],format='%d %b %Y')
        speed60.index = speed60["date"]
        speed60 = speed60[~speed60.index.duplicated()]
        speed60['puts'] = speed60['puts'].bfill()
        speed60['calls'] = speed60['calls'].bfill()
        speed60['puts'] = uniform_filter1d(speed60['puts'],size=3) 
        speed60['calls'] = uniform_filter1d(speed60['calls'],size=3) 

    
        speed = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\speed_range_summ_result_test.csv',index_col=0)
        speed = speed.loc[speed['tiker'] == tick]
        speed["date"]=pd.to_datetime(speed["date"],format='%d %b %Y')
        speed.index = speed["date"]
        speed = speed[~speed.index.duplicated()]
        speed['puts'] = speed['puts'].bfill()
        speed['calls'] = speed['calls'].bfill()
        speed['puts'] = uniform_filter1d(speed['puts'],size=3) 
        speed['calls'] = uniform_filter1d(speed['calls'],size=3) 
        # speed['puts'] = speed['puts'] . loc[speed['puts'] > speed['puts'].quantile(0.09)*6]
        # speed['puts'] = speed['puts'] . loc[speed['puts'] < speed['puts'].quantile(0.9)*-8 ]
        # speed['calls'] = speed['calls'] . loc[speed['calls'] < speed['calls'].quantile(0.9)*6]
        # speed['calls'] = speed['calls'] . loc[speed['calls'] > speed['calls'].quantile(0.09)*6]    
        speed = speed.iloc[speed.index > start_date]

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

    
    
        ultima = pd.read_csv(r'C:\\Users\\maksut\\Dropbox\\Market_stats\\ultima_range_summ_result_test.csv',index_col=0)#ultima_all  ultima_range_summ_result_test
        ultima = ultima.loc[ultima['tiker'] == tick]
        ultima["date"]=pd.to_datetime(ultima["date"],format='%d %b %Y')
        ultima.index = ultima["date"]
        ultima = ultima[~ultima.index.duplicated()]
        ultima = ultima.iloc[ultima.index > start_date]
 
        
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
        cut = start_date
        fig.update_layout(xaxis = dict(range=[cut, stock.index.tolist()[-1]]),
                      xaxis2 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis3 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis4 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis5 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis6 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis7 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis8 = dict(range=[cut, stock.index.tolist()[-1]]),
                    xaxis9 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis10 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis11 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis12 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis13 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis14 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis15 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis16 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis17 = dict(range=[cut, stock.index.tolist()[-1]]),
                     xaxis_rangeslider_visible=False,
                     yaxis2=dict(zerolinecolor="black"),
                     yaxis3=dict(zerolinecolor="black"),
                     yaxis4=dict(zerolinecolor="black"),
                     yaxis6=dict(zerolinecolor="black"),
                     yaxis5=dict(zerolinecolor="black"),
                     yaxis7=dict(zerolinecolor="black"),
                     yaxis8=dict(zerolinecolor="black"),
                      yaxis9=dict(zerolinecolor="black"),
                      yaxis10=dict(zerolinecolor="black"),
                      yaxis11=dict(zerolinecolor="black"),
                      yaxis12=dict(zerolinecolor="black"),
                      yaxis13=dict(zerolinecolor="black"),
                      yaxis14=dict(zerolinecolor="black"),
                      yaxis15=dict(zerolinecolor="black"),
                      yaxis16=dict(zerolinecolor="black"),
                      yaxis17=dict(zerolinecolor="black"),
                      yaxis18=dict(zerolinecolor="black"),
                      yaxis19=dict(zerolinecolor="black"),
                      yaxis20=dict(zerolinecolor="black"),
                      yaxis21=dict(zerolinecolor="black"),
                      yaxis22=dict(zerolinecolor="black"),
                         )


        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_yaxes(zeroline=True,zerolinewidth=1,automargin='top')
        fig.update_layout({"barmode":"stack"},bargroupgap = 0,bargap=0)

        # fig.show()
        suf = []
        if (ultraverse_df['ultraverse'].iloc[-10:] > 0).any():
            suf.append('_CYAN_')
        if (ultraverse_df['ultraverse8'].iloc[-10:] > 0).any():
            suf.append('_CYAN_')
        if (ultraverse_df['ultraverse_3'].iloc[-10:] > 0).any():
            suf.append('_BLUE_')
        if (ultraverse_df['ultraverse_4'].iloc[-10:] > 0).any():
            suf.append('_PURPLE_')
        if (temp_sig_z['z_sig'].iloc[-10:] > 0).any():
            suf.append('_RED_')
        if (ultraverse_df['ultraverse7'].iloc[-10:] > 0).any():
            suf.append('_CRIMSON_')
        if ((speed['calls'].ffill() > -speed['puts'].ffill()).iloc[-10:] > 0).any():
            suf.append('_SPEED_GREEN_')
        if ((-(vanna_low['calls'] + vanna_low['puts'])/(vanna_low['calls'].abs() + vanna_low['puts'].abs())).iloc[-10:] < -0.8).any():
            suf.append('_LOW_VANNA_')
        if (hard_sig['hard_sig'].iloc[-10:] > 0.8).any():
            suf.append('_hard_sig+++_')
        if (hard_sig['hard_sig++++'].iloc[-10:] > 0.8).any():
            suf.append('_hard_sig++++++++++++++++++_')
        fig.write_image("F:/illiq/" + "".join(suf)  + tick  + '_' + '.png',engine='kaleido',width=2300, height=1900)

    # except:
    #     pass

tkrs=[]
with open('C://Users//maksut//Dropbox//Market_stats//tikers.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(',')]
        tkrs.append(inner_list)
tkrs = tkrs[0]

# Draw('SMH','2023-01-01')

import os
for g in os.listdir(r"F:/illiq/"):
    os.remove(r"F:/illiq/"+g)


import multiprocess as mp
with mp.Pool(15) as pool:
    pool.map(Draw, tkrs, chunksize=1)
    pool.close() 
