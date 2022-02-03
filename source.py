import numpy as np
import pandas as pd
import station_data as sd
from scipy.signal import find_peaks
import bisect

def to_time_series(data, binwidth):
    data = data // binwidth
    min_value = int(np.floor(14400 / binwidth))
    max_value = int(np.ceil(104400 / binwidth))
    counts = data.value_counts().reindex([i for i in range(min_value,max_value + 1)], fill_value=0)
    return pd.Series(data=counts.tolist(), index=counts.index * binwidth)

def get_peaks(ts, mode=None):
    moving_average = ts.rolling(180, min_periods=1, center=True).mean()
    moving_std = ts.rolling(180, min_periods=1, center=True).std()
    if mode is None:
        peaks, peak_data = find_peaks(ts)
    elif mode == 'a':
        peaks, peak_data = find_peaks(ts,height=(np.array(moving_average),None))
    elif mode == 'b':
        peaks, peak_data = find_peaks(ts,height=(np.array(moving_average + moving_std),None))
    return ts.index[peaks], peak_data

def get_peaks_of_station_list(df, peak_function, binwidth=1, merge_multiple_exits=True):
    result = {}
    target = 'ESTACAO_S'
    if merge_multiple_exits == True:
        target = 'ESTACAO_S_m'
    df = df[['TOTAL_SECONDS_S', target]]
    groups = df.groupby(target)
    for name, group in groups:
        ts = to_time_series(group['TOTAL_SECONDS_S'], binwidth=binwidth)
        peaks, peak_data = peak_function(ts, mode="b")
        result[name] = peaks
    return result

def generate_mask(stations, duration_data):
    mask = [0]
    for i in range(1,len(stations)):
        dest = stations[i]
        source = stations[i - 1]
        duration = duration_data.query('p_ESTACAO == @source and ESTACAO == @dest')['duration_median']
        mask.append(duration.values[0] + mask[i - 1])
    mask = np.array(mask)
    return mask

def shift_peaks(data, stations, mask):
    res = {}
    mask = np.array(mask)
    for idx, n in enumerate(mask):
        target = data[stations[idx]]
        shifted = target - n
        shifted = [i for i in shifted if i >= 0]
        res[stations[idx]] = shifted
    return res

def get_closest_right_side(n, peaks, limit=float('inf')):
    if n <= peaks[0]:
        if limit <= peaks[0] - n:
            return n + limit
        else:
            return peaks[0]
    elif n > peaks[-1]:
        return n + limit
    lo, hi = 0, len(peaks)
    while lo <= hi:
        mid = (lo + hi) // 2
        if n <= peaks[mid] and n > peaks[mid - 1]:
            if limit < peaks[mid] - n:
                return n + limit
            else:
                return peaks[mid]
        elif n > peaks[mid]:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def get_closest_for_each_station(data, time, get_closest_function, limit=float('inf'), stations=None):
    res = []
    if stations is None:
        stations = list(data.keys())
    for station in stations:
        res.append(get_closest_function(time, data[station], limit=limit))
    return res

def get_score_weighted_squared_error_across_time(data, get_closest_function, limit=float('inf'), start=14400, end=104400, increment=60, stations=None, **kwargs):
    res = []
    indices = []
    closest = {}
    weights = {}
    pairs = kwargs['pairs']
    window_left = kwargs['window_left']
    window_right = kwargs['window_right']
    if stations is None:
        stations = list(data.keys())
    
    station_index = {}
    for idx, station in enumerate(stations):
        station_index[station] = idx
    
    pairs = pairs.query('ESTACAO_E_m in @stations and ESTACAO_S_m in @stations').copy(deep=True)
    pairs['INDEX_E'] = pairs['ESTACAO_E_m']
    pairs['INDEX_S'] = pairs['ESTACAO_S_m']
    pairs = pairs.replace({'INDEX_E': station_index, 'INDEX_S': station_index})
    pairs = pairs.query('INDEX_S > INDEX_E')
    
    for i in range(start, end, increment):
        total = 0
            
        target = pairs.query('TOTAL_SECONDS_S >= @i - @window_left and TOTAL_SECONDS_S <= @i + @window_right')
        counts = target['ESTACAO_S_m'].value_counts().reindex(stations, fill_value=0)
        #if np.sum(counts) == 0:
        #    counts = counts + 1
        counts = counts + 1
        counts = counts / np.sum(counts)
        for station in stations:
            total += ((i - get_closest_function(i, data[station], limit=limit)) ** 2) * counts[station]
        res.append(total)
        indices.append(i)
        closest[i] = get_closest_for_each_station(data, i, get_closest_function, limit, stations)
        weights[i] = counts
        
    res = 1 - (np.array(res) / (limit ** 2))
    return (pd.DataFrame({'score': res}, index=indices), closest, weights)

def get_troughs(ts, interval, mode=None):
    #ts = ts * -1
    moving_average = ts.rolling(interval, min_periods=1, center=True).mean()
    moving_std = ts.rolling(interval, min_periods=1, center=True).std()
    if mode is None:
        peaks, peak_data = find_peaks(ts)
    elif mode == 'a':
        peaks, peak_data = find_peaks(ts,height=(np.array(moving_average),None))
    elif mode == 'b':
        peaks, peak_data = find_peaks(ts,height=(np.array(moving_average + moving_std * .5),None))
    return ts.index[peaks], peak_data

def get_train_arrivals(line_name, reverse, station, troughs, masks):
    stations = [sd.s_get_stations(sd.line_azul, 'b', reverse=False),
                sd.s_get_stations(sd.line_amarela, 'b', reverse=False),
                sd.s_get_stations(sd.line_verde, 'b', reverse=False),
                sd.s_get_stations(sd.line_vermelha, 'b', reverse=False),
                sd.s_get_stations(sd.line_azul, 'b', reverse=True),
                sd.s_get_stations(sd.line_amarela, 'b', reverse=True),
                sd.s_get_stations(sd.line_verde, 'b', reverse=True),
                sd.s_get_stations(sd.line_vermelha, 'b', reverse=True)]

    lines = ['azul', 'amarela', 'verde', 'vermelha']
    idx = lines.index(line_name)
    if reverse: idx = idx + 4

    return np.array(troughs[idx] + masks[idx][stations[idx].index(station)])

def get_expected_exit_times(entrance, path, masks, troughs, max_lag=0):
    lines = ['azul', 'amarela', 'verde', 'vermelha']
    stations = [sd.s_get_stations(sd.line_azul, 'b', reverse=False),
                sd.s_get_stations(sd.line_amarela, 'b', reverse=False),
                sd.s_get_stations(sd.line_verde, 'b', reverse=False),
                sd.s_get_stations(sd.line_vermelha, 'b', reverse=False),
                sd.s_get_stations(sd.line_azul, 'b', reverse=True),
                sd.s_get_stations(sd.line_amarela, 'b', reverse=True),
                sd.s_get_stations(sd.line_verde, 'b', reverse=True),
                sd.s_get_stations(sd.line_vermelha, 'b', reverse=True)]
    current_time = np.array([entrance])
    current_time_lag = np.array([0])
    first = True
    for p in path:
        next_time = []
        next_time_lag = []
        
        arrivals = get_train_arrivals(p[0], p[1], p[2], masks=masks, troughs=troughs)
        idx = lines.index(p[0])
        if p[1]: idx = idx + 4
        mask = masks[idx]
        station = stations[idx]
        src_mask = mask[station.index(p[2])]
        dest_mask = mask[station.index(p[3])]
        duration = dest_mask - src_mask
        for c, lag in zip(current_time, current_time_lag):
            if c == -1:
                i = len(arrivals)
            else:
                i = bisect.bisect(arrivals, c)
            for l in range(max_lag + 1):
                if (i + l >= len(arrivals)):
                    next_time.append(-1)
                    next_time_lag.append(-1)
                else:
                    next_time.append(arrivals[i + l] + duration)
                    next_time_lag.append(lag + l)
        
        current_time = np.array(next_time)
        current_time_lag = np.array(next_time_lag)
        first = False
    return current_time, current_time_lag


def main():
    duration_data = pd.read_csv('duration_data.csv', index_col=0)
    mask_azul = generate_mask(sd.s_get_stations(sd.line_azul, 'b'), duration_data)
    mask_amarela = generate_mask(sd.s_get_stations(sd.line_amarela, 'b'), duration_data)
    mask_verde = generate_mask(sd.s_get_stations(sd.line_verde, 'b'), duration_data)
    mask_vermelha = generate_mask(sd.s_get_stations(sd.line_vermelha, 'b'), duration_data)
    mask_azul_reverse = generate_mask(sd.s_get_stations(sd.line_azul, 'b', reverse=True), duration_data)
    mask_amarela_reverse = generate_mask(sd.s_get_stations(sd.line_amarela, 'b', reverse=True), duration_data)
    mask_verde_reverse = generate_mask(sd.s_get_stations(sd.line_verde, 'b', reverse=True), duration_data)
    mask_vermelha_reverse = generate_mask(sd.s_get_stations(sd.line_vermelha, 'b', reverse=True), duration_data)
    masks = [mask_azul, mask_amarela, mask_verde, mask_vermelha,
             mask_azul_reverse, mask_amarela_reverse, mask_verde_reverse, mask_vermelha_reverse]


    df = pd.read_csv('sample_synthetic_data.csv', index_col=0)

    station_peaks = get_peaks_of_station_list(df, peak_function=get_peaks, binwidth=15)

    shifted_azul = shift_peaks(station_peaks, stations=sd.s_get_stations(sd.line_azul, 'b'), mask=mask_azul)
    shifted_amarela = shift_peaks(station_peaks, stations=sd.s_get_stations(sd.line_amarela, 'b'), mask=mask_amarela)
    shifted_verde = shift_peaks(station_peaks, stations=sd.s_get_stations(sd.line_verde, 'b'), mask=mask_verde)
    shifted_vermelha = shift_peaks(station_peaks, stations=sd.s_get_stations(sd.line_vermelha, 'b'), mask=mask_vermelha)
    shifted_azul_reverse = shift_peaks(station_peaks, stations=sd.s_get_stations(sd.line_azul, 'b', reverse=True), mask=mask_azul_reverse)
    shifted_amarela_reverse = shift_peaks(station_peaks, stations=sd.s_get_stations(sd.line_amarela, 'b', reverse=True), mask=mask_amarela_reverse)
    shifted_verde_reverse = shift_peaks(station_peaks, stations=sd.s_get_stations(sd.line_verde, 'b', reverse=True), mask=mask_verde_reverse)
    shifted_vermelha_reverse = shift_peaks(station_peaks, stations=sd.s_get_stations(sd.line_vermelha, 'b', reverse=True), mask=mask_vermelha_reverse)

    score_function = get_score_weighted_squared_error_across_time
    get_closest_function=get_closest_right_side
    limit=180 # alpha
    interval=90
    increment = 60

    print('getting likelihood scores for each line...')
    scores_azul = score_function(data=shifted_azul, get_closest_function=get_closest_function, limit=limit,
        pairs=df, window_left=limit, window_right=limit, increment=increment)
    scores_amarela = score_function(data=shifted_amarela, get_closest_function=get_closest_function, limit=limit,
        pairs=df, window_left=limit, window_right=limit, increment=increment)
    scores_verde = score_function(data=shifted_verde, get_closest_function=get_closest_function, limit=limit,
        pairs=df, window_left=limit, window_right=limit, increment=increment)
    scores_vermelha = score_function(data=shifted_vermelha, get_closest_function=get_closest_function, limit=limit,
        pairs=df, window_left=limit, window_right=limit, increment=increment)
    scores_azul_reverse = score_function(data=shifted_azul_reverse, get_closest_function=get_closest_function, limit=limit,
        pairs=df, window_left=limit, window_right=limit, increment=increment)
    scores_amarela_reverse = score_function(data=shifted_amarela_reverse, get_closest_function=get_closest_function, limit=limit,
        pairs=df, window_left=limit, window_right=limit, increment=increment)
    scores_verde_reverse = score_function(data=shifted_verde_reverse, get_closest_function=get_closest_function, limit=limit,
        pairs=df, window_left=limit, window_right=limit, increment=increment)
    scores_vermelha_reverse = score_function(data=shifted_vermelha_reverse, get_closest_function=get_closest_function, limit=limit,
        pairs=df, window_left=limit, window_right=limit, increment=increment)

    print('estimating train spawn times for each line...')
    mode='b'
    troughs_azul = get_troughs(scores_azul[0]['score'], interval=interval, mode=mode)[0]
    troughs_amarela = get_troughs(scores_amarela[0]['score'], interval=interval, mode=mode)[0]
    troughs_verde = get_troughs(scores_verde[0]['score'], interval=interval, mode=mode)[0]
    troughs_vermelha = get_troughs(scores_vermelha[0]['score'], interval=interval, mode=mode)[0]
    troughs_azul_reverse = get_troughs(scores_azul_reverse[0]['score'], interval=interval, mode=mode)[0]
    troughs_amarela_reverse = get_troughs(scores_amarela_reverse[0]['score'], interval=interval, mode=mode)[0]
    troughs_verde_reverse = get_troughs(scores_verde_reverse[0]['score'], interval=interval, mode=mode)[0]
    troughs_vermelha_reverse = get_troughs(scores_vermelha_reverse[0]['score'], interval=interval, mode=mode)[0]
    troughs = [troughs_azul, troughs_amarela, troughs_verde, troughs_vermelha,
              troughs_azul_reverse, troughs_amarela_reverse, troughs_verde_reverse, troughs_vermelha_reverse] 

    # Defining two example paths from AM to CG
    path_1 = [
        ('verde', True, 'AM', 'CG')
    ]

    path_2 = [
        ('vermelha', True, 'AM', 'SA'),
        ('amarela', True, 'SA', 'CG')
    ]

    print('Getting train arrivals for a given line...')
    print(get_train_arrivals('verde', True, 'CG', masks=masks, troughs=troughs)[:10])

    print('Predicting exit times given entrance time and path...')
    print('Path 1')
    print(get_expected_exit_times(46755, path_1, max_lag=1, masks=masks, troughs=troughs))
    print('Path 2')
    print(get_expected_exit_times(46755, path_2, max_lag=1, masks=masks, troughs=troughs))

if __name__ == "__main__":
    main()