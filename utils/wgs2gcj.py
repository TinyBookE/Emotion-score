from math import *
import pandas as pd
import os

a = 6378245.0
ee = 0.00669342162296594323


def converAndConbine(dir):
    file_list = []
    emotion_list = ['beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy']
    for e in emotion_list:
        path = os.path.join(dir, e+'_score.csv')
        assert (os.path.exists(path))
        file_list.append(path)

    result_pd = pd.DataFrame(columns=['filename', 'wgs_lng', 'wgs_lat', 'gcj_lng', 'gcj_lat',
                                      'beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy'])
    file = file_list[0]
    info_pd = pd.read_csv(file, index_col=False)
    filenames = info_pd.filename.values
    wgs_lng = []
    wgs_lat = []
    gcj_lng = []
    gcj_lat = []

    for i in range(len(filenames)):
        wglon, wglat = filenames[i].split('_')
        wglon = float(wglon)
        wglat = float(wglat)
        mglon, mglat = wgs2gcj(wglat, wglon)

        wgs_lng.append(wglon)
        wgs_lat.append(wglat)
        gcj_lng.append(mglon)
        gcj_lat.append(mglat)

    result_pd.iloc[:, 0] = filenames
    result_pd.iloc[:, 1] = wgs_lng
    result_pd.iloc[:, 2] = wgs_lat
    result_pd.iloc[:, 3] = gcj_lng
    result_pd.iloc[:, 4] = gcj_lat

    for i, file_name in enumerate(file_list):
        score_pd = pd.read_csv(file_name)
        result_pd.iloc[:, i+5] = score_pd.score.values

    to_file = os.path.join(dir, 'result.csv')
    result_pd.to_csv(to_file, index=False)

def wgs2gcj(wgLat, wgLon):
    if outOfChina(wgLat, wgLon):
        mgLat = wgLat
        mgLon = wgLon
        return mgLon, mgLat

    dLat = transformLat(wgLon - 105.0, wgLat - 35.0)
    dLon = transformLon(wgLon - 105.0, wgLat - 35.0)
    radLat = wgLat / 180.0 * pi
    magic = sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * cos(radLat) * pi)
    mgLat = wgLat + dLat
    mgLon = wgLon + dLon
    return mgLon, mgLat


def outOfChina(lat, lon):
    if lon < 72.004 or lon > 137.8347:
        return True
    if lat < 0.8293 or lat > 55.8271:
        return True
    return False


def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * sqrt(abs(x))
    ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(y * pi) + 40.0 * sin(y / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * sin(y / 12.0 * pi) + 320 * sin(y * pi / 30.0)) * 2.0 / 3.0
    return ret


def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * sqrt(abs(x))
    ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(x * pi) + 40.0 * sin(x / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * sin(x / 12.0 * pi) + 300.0 * sin(x / 30.0 * pi)) * 2.0 / 3.0
    return ret
if __name__ == "__main__":
    converAndConbine('../results/ex_2_resnet')