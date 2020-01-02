from math import *
import pandas as pd
import os

a = 6378245.0
ee = 0.00669342162296594323


def converCsv(dir):
    file_list = os.listdir(dir)
    for file_name in file_list:
        file = os.path.join(dir, file_name)
        gcj_pd = pd.read_csv(file)
        wgs_pd = pd.DataFrame(columns=['gcj_lng', 'gcj_lat', 'score'])
        to_file = os.path.join(dir, "result_" + file_name)
        for i in range(len(gcj_pd)):
            pos = gcj_pd.iloc[i, 0]
            wglon, wglat = pos.split('_')
            wglon = float(wglon)
            wglat = float(wglat)
            mglon, mglat = wgs2gcj(wglat, wglon)
            wgs_pd.loc[i] = [mglon, mglat, gcj_pd.iloc[i, 1]]
        wgs_pd.to_csv(to_file, index=False)


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
    converCsv('../results/school')