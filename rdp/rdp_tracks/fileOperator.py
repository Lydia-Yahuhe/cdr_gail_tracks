import os
import random

import xlrd
import pymongo

from utils.computation import clock2time, str2coord, distance, bearing


def get_file_list(data_path, shuffle=False):
    print('>>> Get file dictionary from ' + data_path, end=' ')

    data_path = os.path.abspath(data_path)

    # 是否存在data文件夹
    assert os.path.exists(data_path)

    file_list = []
    for dir_or_file in os.listdir(data_path):
        dir_or_file_ = os.path.join(data_path, dir_or_file)
        if os.path.isfile(dir_or_file_) and dir_or_file_.endswith('.xlsx'):
            file_list.append({'abs_path': dir_or_file_, 'file_name': dir_or_file})

    if shuffle:
        random.shuffle(file_list)
    print(len(file_list))
    return file_list


# 将excel文件里的时间和坐标进行转换
def micro_processing(track):
    # time, track_id, _, _, ground speed, altitude, longitude, latitude
    track[0] = clock2time(track[0])
    if track[-1] > 180:
        track[-1] = str2coord(track[-1])
        track[-2] = str2coord(track[-2])
    return track


# 消除相同点、计算航向
def middle_processing(tracks_dict):
    if len(tracks_dict) <= 0:
        return []

    # 去除时间相同点
    tracks = list(tracks_dict.values())

    new_tracks = []
    last = tracks[0]
    for t in tracks[1:]:
        # 去除坐标相同点
        if distance(last[1:3], t[1:3]) > 0:
            last = t + [bearing(last[1:3], t[1:3])]
            new_tracks.append(last)
    return new_tracks


def get_sheet_1_track(table, limit=600.0):
    """
    Real radar track points included in the sheet1 of excel file
    return [[time, lng, lat, alt, spd, hdg], ]
    """
    tracks_real, check, RFL = {}, False, 0
    for r_t in range(table.nrows):
        # 预处理：坐标str转float、时间戳转int
        track = micro_processing(table.row_values(r_t))

        # 剔除高度在6000m以下的记录点
        if track[5] < limit:
            if not check:
                continue
            else:
                break
        else:
            check = True
            RFL = max(track[-3]*10, RFL)
            tracks_real[track[0]] = [track[0]] + track[-2:] + [track[-3]*10, track[-4]]

    return middle_processing(tracks_real), RFL


def get_sheet_2_route(table):
    """
    Planning trajectory points included in the sheet2 of excel file
    return [[waypoint_id, lng, lat, alt=8100.0]]
    """
    plan_points, checks = [], []
    for r_o in range(1, table.nrows):
        [_, name, lng, lat, *_] = table.row_values(r_o)
        if name in checks:
            name += str(r_o)
        plan_points.append(dict(id=name, lng=float(lng), lat=float(lat), alt=8100.0))
        checks.append(name)
    return plan_points


def get_sheet_3_plan(table):
    """
    1. from where and go where
    2. ac_id, ac_type
    """
    info = table.row_values(1)
    from_to = info[3] + '-' + info[4]
    aircraft = {'id': info[1], 'type': info[2]}
    return aircraft, from_to


def write_execl_into_db(data_path):
    col = pymongo.MongoClient('localhost')['admin']['historicalTracks']
    for i, file in enumerate(get_file_list(data_path)):
        data = xlrd.open_workbook(file['abs_path'])
        assert len(data.sheets()) == 3

        # 读取excel文件的sheet1
        tracks, RFL = get_sheet_1_track(data.sheets()[0])
        # 读取excel文件的sheet2
        plan_route = get_sheet_2_route(data.sheets()[1])
        # 读取excel文件的sheet3
        aircraft, from_to = get_sheet_3_plan(data.sheets()[2])

        flight_id = file['file_name'].split('-')[0]

        print('\t', i, flight_id, aircraft, from_to, RFL, len(tracks), end=' ')
        if len(tracks) <= 50 or flight_id == aircraft['id']:
            print('(abandoned)')
        else:
            print()
            col.insert(dict(id=flight_id, ac=aircraft, RFL=RFL,
                            routing=dict(id=from_to, points=plan_route),
                            tracks=tracks))
        if i >= 1000:
            break


# write_execl_into_db('..\\dataset\\RealTracks')
