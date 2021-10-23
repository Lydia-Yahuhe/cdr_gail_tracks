import simplekml
import random
import numpy as np
import cv2

from fltsim.load import routings
from fltsim.utils import pnpoly, convert_coord_to_pixel, distance

alt_mode = simplekml.AltitudeMode.absolute


def make_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    c = simplekml.Color.rgb(r, g, b, 100)
    return c


def tuple2kml(kml, name, tracks, color=simplekml.Color.chocolate):
    ls = kml.newlinestring(name=name)
    ls.coords = [(wpt[0], wpt[1], wpt[2]) for wpt in tracks]
    ls.extrude = 1
    ls.altitudemode = alt_mode
    ls.style.linestyle.width = 1
    ls.style.polystyle.color = color
    ls.style.linestyle.color = color


def place_mark(point, kml, name='test', hdg=None, description=None):
    pnt = kml.newpoint(name=name, coords=[point],
                       altitudemode=alt_mode, description=description)

    pnt.style.labelstyle.scale = 0.25
    pnt.style.iconstyle.icon.href = '.\\placemark.png'
    # pnt.style.iconstyle.icon.href = '.\\plane.png'
    if hdg is not None:
        pnt.style.iconstyle.heading = (hdg + 270) % 360


def save_to_kml(tracks, plan, save_path='agent_set'):
    kml = simplekml.Kml()

    folder = kml.newfolder(name='real')
    for key, t in tracks.items():
        tuple2kml(folder, key, t, color=simplekml.Color.chocolate)

    folder = kml.newfolder(name='plan')
    for key, t in plan.items():
        tuple2kml(folder, key, t, color=simplekml.Color.royalblue)

    print("Save to "+save_path+".kml successfully!")
    kml.save(save_path+'.kml')


# ---------
# opencv
# ---------
def search_routing_in_a_area(vertices):
    segments = {}
    check_list = []
    for key, routing in routings.items():
        wpt_list = routing.waypointList

        in_poly_idx = []
        for i, wpt in enumerate(wpt_list):
            loc = wpt.location
            in_poly = pnpoly(vertices, [loc.lng, loc.lat])
            if in_poly:
                in_poly_idx.append(i)

        if len(in_poly_idx) <= 0:
            continue

        size = i + 1
        min_idx, max_idx = max(min(in_poly_idx) - 1, 0), min(size, max(in_poly_idx) + 2)
        # print(key, min_idx, max_idx, in_poly_idx, size, len(wpt_list))

        new_wpt_list = wpt_list[min_idx:max_idx]
        assert len(new_wpt_list) >= 2
        for i, wpt in enumerate(new_wpt_list[1:]):
            last_wpt = new_wpt_list[i]
            name_f, name_l = last_wpt.id + '-' + wpt.id, wpt.id + '-' + last_wpt.id

            if name_f not in check_list:
                segments[name_f] = [[last_wpt.location.lng, last_wpt.location.lat],
                                    [wpt.location.lng, wpt.location.lat]]
                check_list += [name_l, name_f]

    return segments


def generate_wuhan_base_map(size=(700, 1000, 3), save=None, show=False, **kwargs):
    # 武汉空域
    vertices = [(109.51666666666667, 31.9), (110.86666666666666, 33.53333333333333),
                (114.07, 32.125), (115.81333333333333, 32.90833333333333),
                (115.93333333333334, 30.083333333333332), (114.56666666666666, 29.033333333333335),
                (113.12, 29.383333333333333), (109.4, 29.516666666666666),
                (109.51666666666667, 31.9), (109.51666666666667, 31.9)]

    # 创建一个宽512高512的黑色画布，RGB(0,0,0)即黑色
    image = np.zeros(size, np.uint8)

    points = convert_coord_to_pixel(vertices, **kwargs)

    segments = search_routing_in_a_area(vertices)
    for name, coord in segments.items():
        coord_idx = convert_coord_to_pixel(coord, **kwargs)
        cv2.line(image, coord_idx[0], coord_idx[1], (128, 0, 128), 1)
    pts = np.array(points, np.int32).reshape((-1, 1, 2,))
    cv2.polylines(image, [pts], True, (255, 255, 0), 1)

    if save is not None:
        cv2.imwrite(save, image)

    if show:
        cv2.imshow("wuhan", image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    return image


def add_point_on_base_map(points, image, save=False, speed=1, font_scale=0.4, font=cv2.FONT_HERSHEY_SIMPLEX, **kwargs):
    color = (255, 255, 255)
    now, ac_en = points[0][1], len(points)
    info = 'Time: {}, ac_en: {}, speed: x{}'.format(now, ac_en, speed)
    cv2.putText(image, info, (700, 50), font, font_scale, color, 1)

    for [name, _, lng, lat, *point] in points:
        [x, y] = convert_coord_to_pixel([[lng, lat]], **kwargs)[0]
        cv2.circle(image, [x, y], 2, (0, 255, 0), -1)

        cv2.putText(image, name, (x, y+10), font, font_scale, color, 1)
        state = 'Altitude: {}'.format(point[0])
        cv2.putText(image, state, (x, y+30), font, font_scale, color, 1)
        state = '   Speed: {}'.format(point[1])
        cv2.putText(image, state, (x, y+50), font, font_scale, color, 1)
        state = ' Heading: {}'.format(point[2])
        cv2.putText(image, state, (x, y+70), font, font_scale, color, 1)

    if save:
        cv2.imwrite("script/wuhan.jpg", image)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


def add_line_on_base_map(lines, image, save=False, font_scale=0.4, font=cv2.FONT_HERSHEY_SIMPLEX, **kwargs):
    if len(lines) <= 0:
        return image

    color = (255, 0, 0)
    for [pos0, pos1, h_dist, v_dist] in lines:
        [start, end] = convert_coord_to_pixel([pos0, pos1], **kwargs)
        cv2.line(image, start, end, color, 1)

        mid_idx = (int((start[0]+end[0])/2)+10, int((start[1]+end[1])/2)+10)
        state = ' H_dist: {}, V_dist'.format(h_dist, v_dist)
        cv2.putText(image, state, mid_idx, font, font_scale, color, 1)

    if save:
        cv2.imwrite("script/wuhan.jpg", image)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image