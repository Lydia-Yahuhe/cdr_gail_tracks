import numpy as np
import cv2
import pymongo

from fltsim.load import load_data_set
from fltsim.utils import pnpoly, distance

database = pymongo.MongoClient('localhost')['admin']
vertices = [(109.51666666666667, 31.9), (110.86666666666666, 33.53333333333333),
            (114.07, 32.125), (115.81333333333333, 32.90833333333333),
            (115.93333333333334, 30.083333333333332), (114.56666666666666, 29.033333333333335),
            (113.12, 29.383333333333333), (109.4, 29.516666666666666),
            (109.51666666666667, 31.9), (109.51666666666667, 31.9)]
kwargs = dict(border=[108, 118, 28, 35], scale=100)

data_set = load_data_set()
ro_set = data_set.routings


def open_cv_demo():
    # np.set_printoptions(threshold='nan')
    # 创建一个宽512高512的黑色画布，RGB(0,0,0)即黑色
    img = np.zeros((512, 512, 3), np.uint8)

    # 画直线,图片对象，起始坐标(x轴,y轴)，结束坐标，颜色，宽度
    cv2.line(img, (0, 0), (311, 511), (255, 0, 0), 10)
    # # 画矩形，图片对象，左上角坐标，右下角坐标，颜色，宽度
    # cv2.rectangle(img, (30, 166), (130, 266), (0, 255, 0), 3)
    # # 画圆形，图片对象，中心点坐标，半径大小，颜色，宽度
    # cv2.circle(img, (222, 222), 50, (255.111, 111), -1)
    # # 画椭圆形，图片对象，中心点坐标，长短轴，顺时针旋转度数，开始角度(右长轴表0度，上短轴表270度)，颜色，宽度
    # cv2.ellipse(img, (333, 333), (50, 20), 0, 0, 150, (255, 222, 222), -1)
    #
    # # 画多边形，指定各个点坐标,array必须是int32类型
    # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    # # -1表示该纬度靠后面的纬度自动计算出来，实际上是4
    #
    # pts = pts.reshape((-1, 1, 2,))
    # # print(pts)
    # # 画多条线，False表不闭合，True表示闭合，闭合即多边形
    # cv2.polylines(img, [pts], True, (255, 255, 0), 5)
    #
    # # 写字,字体选择
    # font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    #
    # # 图片对象，要写的内容，左边距，字的底部到画布上端的距离，字体，大小，颜色，粗细
    # cv2.putText(img, "OpenCV", (10, 400), font, 3.5, (255, 255, 255), 2)

    a = cv2.imwrite("picture.jpg", img)
    cv2.imshow("picture", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def convert_coord_to_pixel(objects, border=None, scale=None):
    [min_x, max_x, min_y, max_y] = border
    scale_x = (max_x - min_x) * scale
    scale_y = (max_y - min_y) * scale

    tmp = []
    for [x, y] in objects:
        x_idx = int((x - min_x) / (max_x - min_x) * scale_x)
        y_idx = int((max_y - y) / (max_y - min_y) * scale_y)
        tmp.append([x_idx, y_idx])

    return tmp


def search_routing_in_wuhan():
    segments = {}
    check_list = []
    for key, routing in ro_set.items():
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
        print(key, min_idx, max_idx, in_poly_idx, size, len(wpt_list))

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


def generate_wuhan_base_map(save=None, show=False):
    # 创建一个宽512高512的黑色画布，RGB(0,0,0)即黑色
    image = np.zeros((700, 1000, 3), np.uint8)
    points = convert_coord_to_pixel(vertices, **kwargs)
    segments = search_routing_in_wuhan()
    for name, coord in segments.items():
        coord_idx = convert_coord_to_pixel(coord, **kwargs)
        cv2.line(image, coord_idx[0], coord_idx[1], (128, 0, 128), 1)
    pts = np.array(points, np.int32).reshape((-1, 1, 2,))
    cv2.polylines(image, [pts], True, (255, 255, 0), 1)

    if save is not None:
        cv2.imwrite(save, image)

    if show:
        cv2.imshow("wuhan", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return image


def add_point_on_base_map(points, image, save=False):
    for point in points:
        point_idx = convert_coord_to_pixel([point], **kwargs)
        cv2.circle(image, point_idx[0], 2, (0, 255, 0), -1)

    if save:
        cv2.imwrite("script/wuhan.jpg", image)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


if __name__ == '__main__':
    # open_cv_demo()
    save_path = 'script/wuhan.jpg'
    img = generate_wuhan_base_map(save=save_path)

    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    size = (1000, 700)
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)

    # 写字,字体选择
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(100):
        print('No.{} frame'.format(i+1))
        img = cv2.imread(save_path, cv2.IMREAD_COLOR)

        location = [[109.12+0.01*i, 29.00], [111.9, 32.04-0.01*i]]
        dist = distance(location[0], location[1])
        frame = add_point_on_base_map(location, img)

        screen = 'Aircraft a: CCA1234'
        cv2.putText(frame, screen, (800, 100), font, 0.5, (255, 255, 255), 1)
        screen = 'Aircraft b: CSN9876'
        cv2.putText(frame, screen, (800, 120), font, 0.5, (255, 255, 255), 1)
        screen = 'Distance: {} km'.format(round(dist/1000, 2))
        cv2.putText(frame, screen, (800, 140), font, 0.5, (255, 255, 255), 1)

        frame = cv2.resize(frame, size)
        # frame = cv2.flip(frame, 0)
        cv2.imshow('video', frame)
        cv2.waitKey(100)
        out.write(frame)

    out.release()
    cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()
    # # 画直线,图片对象，起始坐标(x轴,y轴)，结束坐标，颜色，宽度
    # cv2.line(img, (100, 70), (311, 511), (255, 0, 0), 10)
    # # 画矩形，图片对象，左上角坐标，右下角坐标，颜色，宽度
    # cv2.rectangle(img, (30, 166), (130, 266), (0, 255, 0), 3)
    # # 画圆形，图片对象，中心点坐标，半径大小，颜色，宽度
    # cv2.circle(img, (222, 222), 50, (255.111, 111), -1)
    # # 画椭圆形，图片对象，中心点坐标，长短轴，顺时针旋转度数，开始角度(右长轴表0度，上短轴表270度)，颜色，宽度
    # cv2.ellipse(img, (333, 333), (50, 20), 0, 0, 150, (255, 222, 222), -1)
    #
    # 画多边形，指定各个点坐标,array必须是int32类型
    # pts = np.array(points, np.int32).reshape((-1, 1, 2,))
    # # 画多条线，False表不闭合，True表示闭合，闭合即多边形
    # cv2.polylines(img, [pts], True, (255, 255, 0), 1)
    #
    # # 写字,字体选择
    # font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    #
    # # 图片对象，要写的内容，左边距，字的底部到画布上端的距离，字体，大小，颜色，粗细
    # cv2.putText(img, "OpenCV", (10, 400), font, 3.5, (255, 255, 255), 2)
    # a = cv2.imwrite("picture.jpg", img)
    # cv2.imshow("picture", img)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
