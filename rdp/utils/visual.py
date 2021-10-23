import simplekml
import random


# 随机透明度的随机颜色
def make_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    c = simplekml.Color.rgb(r, g, b, 100)
    return c


def linestring(kml, tracks, color, name='', description=''):
    ls = kml.newlinestring(name=name, description=description, coords=tracks)
    ls.extrude = 1
    ls.altitudemode = simplekml.AltitudeMode.absolute
    ls.style.linestyle.width = 1
    ls.style.polystyle.color = color
    ls.style.linestyle.color = color


def placemark(kml, tracks, name='', description='', path='.\\plane.png', hdg=None):
    pnt = kml.newpoint(name=name, description=description, coords=[tracks])
    pnt.altitudemode = simplekml.AltitudeMode.absolute
    pnt.style.labelstyle.scale = 0.5
    pnt.style.iconstyle.icon.href = path
    if hdg is not None:
        pnt.style.iconstyle.heading = (hdg+270) % 360


# 输出冲突信息到kml文件中可视化
def conflict_visual(c, pred0, pred1):
    kml = simplekml.Kml()

    # 预测轨迹用purple
    color = simplekml.Color.purple

    # 预测轨迹
    names = c.id.split('-')
    tracks0 = [(p[1], p[2], p[3]*10) for p in pred0.values()]
    linestring(kml, tracks0, color, name=names[0])
    tracks0 = [(p[1], p[2], p[3]*10) for p in pred1.values()]
    linestring(kml, tracks0, color, name=names[1])

    # 两个预测点之间连线
    c_points = c.c_points
    [p0, p1] = c_points[2:]
    tracks0 = [(p[1], p[2], p[3]*10) for p in [p0, p1]]
    linestring(kml, tracks0, simplekml.Color.gray, description=c.conflict_info())

    # 预测点可视化(包括状态位置信息)
    placemark(kml, (p0[1], p0[2], p0[3]*10), name='fake', hdg=p0[-1], description=c.a0_info)
    placemark(kml, (p1[1], p1[2], p1[3]*10), name='fake', hdg=p1[-1], description=c.a1_info)

    # 真实点可视化(包括状态位置信息)
    [p0, p1] = c_points[:2]
    placemark(kml, (p0[1], p0[2], p0[3]*10), name='real', hdg=p0[-1], path='./point.png')
    placemark(kml, (p1[1], p1[2], p1[3]*10), name='real', hdg=p1[-1], path='./point.png')

    # kml名称: c.id
    kml.save(c.id+'.kml')

