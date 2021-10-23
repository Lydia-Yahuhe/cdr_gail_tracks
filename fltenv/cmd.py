from fltsim.aircraft import atccmd

CmdCount = 3
KT2MPS = 0.514444444444444
NM2M = 1852
flight_level = [i*300.0 for i in range(29)]
flight_level += [i*300.0 + 200.0 for i in range(29, 50)]


def calc_level(alt, v_spd, delta):
    delta = int(delta / 300.0)
    lvl = int(alt / 300.0) * 300.0

    if alt < 8700.0:
        idx = flight_level.index(lvl)
        if (v_spd > 0 and alt - lvl != 0) or (v_spd == 0 and alt - lvl > 150):
            idx += 1

        return flight_level[idx+delta]

    lvl += 200.0
    idx = flight_level.index(lvl)
    if v_spd > 0 and alt - lvl > 0:
        idx += 1
    elif v_spd < 0 and alt - lvl < 0:
        idx -= 1

    return flight_level[idx+delta]


def check_cmd(cmd, a, alt_check):
    if not a.is_enroute() or not a.next_leg:
        return False, '1'

    # 最高12000m，最低6000m
    target_alt = cmd.targetAlt
    if target_alt > 12000 or target_alt < 6000:
        return False, '2'

    # 下降的航空器不能上升，或上升的航空器不能下降
    v_spd, delta = a.status.vSpd, cmd.delta
    if v_spd * delta < 0:
        return False, '3'

    # 调过上升，又调下降，或调过下降，又调上升
    if delta == 0.0:
        prefix = int(abs(v_spd) / v_spd) if v_spd != 0.0 else 0
    else:
        prefix = int(abs(delta) / delta)

    if prefix == 0:
        return True, '0'

    if len(alt_check) > 0 and prefix not in alt_check:
        return False, '4'
    alt_check.append(prefix)
    return True, '0'


def int_2_atc_cmd(time: int, idx: int, target):
    if idx < 3:  # [0, 2]  ALT: [-300:300:300]
        delta = (idx - 1) * 300.0
        targetAlt = calc_level(target.status.alt, target.status.vSpd, delta)
        return atccmd.AltCmd(delta=delta, targetAlt=targetAlt, assignTime=time)


def reward_for_cmd(cmd_info):
    cmd = cmd_info['cmd']
    if not cmd_info['ok']:
        reward = -0.5
    else:
        reward = -0.1 if cmd.delta != 0 else 0.0
    return reward
