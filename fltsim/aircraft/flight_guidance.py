from __future__ import annotations

from fltsim.utils import calc_level


def reset_guidance_with_fpl(guidance, fpl):
    guidance.targetAlt = fpl.max_alt
    guidance.targetHSpd = 0
    guidance.targetCourse = 0


# 速度动作被抛弃了
def update_guidance(now, guidance, status, control, profile):
    # 速度引导(均为标称速度)
    v_spd = status.vSpd
    performance = status.performance
    if v_spd == 0.0:
        guidance.targetHSpd = performance.normCruiseTAS
    elif v_spd > 0.0:
        guidance.targetHSpd = performance.normClimbTAS
    else:
        guidance.targetHSpd = performance.normDescentTAS

    # 高度引导
    alt_cmd = control.altCmd
    if alt_cmd is not None and now - alt_cmd.assignTime == 0:
        guidance.targetAlt = alt_cmd.targetAlt
        # delta = alt_cmd.delta
        #
        # # 下降的航空器不能上升，或上升的航空器不能下降
        # is_cmd_ok = False
        # if v_spd * delta >= 0:
        #     target_alt = calc_level(status.alt, v_spd, delta)
        #     alt_cmd.targetAlt = target_alt
        #     if 12000 >= target_alt >= 6000:
        #         guidance.targetAlt = target_alt
        #         is_cmd_ok = True
        control.transition(mode='Alt', ok=True)

    # 航向引导（Dogleg机动）
    hdg_cmd = control.hdgCmd
    if hdg_cmd is None:
        guidance.targetCourse = profile.courseToTarget
        return

    delta, assign_time = hdg_cmd.delta, hdg_cmd.assignTime
    if delta == 0:
        return

    elif now - assign_time == 0:    # 以delta角度出航
        guidance.targetCourse = (delta+status.heading) % 360
    elif now - assign_time == 60:   # 转向后持续60秒飞行，之后以30°角切回航路
        prefix = abs(delta) / delta
        guidance.targetCourse = (-prefix*(abs(delta)+30)+status.heading) % 360
    elif now - assign_time == 120:  # 结束偏置（dogleg机动）
        control.transition(mode='Hdg')
