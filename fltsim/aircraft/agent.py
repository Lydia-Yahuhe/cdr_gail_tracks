from __future__ import annotations

from . import atccmd
from .acft_data import FlightControl, FlightGuidance, FlightStatus, FlightProfile

from .flight_guidance import update_guidance, reset_guidance_with_fpl
from .flight_mechanics import update_status, reset_status_with_fpl
from .flight_profile import update_profile, reset_profile_with_fpl


class AircraftAgent:
    def __init__(self, fpl):
        self.id = fpl.id

        self.control = FlightControl()
        self.guidance = FlightGuidance()
        self.status = FlightStatus(acType=fpl.aircraft.aircraftType)
        self.profile = FlightProfile()

        self.fpl = fpl
        self.tracks = {}

    def is_enroute(self):
        return self.status.is_enroute()

    def is_finished(self):
        return self.status.is_finished()

    def next_leg(self):
        return self.profile.nextLeg

    def next(self, delta=2):
        tmp = []
        for p in self.profile.next(delta):
            p = p.location.toArray()
            tmp += [p[0] / 10, p[1] / 5]
        tmp += [0.0 for _ in range(delta*2)]
        return tmp[:delta*2]

    @property
    def position(self):
        loc = self.status.location
        return [loc.lng, loc.lat, self.status.alt]

    def do_step(self, now, duration=1):
        status = self.status
        profile, control, guidance = self.profile, self.control, self.guidance
        for t in range(duration):
            clock = now + t + 1

            # 到了起飞时间
            if self.fpl.startTime == clock:
                reset_guidance_with_fpl(guidance, self.fpl)
                reset_status_with_fpl(status, self.fpl)
                reset_profile_with_fpl(profile, self.fpl)
                continue

            if self.is_enroute():
                # 在航路上飞行，状态按t转移
                update_guidance(clock, guidance, status, control, profile)
                update_status(status, guidance)
                update_profile(profile, status)

                # 记录轨迹
                self.tracks[clock] = self.position + [status.hSpd, status.course]

    def assign_cmd(self, cmd):
        # 如果指令的执行时间小于目前时间，则报错
        if cmd.cmdType == atccmd.ATCCmdType.Altitude:
            self.control.altCmd = cmd
        elif cmd.cmdType == atccmd.ATCCmdType.Speed:
            self.control.spdCmd = cmd
        elif cmd.cmdType == atccmd.ATCCmdType.Heading:
            self.control.hdgCmd = cmd
        else:
            raise NotImplementedError

    def copy(self):
        other = AircraftAgent(self.fpl)
        other.guidance.set(self.guidance)
        other.control.set(self.control)
        other.status.set(self.status)
        other.profile.set(self.profile)
        return other
