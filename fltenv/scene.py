import numpy as np
import time
import cv2

from contextlib import contextmanager

from baselines.common import colorize

from fltenv.agent_Set import AircraftAgentSet
from fltenv.cmd import CmdCount, int_2_atc_cmd, check_cmd

from fltsim.visual import add_points_on_base_map


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'), end=' ')
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))


class ConflictScene:
    def __init__(self, info, limit=0):
        self.conflict_ac, self.clock = info['conflict_ac'], info['time']
        fpl_list = info['fpl_list']

        self.agentSet = AircraftAgentSet(fpl_list=fpl_list, start=info['start'])
        self.agentSet.do_step(self.clock - 300 + limit, basic=True)
        self.conflict_pos = info['other'][0]

        # print('\nNew scenario--------------------------------')
        # print(' Conflict Info: ', self.conflict_ac, self.clock, self.agentSet.time, len(fpl_list), info['other'])

        self.cmd_list = {}
        for c_ac in self.conflict_ac:
            self.cmd_list[c_ac] = []

    def now(self):
        return self.agentSet.time

    def get_states(self, width, height, channel):
        kwargs = dict(border=[108, 118, 28, 35], scale=200)

        # 轨迹点
        points = []
        for key, agent in self.agentSet.agents.items():
            points.append(agent.get_x_data())

        # 武汉扇区的底图（有航路）
        base_img = cv2.imread('dataset/wuhan_base.jpg', cv2.IMREAD_COLOR)

        # 将当前时刻所有航空器的位置和状态放在图上
        frame = add_points_on_base_map(points, base_img, **kwargs)
        frame = cv2.resize(frame, (width, height))
        # print(frame.shape)
        # cv2.imshow('image', frame)
        # cv2.waitKey(0)
        return frame

    def assign_cmd_idx(self, action):
        action = np.argmax(action)
        agent, idx = self.conflict_ac[action // CmdCount], action % CmdCount
        check = self.cmd_list[agent]

        agent = self.agentSet.agents[agent]
        cmd = int_2_atc_cmd(self.now()+1, idx, agent)
        ok, reason = check_cmd(cmd, agent, check)
        # print(idx, ok, reason, cmd, agent.altitude, end='\t')
        print(action, end=' ')

        if ok:
            agent.assign_cmd(cmd)
        return {'cmd': cmd, 'ok': ok}

    def do_step(self):
        has_conflict = self.__do_real(self.now() + 120)
        return has_conflict, self.now() - self.clock

    def __do_real(self, end_time):
        while self.now() < end_time:
            self.agentSet.do_step(duration=30)
            conflicts = self.agentSet.detect_conflict_list(search=self.conflict_ac)
            if len(conflicts) > 0:
                return True
        return False

