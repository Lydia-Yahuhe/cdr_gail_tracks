import numpy as np
import time
from contextlib import contextmanager

from baselines.common import colorize

from fltenv.agent_Set import AircraftAgentSet
from fltenv.cmd import CmdCount, int_2_atc_cmd, check_cmd

from fltsim.utils import make_bbox, mid_position, build_rt_index


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

    # def get_states(self):
    #     state = [[0.0 for _ in range(7)] for _ in range(30)]
    #
    #     agents = self.agentSet.agents
    #     [a0, a1] = self.conflict_ac
    #     mid_pos = mid_position(agents[a0].position, agents[a1].position)
    #     bbox = make_bbox(mid_pos, ext=(1.0, 1.0, 900))
    #
    #     ac_en = self.agentSet.agent_en
    #     idx = build_rt_index(ac_en)
    #
    #     j = 0
    #     for i in idx.intersection(bbox):
    #         agent = ac_en[i]
    #         pos = agent.position
    #         ele = [int(agent.id in self.conflict_ac),
    #                pos[0] - self.conflict_pos[0],
    #                pos[1] - self.conflict_pos[1],
    #                (pos[2] - self.conflict_pos[2]) / 3000,
    #                (agent.status.hSpd - 150) / 100,
    #                agent.status.vSpd / 20,
    #                agent.status.heading / 180]
    #         j = min(30-1, j)
    #         state[j] = ele
    #         j += 1
    #     return np.concatenate(state)

    def get_states(self):
        state = [[0.0 for _ in range(7)] for _ in range(50)]

        j = 0
        for agent in self.agentSet.agent_en:
            pos = agent.position
            ele = [int(agent.id in self.conflict_ac),
                   pos[0] - self.conflict_pos[0],
                   pos[1] - self.conflict_pos[1],
                   (pos[2] - self.conflict_pos[2]) / 3000,
                   (agent.status.hSpd - 150) / 100,
                   agent.status.vSpd / 20,
                   agent.status.heading / 180]
            j = min(30-1, j)
            state[j] = ele
            j += 1

        return np.concatenate(state)

    def assign_cmd_idx(self, action):
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

