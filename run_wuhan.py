import numpy as np
import matplotlib.pyplot as plt

from rdp.file_processor import get_fpl_list
from rdp.model import AgentSetReal

from fltsim.visual import cv2, add_points_on_base_map, add_lines_on_base_map


def main(picture_size=(670, 450), wait_time=1):
    # 从excel中提取轨迹和航班信息
    fpl_list, starts = get_fpl_list(alt_limit=6000.0, number=1000)

    # 构建agent set类
    agent_set = AgentSetReal(fpl_list, starts)

    kwargs = dict(border=[109.3, 116, 29, 33.5], scale=100)

    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('gail_video.avi', fourcc, 20.0, picture_size)

    # agentSet运行
    while not agent_set.all_done:
        base_img = cv2.imread('.\\dataset\\wuhan_base.jpg', cv2.IMREAD_COLOR)

        states = agent_set.do_step()
        # line_two = agent_set.detection_conflicts()

        if len(states) > 10:
            # base_img = add_lines_on_base_map(line_two, base_img, **kwargs)
            locations = [[key] + state for key, state in states.items()]
            frame, _ = add_points_on_base_map(locations, base_img, **kwargs)
            frame = cv2.resize(frame, picture_size)
            # cv2.imshow('video', frame)
            # cv2.waitKey(wait_time)
            out.write(frame)

    out.release()
    cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()
    print('\n>>> The process of running agent set is finished!')

    # 可视化历史轨迹和计划轨迹
    agent_set.visual(save_path='AgentSet')
    # 可视化流量
    agent_set.flow_visual()
    print('\n>>> Agent set is visualized!')

    return
    print('\n>>> Distance curves are setting to figure:')
    check_list = []
    for a0_id, sur_states in agent_set.around_agents.items():
        print(a0_id, sur_states)
        for a1_id, sur_state in sur_states.items():
            if a0_id == a1_id or a0_id + '-' + a1_id in check_list:
                continue

            agent_states = np.array(sur_state)
            x = list(agent_states[:, 0])
            y1 = list(agent_states[:, 1])
            y2 = list(agent_states[:, 2])

            print('\t', a1_id, agent_states.shape, len(x), len(y1), len(y2))
            if len(x) <= 64 or min(y1) >= 50.0:
                continue

            check_list.append(a0_id + '-' + a1_id)
            check_list.append(a1_id + '-' + a0_id)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(x, y1, label='Horizontal')
            ax1.set_ylim([0, 80.0])
            ax1.set_ylabel('Horizontal Distance/km')
            ax1.set_title('The distance changes between {} and {}'.format(a0_id, a1_id))
            ax1.set_xlabel('Time Line/s')

            ax2 = ax1.twinx()  # this is the important function
            ax2.plot(x, y2, 'r', label='Vertical')
            ax2.set_ylim([0, 900.0])
            ax2.set_ylabel('Vertical Distance/m')

            fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

            plt.savefig('dataset/pictures/' + a0_id + '-' + a1_id)
            # plt.show()


if __name__ == '__main__':
    main()
