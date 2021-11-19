import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns


def kl_divergence(p, q):
    return scipy.stats.entropy(p, q)


def get_kde(x, data_array, bandwidth=0.1):
    def gauss(x):
        import math
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x ** 2))

    N = len(data_array)
    res = 0
    if len(data_array) == 0:
        return 0
    for i in range(len(data_array)):
        res += gauss((x - data_array[i]) / bandwidth)
    res /= (N * bandwidth)
    return res


def get_pdf_or_kde(input_array, bins):
    print(input_array)
    bandwidth = 1.05 * np.std(input_array) * (len(input_array) ** (-1 / 5))
    x_array = np.linspace(0, bins-1, num=100)
    y_array = [get_kde(x_array[i], input_array, bandwidth) for i in range(x_array.shape[0])]
    return y_array


def read_policy_from_npz(path, num_action=2):
    policy = np.load(path)
    acs = policy['acs']

    print(path, acs.shape, np.mean(policy['rews']))

    acs_step = [[] for _ in range(num_action+1)]
    pointer = 0
    for idx in policy['indexes']:
        for i, ac in enumerate(acs[pointer:pointer+idx]):
            acs_step[i].append(int(ac))
        pointer += idx

    acs_step[-1] = acs
    return acs_step


def visual_action_distribution():
    bins = 108
    num_action = 2

    dqn_acs_step = read_policy_from_npz('dqn_policy.npz', num_action=num_action)
    print(dqn_acs_step)
    gail_acs_step = read_policy_from_npz('gail_policy.npz', num_action=num_action)
    print(dqn_acs_step)

    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.rcParams['figure.figsize'] = (100, 100)  # 设定图片大小

    f = plt.figure()  # 确定画布
    for i in range(num_action+1):
        # 可视化动作使用分布
        print('action {}:'.format(i))
        x, y = dqn_acs_step[i], gail_acs_step[i]
        print(x)
        print(y)

        if len(x) == 0 or len(y) == 0:
            continue

        visual(i, x, y, f, bins=bins)

    plt.subplots_adjust(wspace=0.2, hspace=0.5)  # 调整两幅子图的间距
    plt.savefig('distribution_action.svg')
    plt.show()


def visual(i, x, y, f, bins):
    sns.set()  # 设置seaborn默认格式
    np.random.seed(0)  # 设置随机种子数

    # KL散度
    dqn_kde = get_pdf_or_kde(x, bins=bins)
    gail_kde = get_pdf_or_kde(y, bins=bins)
    kl = round(kl_divergence(dqn_kde, gail_kde), 3)
    print(kl)

    f.add_subplot(3, 2, i*2+1)
    plt.hist([x, y], bins=bins, range=(0, bins), density=True, align='left', label=['DQN', 'GAIL'])  # 绘制x的密度直方图

    plt.xlabel('Action index')
    plt.ylabel("Frequency")
    # plt.xticks(np.arange(-1, bins, 1))  # 设置x轴刻度值的字体大小
    plt.yticks(np.arange(0.0, 1.05, 0.1))  # 设置y轴刻度值的字体大小
    plt.title("The histogram of DQN and GAIL policy {}".format(i+1), fontsize=12)  # 设置子图标题
    plt.legend()

    f.add_subplot(3, 2, i*2+2)
    sns.distplot(x, bins=bins, hist=False, label='DQN')  # 绘制x的密度直方图
    sns.distplot(y, bins=bins, hist=False, label='GAIL')  # 绘制y的密度直方图
    plt.xlabel('Action index')
    plt.ylabel("KDE")
    # plt.xticks(np.arange(0, bins, 10))  # 设置x轴刻度值的字体大小
    # plt.yticks(np.arange(0.0, 1.0, 0.1))  # 设置y轴刻度值的字体大小
    plt.title("The similarity of DQN and GAIL policy {}, KL divergence={}".format(i+1, kl), fontsize=12)  # 设置子图标题
    plt.legend()


# 随机数生成
def random_policy_generator():
    data_set = np.load('.\\dqn_policy.npz')

    tmp = {}
    for key, value in data_set.items():
        print(key, value.shape)
        if key == 'acs':
            value = np.random.randint(1, 10, value.shape)
        tmp[key] = value

    np.savez('dqn_policy_1.npz', **tmp)


if __name__ == '__main__':
    # random_policy_generator()
    visual_action_distribution()

