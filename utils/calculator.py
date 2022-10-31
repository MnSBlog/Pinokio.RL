import os
import numpy as np


def get_default_kpi(data, interval):
    info = {'mu': [0.0], 'max': [0.0], 'min': [0.0], 'episode': [0]}
    last_index = 0
    count = interval
    while True:
        if len(data) <= last_index:
            break
        chunk = data[last_index:last_index+interval]
        mu = np.asscalar(np.mean(chunk))
        sigma = np.asscalar(np.std(chunk))

        info['mu'].append(mu)
        info['max'].append(mu + (0.5 * sigma))
        info['min'].append(mu - (0.5 * sigma))
        info['episode'].append(count)

        count += interval
        last_index += interval
    return info


def convert_to_numpy(path, name):
    data = []
    f = open(os.path.join(path, name), 'r')
    while True:
        line = f.readline()
        if not line:
            break
        print(line)
        data.append(float(line))
    f.close()
    return np.array(data)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    filenames = ['historyCartPoleRaw-mem_len-4-layer_len-3_epi_reward',
                 'historyCartPoleRaw-mem_len-16-layer_len-3_epi_reward',
                 'historyCartPoleRaw-mem_len-64-layer_len-3_epi_reward']
    for idx, filename in enumerate(filenames):
        values = convert_to_numpy(r"D:\MnS\Projects\RL_Library", filename + '.txt')
        reward_info = get_default_kpi(values, 1000)

        plt.plot(reward_info['episode'], reward_info['mu'], '-')
        plt.fill_between(reward_info['episode'], reward_info['min'], reward_info['max'], alpha=0.2)
        plt.ylim([0, 400])
    plt.savefig(os.path.join(r"D:\MnS\Projects\RL_Library\figures", "manual_graph(4).jpg"))
    plt.clf()
