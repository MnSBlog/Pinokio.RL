import os
import numpy as np
import matplotlib.pyplot as plt


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


def draw_auto_rl_result(path):
    generations = os.listdir(path)
    generations = [folder for folder in generations if 'Gen' in folder]
    info = {'mu': [0.0], 'max': [0.0], 'min': [0.0], 'iteration': [0]}
    for gen in range(1, len(generations) + 1):
        output_path = os.path.join(path, generations[gen-1])
        outputs = os.listdir(output_path)
        outputs = [float(i) for i in outputs]
        min_val = min(outputs)
        max_val = max(outputs)
        mu = np.mean(outputs).item()

        info['mu'].append(mu)
        info['max'].append(max_val)
        info['min'].append(min_val)
        info['iteration'].append(gen)

    plt.plot(info['iteration'], info['max'], '-')
    plt.fill_between(info['iteration'], info['mu'], info['max'], alpha=0.2)
    plt.savefig(os.path.join(path, "progress.jpg"))
    plt.clf()


if __name__ == '__main__':
    draw_auto_rl_result(r'D:\MnS\Projects\RL_Library\figures\AutoRL\CartPole-v1\2022-12-02-14-38-10')

