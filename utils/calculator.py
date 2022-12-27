import os
import pandas as pd
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


def draw_game_result(path, filename):
    col_title = ['TypeID', 'FirstGun', 'SecondGun', 'HP', 'Kill', 'Step', 'Playtime', 'Result', 'None']
    data = pd.read_csv(path, header=None)
    data.columns = col_title
    window = 10
    results = data['Result'].tolist()

    timeover = []
    lose = []
    win = []
    for idx in range(0, len(results), window):
        batch = results[idx:idx+window]
        timeover.append(batch.count(-1.0))
        lose.append(batch.count(0.0))
        win.append(batch.count(1.0))
        test = 1

    x = list(range(len(timeover)))

    lose = [x + y for x, y in zip(win, lose)]
    timeover = [x + y for x, y in zip(lose, timeover)]
    plt.fill_between(x, 0, win, alpha=0.7, label='Win')
    plt.fill_between(x, win, lose, alpha=0.7, label='Lose')
    plt.fill_between(x, lose, timeover, alpha=0.7, label='TimeOver')

    plt.ylim(0, 11)
    plt.legend(bbox_to_anchor=(0.8, 1.05), loc='center left')
    plt.savefig(filename + 'game_result.png')
    plt.clf()

    results = data['Kill'].tolist()
    kill_min = []
    kill_mean = []
    kill_max = []
    for idx in range(0, len(results), window):
        batch = results[idx:idx+window]
        kill_min.append(min(batch))
        kill_mean.append(sum(batch) / len(batch))
        kill_max.append(max(batch))

    plt.plot(x, kill_max, '-')
    plt.fill_between(x, kill_mean, kill_max, alpha=0.2)
    plt.ylim(0.0, 4.0)
    plt.savefig(filename + 'game_kill.png')
    plt.clf()


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
    root = r'D:\MnS\Projects\RL_Library\history'
    draw_alters = ['zig_zag_gameresult.csv', 'battle_hall_gameresult.csv', 'total_test.csv']
    for alter in draw_alters:
        final_path = os.path.join(root, alter)
        filename = alter.replace('.csv', '')
        draw_game_result(path=final_path, filename=filename)


