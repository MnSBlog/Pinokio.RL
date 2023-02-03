import copy
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
        chunk = data[last_index:last_index + interval]
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
    data = pd.read_csv(os.path.join(path, filename), header=None)
    data.columns = col_title
    window = 10
    results = data['Result'].tolist()

    timeover = []
    lose = []
    win = []
    for idx in range(0, len(results), window):
        batch = results[idx:idx + window]
        timeover.append(batch.count(-1.0))
        lose.append(batch.count(0.0))
        win.append(batch.count(1.0))
        test = 1

    x = list(range(len(timeover)))

    lose = [x + y for x, y in zip(win, lose)]
    timeover = [x + y for x, y in zip(lose, timeover)]
    plt.fill_between(x, 0, win, alpha=0.7, label='Win')
    plt.fill_between(x, win, lose, alpha=0.7, label='Lose')
    plt.fill_between(x, lose, timeover, alpha=0.7, label='Others')

    plt.ylim(0, 11)
    plt.legend(bbox_to_anchor=(0.8, 1.05), loc='center left')
    plt.savefig(filename + 'game_result.png')
    plt.clf()

    results = data['Kill'].tolist()
    kill_min = []
    kill_mean = []
    kill_max = []
    for idx in range(0, len(results), window):
        batch = results[idx:idx + window]
        kill_min.append(min(batch))
        kill_mean.append(sum(batch) / len(batch))
        kill_max.append(max(batch))

    plt.plot(x, kill_max, '-')
    plt.fill_between(x, kill_mean, kill_max, alpha=0.2)
    plt.ylim(0.0, 2.0)
    plt.savefig(filename + 'game_kill.png')
    plt.clf()


def draw_metric_solver(path):
    draw_auto_rl_result(path)
    generations = os.listdir(path)
    generations = [folder for folder in generations if 'Gen' in folder]
    index = [int(folder.replace('-Gen', '')) for folder in generations if 'Gen' in folder]
    index.sort()
    observation = []
    # 전체의 Min/Max값 찾기 (x - min) / (max - min)
    min_value = np.Inf
    max_value = np.NINF
    for gen in index:
        current_folder = os.path.join(path, str(gen) + '-Gen')
        outputs = os.listdir(current_folder)
        outputs = [float(i) for i in outputs]
        if min(outputs) < min_value:
            min_value = copy.deepcopy(min(outputs))
        if max(outputs) > max_value:
            max_value = copy.deepcopy(max(outputs))

    for gen in index:
        gen_results = []
        current_folder = os.path.join(path, str(gen) + '-Gen')
        outputs = os.listdir(current_folder)
        for output in outputs:
            search = os.path.join(path, str(gen) + '-Gen', output)
            files = os.listdir(search)
            for _ in range(len(files) // 2):
                gen_results.append((float(output) - min_value) / (max_value - min_value))
        if len(gen_results) != 8:
            for _ in range(8 - len(gen_results)):
                gen_results.append(min(gen_results))
        observation.append(gen_results)

    for l in index:
        plt.axvline(l, 0.0, 1.2, color='lightgray', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlim([0, len(index) + 1])
    plt.plot(index, observation, 'ro')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(path, "observed.jpg"))
    plt.clf()


def draw_bayes_result(path, batch=16):
    created_times = []
    output_list = []
    outputs = os.listdir(path)
    outputs = [output for output in outputs if 'json' not in output]
    for output in outputs:
        files = os.listdir(os.path.join(path, output))
        sub_files = [file for file in files if 'metric' in file]
        for sub_file in sub_files:
            full_path = os.path.join(path, output, sub_file)
            created_times.append(os.path.getmtime(full_path))
            output_list.append(float(output) / 300)

    created_times = np.array(created_times)
    output_list = np.array(output_list)
    time_index = np.argsort(created_times)

    observation = []
    sub_obs = []
    for index in time_index:
        sub_obs.append(output_list[index])
        if len(sub_obs) == batch:
            observation.append(copy.deepcopy(sub_obs))
            sub_obs = []

    index = list(range(1, len(observation) + 1))
    for l in index:
        plt.axvline(l, 0, 500, color='lightgray', linestyle='--')
    plt.ylim([0, 300])
    plt.xlim([0, len(index) + 1])
    plt.plot(index, observation, 'ro')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(path, "bayes_observed.jpg"))
    plt.clf()


def draw_auto_rl_result(path):
    generations = os.listdir(path)
    generations = [folder for folder in generations if 'Gen' in folder]
    index = [int(folder.replace('-Gen', '')) for folder in generations if 'Gen' in folder]
    index.sort()
    info = {'mu': [0.0], 'max': [0.0], 'min': [0.0], 'iteration': [0]}
    for gen in index:
        output_path = os.path.join(path, str(gen) + '-Gen')
        outputs = os.listdir(output_path)
        outputs = [float(i) for i in outputs]
        min_val = min(outputs) / 300
        max_val = max(outputs) / 300
        mu = np.mean(outputs).item() / 300

        info['mu'].append(mu)
        info['max'].append(max_val)
        info['min'].append(min_val)
        info['iteration'].append(gen)

    plt.plot(info['iteration'], info['max'], '-')
    plt.fill_between(info['iteration'], info['mu'], info['max'], alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(path, "progress.jpg"))
    plt.clf()


if __name__ == '__main__':
    root = r'D:\MnS\Projects\RL_Library'
    draw_metric_solver(path=r'D:\MnS\Projects\RL_Library\figures\AutoRL\FrozenLake-v1\2023-01-30-21-54-33')
