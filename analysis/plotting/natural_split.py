import csv
import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    formats = ['png', 'pdf', 'svg', 'eps']

    gmetric = 'groc'
    lmetric = 'lroc'
    metric = 'AUC'

    title = 'BCTCGA | Weighted and Unweighted Models'
    file = '../../results/evaluation/tcga_100_each.csv'

    stats = {}
    # xs = []
    xs = ['1,0', '19,1', '19,0']

    with open(file, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=';')
        headers = next(data)
        gauc_idx = headers.index(gmetric)
        lauc_idx = headers.index(lmetric)

        for row in data:
            x_label = f'{row[1]},{row[2]}'
            stat = stats.get(x_label)
            if not stat:
                stat = {
                    gmetric: [],
                    lmetric: [],
                }
                stats[x_label] = stat

            gval = json.loads(row[gauc_idx])
            stat[gmetric].append(gval)

            lvals = json.loads(row[lauc_idx])

            if len(lvals) > 0:
                stat[lmetric].extend(lvals)
            else:
                stat[lmetric].append(gval)

    datainfo = str(len(stats['19,1'][f'{gmetric}'])) + ' / ' + str(len(stats['19,0'][f'{gmetric}']))
    title += ' | ' + datainfo

    y_gauc_median = [np.median(stats[x][gmetric]) for x in xs]
    y_gauc_q25 = [np.quantile(stats[x][gmetric], 0.25) for x in xs]
    y_gauc_q75 = [np.quantile(stats[x][gmetric], 0.75) for x in xs]

    y_lauc_median = [np.median(stats[x][lmetric]) for x in xs]
    y_lauc_q25 = [np.quantile(stats[x][lmetric], 0.25) for x in xs]
    y_lauc_q75 = [np.quantile(stats[x][lmetric], 0.75) for x in xs]

    global_col = '#424ef5'
    local_col = '#f57542'

    alpha_mean = 1.0
    alpha_q = 0.25
    alpha_area = 0.2

    plt.fill_between(xs, y_gauc_q25, y_gauc_median, color=global_col, alpha=alpha_area)
    plt.fill_between(xs, y_gauc_q75, y_gauc_median, color=global_col, alpha=alpha_area)

    plt.fill_between(xs, y_lauc_q25, y_lauc_median, color=local_col, alpha=alpha_area)
    plt.fill_between(xs, y_lauc_q75, y_lauc_median, color=local_col, alpha=alpha_area)

    plt.plot(xs, y_gauc_q25, '_', color=global_col, alpha=alpha_q)
    plt.plot(xs, y_gauc_median, '.', label='Combined', color=global_col, alpha=alpha_mean)
    plt.plot(xs, y_gauc_q75, '_', color=global_col, alpha=alpha_q)

    plt.plot(xs, y_lauc_q25, '_', color=local_col, alpha=alpha_q)
    plt.plot(xs, y_lauc_median, '.', label='Local', color=local_col, alpha=alpha_mean)
    plt.plot(xs, y_lauc_q75, '_', color=local_col, alpha=alpha_q)

    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xticks(xs, ['Centralized', '19 (weighted)', '19 (unweighted)'])
    plt.ylabel(metric)
    plt.legend(loc='lower left')
    plt.title(title)
    # plt.show()

    regular_col = '#a0a0a0'
    global_col = '#424ef5'
    local_col = '#f57542'

    alpha_mean = 1.0
    alpha_q = 0.25
    alpha_area = 0.2

    fig = plt.figure(figsize=(6, 4.5))

    ax = fig.add_subplot()

    ax.fill_between(xs, y_gauc_q25, y_gauc_median, color=global_col, alpha=alpha_area)
    ax.fill_between(xs, y_gauc_q75, y_gauc_median, color=global_col, alpha=alpha_area)

    ax.fill_between(xs, y_lauc_q25, y_lauc_median, color=local_col, alpha=alpha_area)
    ax.fill_between(xs, y_lauc_q75, y_lauc_median, color=local_col, alpha=alpha_area)

    ax.plot(xs, y_gauc_q25, '_', color=global_col, alpha=alpha_q)
    ax.plot(xs, y_gauc_median, '.', label='Combined', color=global_col, alpha=alpha_mean)
    ax.plot(xs, y_gauc_q75, '_', color=global_col, alpha=alpha_q)

    ax.plot(xs, y_lauc_q25, '_', color=local_col, alpha=alpha_q)
    ax.plot(xs, y_lauc_median, '.', label='Local', color=local_col, alpha=alpha_mean)
    ax.plot(xs, y_lauc_q75, '_', color=local_col, alpha=alpha_q)

    ax.hlines(y_gauc_q25[0], 0, 2, linestyles='dotted', colors=[regular_col])
    ax.hlines(y_gauc_median[0], 0, 2, linestyles='dotted', colors=[regular_col])
    ax.hlines(y_gauc_q75[0], 0, 2, linestyles='dotted', colors=[regular_col])

    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xticks(xs, ['Classical', '19 (weighted)', '19 (unweighted)'])
    plt.ylabel(metric)
    plt.xlabel('Number of Local Models')
    plt.legend()
    plt.title(title)

    for format in formats:
        plt.savefig(f'../../results/plots/BCTCGA_{metric}_sites.{format}', format=format, bbox_inches='tight')
