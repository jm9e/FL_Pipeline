import csv
import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    formats = ['png', 'pdf', 'svg', 'eps']

    gmetric = 'gacc'
    lmetric = 'lacc'
    metric = 'ACC'

    datasets = [
        {'name': 'HCC', 'file': '../../results/evaluation/hcc_multi_sites_100_each.csv'},
        {'name': 'ILPD', 'file': '../../results/evaluation/ilpd_multi_sites_100_each.csv'},
        {'name': 'LTD', 'file': '../../results/evaluation/tumor_multi_sites_100_each.csv'},
        {'name': 'BCD', 'file': '../../results/evaluation/diag_multi_sites_100_each.csv'},
    ]

    for ds in datasets:
        file = ds['file']
        name = ds['name']

        title = f'{name} | Multiple Local Models'

        stats = {}
        xs = ['1', '2', '5', '10', '20', '50', '100']

        with open(file, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=';')
            headers = next(data)
            gcm_idx = headers.index('gcm')
            lcm_idx = headers.index('lcm')

            for row in data:
                stat = stats.get(row[1])
                if not stat:
                    stat = {
                        gmetric: [],
                        lmetric: [],
                    }
                    stats[row[1]] = stat

                gcm = json.loads(row[gcm_idx])
                gacc = (gcm[0][0] + gcm[1][1]) / (gcm[0][0] + gcm[0][1] + gcm[1][0] + gcm[1][1])

                stat[gmetric].append(gacc)

                lvals = json.loads(row[lcm_idx])

                if len(lvals) > 0:
                    for lcm in lvals:
                        lacc = (lcm[0][0] + lcm[1][1]) / (lcm[0][0] + lcm[0][1] + lcm[1][0] + lcm[1][1])
                        stat[lmetric].append(lacc)
                else:
                    stat[lmetric].append(gacc)

        # datainfo = str(len(stats['100'][gmetric]))
        # title += ' | ' + datainfo

        y_gauc_median = [np.median(stats[x][gmetric]) for x in xs]
        y_gauc_q25 = [np.quantile(stats[x][gmetric], 0.25) for x in xs]
        y_gauc_q75 = [np.quantile(stats[x][gmetric], 0.75) for x in xs]

        y_lauc_median = [np.median(stats[x][lmetric]) for x in xs]
        y_lauc_q25 = [np.quantile(stats[x][lmetric], 0.25) for x in xs]
        y_lauc_q75 = [np.quantile(stats[x][lmetric], 0.75) for x in xs]

        xs = [int(x) for x in xs]

        regular_col = '#a0a0a0'
        global_col = '#424ef5'
        local_col = '#f57542'

        alpha_mean = 1.0
        alpha_q = 0.25
        alpha_area = 0.2

        fig = plt.figure(figsize=(6, 4.5))

        ax = fig.add_subplot()

        ax.hlines(y_gauc_q25[0], 1, 100, linestyles='dotted', colors=[regular_col])
        ax.hlines(y_gauc_median[0], 1, 100, linestyles='dotted', colors=[regular_col])
        ax.hlines(y_gauc_q75[0], 1, 100, linestyles='dotted', colors=[regular_col])

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

        plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.xscale('log')
        plt.xticks([1, 2, 5, 10, 20, 50, 100], ['Centralized', '2', '5', '10', '20', '50', '100'])
        plt.ylabel(metric)
        plt.xlabel('Number of Local Models')
        plt.legend()
        plt.title(title)

        for format in formats:
            plt.savefig(f'../../results/plots/{name}_{metric}_sites.{format}', format=format, bbox_inches='tight')
