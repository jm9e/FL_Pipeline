import csv
import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    formats = ['png', 'pdf', 'svg']

    metrics = [
        {'gmetric': 'groc', 'lmetric': 'lroc', 'metric': 'AUC'},
        {'gmetric': 'gauc', 'lmetric': 'lauc', 'metric': 'PRAUC'},
    ]

    datasets = [
        {'name': 'HCC', 'file': '../../results/evaluation/hcc_size_imbalance_100.csv', 'file2': '../../results/evaluation/hcc_multi_sites_100_each.csv'},
        {'name': 'ILPD', 'file': '../../results/evaluation/ilpd_size_imbalance_100.csv', 'file2': '../../results/evaluation/ilpd_multi_sites_100_each.csv'},
        {'name': 'LTD', 'file': '../../results/evaluation/tumor_size_imbalance_100.csv', 'file2': '../../results/evaluation/tumor_multi_sites_100_each.csv'},
        {'name': 'BCD', 'file': '../../results/evaluation/diag_size_imbalance_100.csv', 'file2': '../../results/evaluation/diag_multi_sites_100_each.csv'},
    ]

    for metric in metrics:
        gmetric = metric['gmetric']
        lmetric = metric['lmetric']
        metric = metric['metric']

        for ds in datasets:
            file = ds['file']
            file2 = ds['file2']
            name = ds['name']

            title = f'{name} | Size Imbalance'

            stats = {}
            # xs = []
            xs = ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5']
            # xs = ['1', '2', '5']

            with open(file, newline='') as csvfile:
                data = csv.reader(csvfile, delimiter=';')
                headers = next(data)
                gauc_idx = headers.index(gmetric)
                lauc_idx = headers.index(lmetric)

                for row in data:
                    stat = stats.get(row[1])
                    if not stat:
                        stat = {
                            f'{gmetric}_u': [],
                            f'{gmetric}_w': [],
                            'small': [],
                            'big': [],
                        }
                        stats[row[1]] = stat
                        # xs.append(row[1])

                    lvals = json.loads(row[lauc_idx])

                    if row[2] == '1':
                        stat[f'{gmetric}_w'].append(json.loads(row[gauc_idx]))
                    else:
                        stat[f'{gmetric}_u'].append(json.loads(row[gauc_idx]))
                    stat['small'].append(lvals[0])
                    stat['big'].append(lvals[1])

            # datainfo = str(len(stats['0.5'][f'{gmetric}_u'])) + ' / ' + str(len(stats['0.5'][f'{gmetric}_w']))
            # title += ' | ' + datainfo

            gstats = {}

            with open(file2, newline='') as csvfile:
                data = csv.reader(csvfile, delimiter=';')
                headers = next(data)
                gauc_idx = headers.index(gmetric)
                lauc_idx = headers.index(lmetric)

                for row in data:
                    if row[1] != '1':
                        continue
                    stat = gstats.get(row[1])
                    if not stat:
                        stat = {
                            gmetric: [],
                            lmetric: [],
                        }
                        gstats[row[1]] = stat

                    stat[gmetric].append(json.loads(row[gauc_idx]))

            y_g1auc_median = np.median(gstats['1'][f'{gmetric}'])
            y_g1auc_q25 = np.quantile(gstats['1'][f'{gmetric}'], 0.25)
            y_g1auc_q75 = np.quantile(gstats['1'][f'{gmetric}'], 0.75)

            y_ugauc_median = [np.median(stats[x][f'{gmetric}_u']) for x in xs]
            y_ugauc_q25 = [np.quantile(stats[x][f'{gmetric}_u'], 0.25) for x in xs]
            y_ugauc_q75 = [np.quantile(stats[x][f'{gmetric}_u'], 0.75) for x in xs]

            y_wgauc_median = [np.median(stats[x][f'{gmetric}_w']) for x in xs]
            y_wgauc_q25 = [np.quantile(stats[x][f'{gmetric}_w'], 0.25) for x in xs]
            y_wgauc_q75 = [np.quantile(stats[x][f'{gmetric}_w'], 0.75) for x in xs]

            y_sauc_median = [np.median(stats[x]['small']) for x in xs]
            y_sauc_q25 = [np.quantile(stats[x]['small'], 0.25) for x in xs]
            y_sauc_q75 = [np.quantile(stats[x]['small'], 0.75) for x in xs]

            y_bauc_median = [np.median(stats[x]['big']) for x in xs]
            y_bauc_q25 = [np.quantile(stats[x]['big'], 0.25) for x in xs]
            y_bauc_q75 = [np.quantile(stats[x]['big'], 0.75) for x in xs]

            regular_col = '#b0b0b0'
            globalu_col = '#424ef5'
            globalw_col = '#f53ee5'
            big_col = '#f57542'
            small_col = '#8bc23a'

            alpha_mean = 1.0
            alpha_q = 0.25
            alpha_area = 0.2

            fig = plt.figure(figsize=(6, 4.5))

            ax = fig.add_subplot()

            ax.hlines(y_g1auc_median, 0, len(xs) - 1, label='Classical', linestyles='dotted', colors=[regular_col])

            # middle = len(xs) // 2
            # ax.hlines(y_gauc_q25[middle], 0, len(xs) - 1, linestyles='dotted', colors=[regular_col])
            # ax.hlines(y_gauc_median[middle], 0, len(xs) - 1, colors=[regular_col])
            # ax.hlines(y_gauc_q75[middle], 0, len(xs) - 1, linestyles='dotted', colors=[regular_col])

            # ax.fill_between(xs, y_ugauc_q25, y_ugauc_median, color=globalu_col, alpha=alpha_area)
            # ax.fill_between(xs, y_ugauc_q75, y_ugauc_median, color=globalu_col, alpha=alpha_area)
            #
            # ax.fill_between(xs, y_wgauc_q25, y_wgauc_median, color=globalw_col, alpha=alpha_area)
            # ax.fill_between(xs, y_wgauc_q75, y_wgauc_median, color=globalw_col, alpha=alpha_area)
            #
            # ax.fill_between(xs, y_sauc_q25, y_sauc_median, color=small_col, alpha=alpha_area)
            # ax.fill_between(xs, y_sauc_q75, y_sauc_median, color=small_col, alpha=alpha_area)
            #
            # ax.fill_between(xs, y_bauc_q25, y_bauc_median, color=big_col, alpha=alpha_area)
            # ax.fill_between(xs, y_bauc_q75, y_bauc_median, color=big_col, alpha=alpha_area)

            # ax.plot(xs, y_ugauc_q25, '_', color=globalu_col, alpha=alpha_q)
            ax.plot(xs, y_ugauc_median, label='Combined (unweighted)', color=globalu_col, alpha=alpha_mean)
            ax.plot(xs, y_ugauc_median, '.', color=globalu_col, alpha=alpha_mean)
            # ax.plot(xs, y_ugauc_q75, '_', color=globalu_col, alpha=alpha_q)

            # ax.plot(xs, y_wgauc_q25, '_', color=globalw_col, alpha=alpha_q)
            ax.plot(xs, y_wgauc_median, label='Combined (weighted)', color=globalw_col, alpha=alpha_mean)
            ax.plot(xs, y_wgauc_median, '.', color=globalw_col, alpha=alpha_mean)
            # ax.plot(xs, y_wgauc_q75, '_', color=globalw_col, alpha=alpha_q)

            # ax.plot(xs, y_bauc_q25, '_', color=big_col, alpha=alpha_q)
            ax.plot(xs, y_bauc_median, label='Big Model', color=big_col, alpha=alpha_mean)
            ax.plot(xs, y_bauc_median, '.', color=big_col, alpha=alpha_mean)
            # ax.plot(xs, y_bauc_q75, '_', color=big_col, alpha=alpha_q)

            # ax.plot(xs, y_sauc_q25, '_', color=small_col, alpha=alpha_q)
            ax.plot(xs, y_sauc_median, label='Small Model', color=small_col, alpha=alpha_mean)
            ax.plot(xs, y_sauc_median, '.', color=small_col, alpha=alpha_mean)
            # ax.plot(xs, y_sauc_q75, '_', color=small_col, alpha=alpha_q)

            labels = ['5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', 'Balanced']

            # plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            plt.xticks(xs, labels)
            plt.ylabel(metric)
            plt.xlabel('Unbalancedness')
            plt.legend()
            plt.title(title)

            for format in formats:
                plt.savefig(f'../../results/plots/{name}_{metric}_size_imbalance.{format}', format=format, bbox_inches='tight')
