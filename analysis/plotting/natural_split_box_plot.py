import csv
import json

import matplotlib.pyplot as plt

if __name__ == '__main__':
    formats = ['png', 'pdf', 'svg']

    metrics = [
        {'gmetric': 'groc', 'lmetric': 'lroc', 'metric': 'AUC'},
        {'gmetric': 'gauc', 'lmetric': 'lauc', 'metric': 'PRAUC'},
    ]

    for metric in metrics:
        gmetric = metric['gmetric']
        lmetric = metric['lmetric']
        metric = metric['metric']

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

        # datainfo = str(len(stats['19,1'][f'{gmetric}'])) + ' / ' + str(len(stats['19,0'][f'{gmetric}']))
        # title += ' | ' + datainfo

        classical = stats['1,0'][gmetric]
        weighted = stats['19,1'][gmetric]
        unweighted = stats['19,0'][gmetric]

        fig, ax = plt.subplots()

        ax.boxplot([classical, weighted, unweighted])

        plt.xticks([1, 2, 3], ['Centralized', 'Weighted', 'Unweighted'])
        plt.ylabel(metric)
        # plt.legend()
        plt.title(title)

        for format in formats:
            plt.savefig(f'../../results/plots/BCTCGA_{metric}_sites.{format}', format=format, bbox_inches='tight')
