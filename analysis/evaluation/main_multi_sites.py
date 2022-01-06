from datetime import datetime

from pipeline.ensemble import EnsembleStep
from pipeline.evaluation import EvaluateStep, AnalysisStep
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import DataStep, RepeatStep, SitesSplitStep, PrepareStep
from pipeline.result import WriteResultsStep
from pipeline.training import TrainRandomForestStep
from pipeline.util import PrintStep

if __name__ == "__main__":
    # imbalances = [two_class_imbalance(i / steps / 2) for i in range(steps + 1)]

    # imbalance = np.array([[0.225, 0.225, 0.05], [0.225, 0.225, 0.05]])
    # imbalance = np.array([[0.4, 0.1], [0.4, 0.1]])
    # imbalance = np.array([
    #     [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    #     [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    # ])

    i = 0
    while True:
        start = datetime.now()

        sites = [1, 2, 5, 10, 20, 50, 100]

        # s = i % 4
        s = 0
        if s == 0:
            estimators = 100
            total = False
            sites = [1]
            dataset = 'hcc'
            dataset_file = 'hcc_data.csv'
            target = 'HCC'
            index = None

        elif s == 1:
            estimators = 100
            total = False
            sites = [1, 2, 5, 10, 20, 50, 100]
            dataset = 'ilpd'
            dataset_file = 'ilpd.csv'
            target = '10'
            index = 'sid'

        elif s == 2:
            estimators = 100
            total = False
            sites = [1, 2, 5, 10, 20, 50, 100]
            dataset = 'tumor'
            dataset_file = 'tumor_data_thresh30.csv'
            target = 'os_time_binary'
            index = 'patient'

        else:
            estimators = 100
            total = False
            sites = [1, 2, 5, 10, 20, 50, 100]
            dataset = 'diag'
            dataset_file = 'diagnosis.csv'
            target = '1'
            index = '0'

        i += 1

        print(f'Round {i} (file: {dataset_file})')

        p = Pipeline()

        # Specify dataset
        p.add_step(DataStep(f'../../datasets/{dataset_file}', index))

        # Repeat this dataset to increase robustness
        p.add_step(RepeatStep(1, 'done'))

        p.add_step(PrepareStep(target))
        # p.add_step(BalanceStep(target))

        # Ensure there is a 50:50 balance of target labels
        # p.add_step(BalanceStep('HCC'))

        # Split the dataset
        # p.add_step(SitesSplitStep(list(range(1, 21, 1)), 0.1))
        # p.add_step(SitesSplitStep([1, 2, 3, 5, 10, 20, 30, 50, 100], 0.1))
        # p.add_step(SitesSplitStep([1, 2, 5, 10, 20, 50, 100], lambda sites: 100 // sites, 0.1))

        if total:
            name = f'../../results/evaluation/{dataset}_multi_sites_{estimators}_total'
            p.add_step(SitesSplitStep(sites, lambda _: estimators, 0.1))
            p.add_step(TrainRandomForestStep(lambda sites: estimators // sites))

        else:
            name = f'../../results/evaluation/{dataset}_multi_sites_{estimators}_each'
            p.add_step(SitesSplitStep(sites, lambda sites: estimators // sites, 0.1))
            p.add_step(TrainRandomForestStep(lambda sites: estimators))

        # Create a global model from the individual classifiers
        p.add_step(EnsembleStep())

        # Evaluate the global model
        p.add_step(EvaluateStep())

        # Calculate values for further analysis
        p.add_step(AnalysisStep(name, pr_samples=10))

        p.add_step(PrintStep(['meta'], data=True))

        # Write results
        p.add_step(WriteResultsStep(['sites', 'gauc', 'lauc', 'groc', 'lroc', 'gcm', 'lcm', 'gpr', 'lpr'],
                                    ['analysis']))

        p.compile()
        p.start()
        p.join()

        end = datetime.now()
        print(f'Took {(end - start).seconds} seconds to complete')
