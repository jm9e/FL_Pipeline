from datetime import datetime

from pipeline.ensemble import EnsembleStep
from pipeline.evaluation import EvaluateStep, AnalysisStep
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import RepeatStep, NaturalSplitStep, DataStep, PrepareStep
from pipeline.result import WriteResultsStep
from pipeline.training import TrainRandomForestStep
from pipeline.util import PrintStep

if __name__ == "__main__":

    estimators = 100

    i = 0
    while True:
        start = datetime.now()

        s = i % 3
        if s == 0:
            regular = False
            weighted = True
        elif s == 1:
            regular = False
            weighted = False
        else:
            regular = True
            weighted = False

        i += 1

        print(f'Round {i}')

        p = Pipeline()

        # Specify dataset
        p.add_step(DataStep('../../datasets/tcga.csv', index_column='patient'))

        # Get classes
        p.add_step(PrepareStep('mol_subt'))

        # Repeat this dataset to increase robustness
        p.add_step(RepeatStep(4, 'done'))

        p.add_step(NaturalSplitStep('site', regular, weighted, 0.1))
        p.add_step(TrainRandomForestStep(lambda sites: estimators))

        # Create a global model from the individual classifiers
        p.add_step(EnsembleStep())

        # Evaluate the global model
        p.add_step(EvaluateStep())

        # Calculate values for further analysis
        p.add_step(AnalysisStep(f'../../results/evaluation/tcga_{estimators}_each', pr_samples=10))

        p.add_step(PrintStep(['meta'], data=True))
        p.add_step(PrintStep(['analysis'], data=False))

        # Write results
        p.add_step(WriteResultsStep(['sites', 'weighted', 'gauc', 'lauc', 'groc', 'lroc', 'gcm', 'lcm', 'gpr', 'lpr'],
                                    ['analysis']))

        p.compile()
        p.start()
        p.join()

        end = datetime.now()
        print(f'Round {i} took {(end - start).seconds} seconds')
