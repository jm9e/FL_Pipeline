from typing import Generator

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from pipeline.pipeline import PipelineStep, PipelinePacket


class VoidClassifier(ClassifierMixin):

    def __init__(self, classes, label):
        self.cn = classes.shape[0]
        self.idx = list(classes).index(label)

    def predict(self, X):
        preds = np.zeros((X.shape[0], 1), dtype=np.int)
        preds[:] = self.idx
        return preds.T

    def predict_proba(self, X):
        preds = np.zeros((X.shape[0], self.cn))
        preds[:, self.idx] = 1
        return preds


class TrainRandomForestStep(PipelineStep):

    def __init__(self, estimators):
        self.estimators = estimators
        self.meta = None
        super().__init__(['meta', 'train_data'])

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        if item.label == 'meta':
            self.meta = item.data
            yield item

        else:
            data = item.data[0]
            classes = item.data[1]
            target = item.data[2]

            X = data.drop(columns=[target]).values
            y = data[target].values
            estimators = self.estimators(self.meta[0])

            if len(set(y)) > 1:
                rf = RandomForestClassifier(n_estimators=estimators)
                rf.fit(X, y)
                yield PipelinePacket('classifier_list', (list(rf.estimators_), (X, y), classes, target))
            else:
                print(f'Degenerated tree ({y})')
                classifiers = [VoidClassifier(classes, y[0]) for _ in range(estimators)]
                yield PipelinePacket('classifier_list', (classifiers, (X, y), classes, target))

    def setup(self):
        pass

    def cleanup(self):
        pass
