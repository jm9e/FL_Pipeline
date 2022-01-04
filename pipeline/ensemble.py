import functools
import operator
import random

import numpy as np

from sklearn.base import ClassifierMixin

from pipeline.pipeline import PipelineStep, PipelinePacket


class ConsensusEnsemble(ClassifierMixin):

    def __init__(self, classifiers, classes, weights=None):
        self.classifiers = classifiers
        self.classes = classes
        if not weights:
            self.weights = [1 for _ in classifiers]
        else:
            if len(classifiers) == len(weights):
                self.weights = weights
            else:
                raise RuntimeError(f'Expected {len(classifiers)} weights, got {len(weights)}')

    def predict(self, X):
        my = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classifiers):
            y = c.predict(X)
            for j, yl in enumerate(y):
                class_idx = np.where(self.classes == yl)
                my[j, class_idx] = my[j, class_idx] + self.weights[i]
        # pred_idx = np.argmax(my, axis=1)
        return my

    def predict_proba(self, X):
        my = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classifiers):
            y = c.predict_proba(X)
            for j, yl in enumerate(y):
                my[j, :] = my[j, :] + yl * self.weights[i]
        return my / my.sum(axis=1, keepdims=True)


class EnsembleStep(PipelineStep):

    def __init__(self):
        super().__init__(['classifier_list', 'test_data', 'meta'])
        self.site_classifiers = []
        self.site_weights = None
        self.meta = None

    def process(self, item):
        if item.label == 'classifier_list':
            # Collect
            self.site_classifiers.append(item.data[0])
            if self.site_weights is not None:
                self.site_weights.append(item.data[1][0].shape[0])

        if item.label == 'meta':
            self.meta = item.data

            self.site_classifiers = []
            if item.data[1]:
                self.site_weights = []
            else:
                self.site_weights = None

            yield item

        elif item.label == 'test_data':
            sample_k = 100 // len(self.site_classifiers)
            global_model = ConsensusEnsemble(
                functools.reduce(operator.iconcat,
                                 [random.sample(cfs, k=sample_k) for cfs in self.site_classifiers],
                                 []),
                item.data[1],
                functools.reduce(operator.iconcat,
                                 [[weight] * sample_k for weight in self.site_weights],
                                 []) if self.site_weights is not None else None)

            local_models = None
            if len(self.site_classifiers) > 1:
                local_models = [ConsensusEnsemble(clf, item.data[1]) for clf in self.site_classifiers]

            yield PipelinePacket('classifier', {
                'global_model': global_model,
                'local_models': local_models,
            })
            yield item

    def setup(self):
        pass

    def cleanup(self):
        pass
