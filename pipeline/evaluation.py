import json
import random

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score

import numpy as np

from pipeline.pipeline import PipelineStep, PipelinePacket


class EvaluateStep(PipelineStep):

    def __init__(self):
        super().__init__(['classifier', 'test_data'])
        self.global_model = None
        self.local_models = None

    def process(self, item):
        if item.label == 'classifier':
            self.global_model = item.data['global_model']
            self.local_models = item.data['local_models']
            yield item

        elif item.label == 'test_data':
            cm, pr, roc = evaluate_classifier(self.global_model, item.data[0], item.data[1], item.data[2])
            yield PipelinePacket('global_cm', cm)
            if pr:
                yield PipelinePacket('global_pr', pr)
                yield PipelinePacket('global_roc', roc)
            if self.local_models is not None:
                for local_model in self.local_models:
                    cm, pr, roc = evaluate_classifier(local_model, item.data[0], item.data[1], item.data[2])
                    yield PipelinePacket('local_cm', cm)
                    if pr:
                        yield PipelinePacket('local_pr', pr)
                        yield PipelinePacket('local_roc', roc)
            yield PipelinePacket('end_evaluation', [item.data[0].shape[0]])

    def setup(self):
        pass

    def cleanup(self):
        pass


class AnalysisStep(PipelineStep):

    def __init__(self, name, variable='sites', pr_samples=10):
        super().__init__(['meta', 'end_evaluation', 'global_pr', 'global_cm', 'global_roc', 'local_pr', 'local_cm', 'local_roc'])

        self.name = name
        self.variable = variable

        self.cm = None
        self.pr = None
        self.roc = None

        self.cms = []
        self.prs = []
        self.rocs = []

        self.meta = None

        self.pr_samples = pr_samples

    def process(self, item):
        if item.label == 'end_evaluation':
            cm = self.cm
            pr = self.pr
            roc = self.roc

            cms = self.cms
            prs = self.prs
            rocs = self.rocs

            self.cm = None
            self.pr = None
            self.roc = None
            self.cms = []
            self.prs = []
            self.rocs = []

            global_cm = [[int(cm[0, 0]), int(cm[0, 1])],
                         [int(cm[1, 0]), int(cm[1, 1])]]
            global_pr = None
            global_auc = None
            global_roc = roc

            if pr:
                global_auc = auc(pr[1], pr[0])
                global_pr = list(zip(pr[1], pr[0])), self.pr_samples
                if len(global_pr) > self.pr_samples:
                    global_pr = random.sample(global_pr, self.pr_samples)

            local_cms = []
            local_prs = []
            local_aucs = []
            local_rocs = rocs

            for local_cm in cms:
                local_cms.append([[int(local_cm[0, 0]), int(local_cm[0, 1])],
                                  [int(local_cm[1, 0]), int(local_cm[1, 1])]])

            for local_pr in prs:
                local_aucs.append(auc(local_pr[1], local_pr[0]))

                local_pr = list(zip(local_pr[1], local_pr[0]))
                if len(local_pr) > self.pr_samples:
                    local_pr = random.sample(local_pr, self.pr_samples)
                local_prs.append(local_pr)

            yield PipelinePacket('analysis', {
                'name': self.name,
                self.variable: self.meta[0],
                'weighted': '1' if self.meta[1] else '0',

                'gauc': json.dumps(global_auc),
                'groc': json.dumps(global_roc),
                'gcm': json.dumps(global_cm),
                'gpr': json.dumps(global_pr),

                'lauc': json.dumps(local_aucs),
                'lroc': json.dumps(local_rocs),
                'lcm': json.dumps(local_cms),
                'lpr': json.dumps(local_prs),
            })

        elif item.label == 'global_cm':
            self.cm = item.data

        elif item.label == 'local_cm':
            self.cms.append(item.data)

        elif item.label == 'global_pr':
            self.pr = item.data

        elif item.label == 'local_pr':
            self.prs.append(item.data)

        elif item.label == 'global_roc':
            self.roc = item.data

        elif item.label == 'local_roc':
            self.rocs.append(item.data)

        elif item.label == 'meta':
            self.meta = item.data
            yield item

    def setup(self):
        pass

    def cleanup(self):
        pass


class StatsStep(PipelineStep):

    def __init__(self, labels):
        super().__init__(labels + ['done'])
        self.data = []

    def process(self, item: PipelinePacket):
        if item.label != 'done':
            self.data.append(item.data)

        # a = np.array(self.data)
        # acc = ((a[:, 0, 0] + a[:, 1, 1]) / a.sum(axis=(1, 2)))
        # print()
        # print(f'Total:      {a.shape[0]}')
        # print(f'Acc mean:   {np.mean(acc)}')
        # print(f'Acc median: {np.median(acc)}')
        # print(f'Acc std:    {np.std(acc)}')
        # print(f'Acc min:    {np.min(acc)}')
        # print(f'Acc max:    {np.max(acc)}')
        # print()

        if item.label == 'done':
            a = np.array(self.data)
            acc = ((a[:, 0, 0] + a[:, 1, 1]) / a.sum(axis=(1, 2)))
            cm = (a.sum(axis=0) / a.sum(axis=(0, 1, 2)))

            print()
            # print(f'CM:         {cm}')
            print(f'Total:      {a.shape[0]}')
            print(f'Acc mean:   {np.mean(acc)}')
            print(f'Acc median: {np.median(acc)}')
            print(f'Acc std:    {np.std(acc)}')
            print(f'Acc min:    {np.min(acc)}')
            print(f'Acc max:    {np.max(acc)}')
            print()

        yield item

    def setup(self):
        pass

    def cleanup(self):
        pass


def evaluate_classifier(model, data, classes, target):
    """
    :param model: needs to provide a predict and predict_proba method
    :param data: test dataset
    :param classes: labels in the dataset
    :param target: target label to predict
    :return: confusion matrix and precision/recall curve
    """
    X = data.drop(columns=[target]).values
    y_true = data[target].values

    # Fixed threshold prediction
    y_preds = model.predict(X)
    y_pred_idx = np.argmax(y_preds, axis=1)
    y_pred = classes[y_pred_idx]

    cm = confusion_matrix(y_true, y_pred)
    pr = None
    roc = None

    if len(classes) == 2:
        # Probability predictions
        y_preds_prob = model.predict_proba(X)[:, 1]
        p, r, t = precision_recall_curve(y_true, y_preds_prob)
        roc = roc_auc_score(y_true, y_preds_prob)
        pr = [p, r, t]

    return cm, pr, roc
