from typing import Generator

import numpy as np
import pandas as pd

from pipeline.pipeline import PipelineStep, PipelinePacket


class DataStep(PipelineStep):

    def __init__(self, data_path, index_column = None):
        super().__init__([])
        self.data_path = data_path
        self.index_column = index_column

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        data = pd.read_csv(self.data_path, index_col=self.index_column)
        data = data[~data.isnull().any(axis=1)]
        yield PipelinePacket('raw_data', data)

    def setup(self):
        pass

    def cleanup(self):
        pass


class NaturalSplitStep(PipelineStep):

    def __init__(self, site_col, regular=False, weighted=False, test=0.2):
        super().__init__(['data'])
        self.site_col = site_col
        self.regular = regular
        self.test = test
        self.weighted = weighted

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        data = item.data[0]
        classes = item.data[1]
        target = item.data[2]

        data = data.sample(frac=1)
        test_split = data.sample(n=int(data.shape[0] * self.test))
        train_data = data.drop(test_split.index)
        train_data = train_data.sample(frac=1)

        if not self.regular:
            sites = train_data['site'].unique()
            yield PipelinePacket('meta', (len(sites), self.weighted))
            for site in sites:
                train_data_split = train_data[train_data['site'] == site]
                train_data_split = train_data_split.drop(['site'], axis=1)
                yield PipelinePacket('train_data', (train_data_split, classes, target))
        else:
            yield PipelinePacket('meta', (1, False))
            train_data_split = train_data.drop(['site'], axis=1)
            yield PipelinePacket('train_data', (train_data_split, classes, target))

        test_split = test_split.drop(['site'], axis=1)
        yield PipelinePacket('test_data', (test_split, classes, target))

    def setup(self):
        pass

    def cleanup(self):
        pass


pd.options.mode.chained_assignment = None


class KalkPreprocessStep(PipelineStep):

    def setup(self):
        pass

    def cleanup(self):
        pass

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        df = item.data

        # Remove missing values
        df = df[df['kalk'].notnull()]
        # df = df.loc[df['kalk'] > 0, :]
        df = df[df['height'].notna()]
        df = df[df['weight'].notna()]
        df = df[df['waist'].notna()]
        df = df[df['chol'].notna()]
        df = df[df['tri'].notna()]
        df = df[df['hdl'].notna()]
        df = df[df['ldl'].notna()]
        df = df[df['hba'].notna()]
        # [df[col].fillna(df[col].median(), inplace=True) for col in
        #  ['height', 'weight', 'waist', 'chol', 'tri', 'hdl', 'ldl', 'hba']]

        df['age'] = 2013 - df['birth_year']

        df.sex = pd.Categorical(df.sex)
        df.sex = df.sex.cat.codes

        df.loc[df['kalk'] < 5, 'kalk_cat'] = 0
        df.loc[df['kalk'] >= 5, 'kalk_cat'] = 1

        virtual = []
        features = ['ldl', 'hdl', 'hba', 'chol', 'tri', 'waist', 'height']
        # features = []
        for a in range(len(features)):
            for b in range(a + 1, len(features), 1):
                col_a = features[a]
                col_b = features[b]
                if col_a == col_b:
                    continue
                nans = df[df[col_b] == 0]
                if nans.shape[0] != 0:
                    continue
                virt = f'{col_a}_{col_b}'
                df[virt] = df[col_a] / df[col_b]
                virtual.append(virt)

        df = df.filter(['kalk_cat', 'height', 'weight', 'waist', 'chol', 'tri', 'hdl', 'ldl', 'hba', 'age'] + virtual)

        yield PipelinePacket(item.label, df)


class RepeatStep(PipelineStep):

    def __init__(self, repetitions=1, done=None):
        super().__init__(None)
        self.repetitions = repetitions
        self.done = done

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        for _ in range(self.repetitions):
            yield item
        if self.done:
            yield PipelinePacket(self.done, None)

    def setup(self):
        pass

    def cleanup(self):
        pass


class BalanceStep(PipelineStep):

    def __init__(self, target, mode='remove'):
        super().__init__(['raw_data'])
        self.target = target
        self.mode = mode

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        data = item.data

        classes = data[self.target].unique()
        class_counts = data[self.target].value_counts()
        lower = np.min(class_counts)
        upper = np.max(class_counts)
        if lower < upper:
            if self.mode == 'remove':
                for i, c in enumerate(classes):
                    cls_data = data[data[self.target] == c]
                    cls_count = class_counts.iloc[i]
                    drop_indices = np.random.choice(cls_data.index, cls_count - lower,
                                                    replace=False)
                    data = data.drop(drop_indices)
            else:
                raise RuntimeError('unknown mode')
        yield PipelinePacket('data', (data, classes, self.target))

    def setup(self):
        pass

    def cleanup(self):
        pass


class PrepareStep(PipelineStep):

    def __init__(self, target):
        super().__init__(['raw_data'])
        self.target = target

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        data = item.data
        classes = np.array([0, 1])
        yield PipelinePacket('data', (data, classes, self.target))

    def setup(self):
        pass

    def cleanup(self):
        pass


class ImbalanceSplitStep(PipelineStep):

    def __init__(self, imbalance):
        super().__init__(['data'])
        self.imbalance = imbalance

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        data = item.data[0]
        classes = item.data[1]
        target = item.data[2]

        imb_int = (self.imbalance * data.shape[0]).astype(int).T
        for site, imb in enumerate(imb_int):
            offset = 0
            idx = np.zeros((imb.sum(),), dtype=int)

            for c_idx, c_count in enumerate(imb):
                cls = classes[c_idx]
                idx_cls = data[data[target] == cls].index
                idx[offset:offset + c_count] = \
                    np.random.choice(idx_cls, c_count, replace=False)
                offset += c_count

            label = 'train_data' if site < len(imb_int) - 1 else 'test_data'
            yield PipelinePacket(label, (data.loc[idx], classes, target))
            data = data.drop(idx)

    def setup(self):
        pass

    def cleanup(self):
        pass


class SitesSplitStep(PipelineStep):

    def __init__(self, sites, number_func, test=0.2):
        super().__init__(['data'])
        self.sites = sites
        self.test = test
        self.number_func = number_func

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        data = item.data[0]
        classes = item.data[1]
        target = item.data[2]

        for sites in self.sites:
            for _ in range(self.number_func(sites)):
                data = data.sample(frac=1)
                test_split = data.sample(n=int(data.shape[0] * self.test))
                train_data = data.drop(test_split.index)

                yield PipelinePacket('meta', (sites, False,))

                for i in range(sites):
                    split_samples = train_data.shape[0] // (sites - i)
                    train_data_split = train_data.iloc[:split_samples]

                    train_data = train_data.drop(train_data_split.index)

                    yield PipelinePacket('train_data', (train_data_split, classes, target))

                yield PipelinePacket('test_data', (test_split, classes, target))

    def setup(self):
        pass

    def cleanup(self):
        pass


class ClassImbalanceSplitStep(PipelineStep):

    def __init__(self, imbalances, test=0.2):
        super().__init__(['data'])
        self.imbalances = imbalances
        self.test = test

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        data = item.data[0]
        classes = item.data[1]
        target = item.data[2]

        for imbalance in self.imbalances:
            data = data.sample(frac=1)

            negatives = data[data[target] == 0]
            positives = data[data[target] == 1]

            pos_rel = imbalance
            neg_rel = 1 - imbalance

            pos = positives.shape[0]
            neg = (neg_rel / pos_rel) * pos
            if neg > negatives.shape[0]:
                neg = negatives.shape[0]
                pos = (pos_rel / neg_rel) * neg
            neg = int(neg)
            pos = int(pos)

            print(pos, neg)

            class_split1 = negatives.sample(n=neg)
            class_split2 = positives.sample(n=pos)
            
            test_split1 = class_split1.sample(frac=self.test)
            class_split1 = class_split1.drop(test_split1.index)
            test_split2 = class_split2.sample(frac=self.test)
            class_split2 = class_split2.drop(test_split2.index)

            train_splita1 = class_split1.sample(frac=0.5)
            train_splitb1 = class_split1.drop(train_splita1.index)
            train_splita2 = class_split2.sample(frac=0.5)
            train_splitb2 = class_split2.drop(train_splita2.index)

            yield PipelinePacket('meta', (imbalance, False))

            yield PipelinePacket('train_data', (train_splita1.append(train_splita2), classes, target))
            yield PipelinePacket('train_data', (train_splitb1.append(train_splitb2), classes, target))

            yield PipelinePacket('test_data', (test_split1.append(test_split2), classes, target))

            # data = class_split1.append(class_split2)
            # 
            # test_split = data.sample(n=int(data.shape[0] * self.test))
            # train_data = data.drop(test_split.index)
            # 
            # split1 = train_data.sample(frac=0.5)
            # split2 = train_data.drop(split1.index)
            # 
            # yield PipelinePacket('meta', (imbalance, False))
            # 
            # yield PipelinePacket('train_data', (split1, classes, target))
            # yield PipelinePacket('train_data', (split2, classes, target))
            # 
            # yield PipelinePacket('test_data', (test_split, classes, target))

    def setup(self):
        pass

    def cleanup(self):
        pass


class SizeImbalanceSplitStep(PipelineStep):

    def __init__(self, imbalances, weighted=False, test=0.2):
        super().__init__(['data'])
        self.imbalances = imbalances
        self.test = test
        self.weighted = weighted

    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        data = item.data[0]
        classes = item.data[1]
        target = item.data[2]

        for imbalance in self.imbalances:
            data = data.sample(frac=1)
            test_split = data.sample(n=int(data.shape[0] * self.test))
            train_data = data.drop(test_split.index)

            train_split1 = train_data.sample(frac=imbalance)
            train_split2 = train_data.drop(train_split1.index)

            yield PipelinePacket('meta', (imbalance, self.weighted,))
            yield PipelinePacket('train_data', (train_split1, classes, target))
            yield PipelinePacket('train_data', (train_split2, classes, target))
            yield PipelinePacket('test_data', (test_split, classes, target))

    def setup(self):
        pass

    def cleanup(self):
        pass


def get_stddev(d):
    """
    Returns class variance and size variance across sites.
    """
    classes = d.shape[0]
    sites = d.shape[1]
    s = d.sum()
    d = d / s
    cs = d.sum(axis=0)
    cv = d.std(axis=0)
    std1 = cv.sum() * classes
    std2 = cs.var() * sites * sites / (sites - 1) if sites > 1 else 0
    return std1, std2


def get_loss(d, class_stddev, size_stddev):
    cv, sv = get_stddev(d)
    dc, ds = abs(cv - class_stddev), abs(sv - size_stddev)
    return dc ** 2 + ds ** 2


def balanced(classes, splits=2, test_size=0.2):
    ib = np.zeros((classes, splits + 1))
    ib[:, :splits] = (1 - test_size) / (classes * splits)
    ib[:, splits] = test_size / classes
    return ib


def two_class_imbalance(imbalance, test_size=0.2):
    ib = np.zeros((2, 3))
    ib[0, 0] = imbalance
    ib[1, 0] = 1 - imbalance
    ib[0, 1] = 1 - imbalance
    ib[1, 1] = imbalance
    ib[:, :2] *= 1 - test_size
    ib[:, 2] = test_size
    return ib / 2


def imbalanced(classes, splits, class_stddev=0.0, size_stddev=0.0, test_size=0.2, resolution=1000):
    labels = list(range(classes))

    counts = [resolution // classes for i in range(classes)]
    c = np.array(counts)
    d = np.zeros((classes, splits), dtype=int)

    threshold = 1e-4

    while True:
        for _ in range(16):
            new_c = c.copy()
            new_d = d.copy()

            while new_c.sum() > 0:
                best_c = new_c.copy()
                best_d = new_d.copy()
                loss = None
                for _ in range(min(2 ** (classes * splits), 16 * 1048)):
                    c_cpy = new_c.copy()
                    d_cpy = new_d.copy()

                    for _ in range(min(classes, c_cpy.sum())):
                        cls = np.random.choice(labels, 1, p=c_cpy / c_cpy.sum())[0]
                        i = np.random.choice(list(range(splits)))
                        c_cpy[cls] -= 1
                        d_cpy[cls, i] += 1

                    new_loss = get_loss(d_cpy, class_stddev, size_stddev)

                    if loss is None or new_loss < loss:
                        best_c = c_cpy
                        best_d = d_cpy
                        loss = new_loss

                    if loss == 0:
                        break

                new_c = best_c
                new_d = best_d

            new_loss = get_loss(new_d, class_stddev, size_stddev)
            if new_loss < threshold:
                imb = np.zeros((classes, splits + 1))
                imb[:, :-1] = (new_d / resolution) * (1 - test_size)
                imb[:, -1] = test_size / classes
                return imb

        threshold *= 2
