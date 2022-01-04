from typing import List

from pipeline.pipeline import PipelineStep, PipelinePacket


class StartStep(PipelineStep):

    def __init__(self, items: List[PipelinePacket]):
        super().__init__([])
        self.items = items

    def process(self, item: PipelinePacket):
        for i in self.items:
            yield i

    def setup(self):
        pass

    def cleanup(self):
        pass


class PrintStep(PipelineStep):

    def __init__(self, labels, prefix='', data=False):
        super().__init__(labels)
        self.prefix = prefix
        self.data = data

    def process(self, item: PipelinePacket):
        if self.prefix:
            print(f'{self.prefix}{item.label}', end='')
        else:
            print(item.label, end='')

        if self.data:
            print(f' {item.data}', flush=True)
        else:
            print(flush=True)

        yield item

    def setup(self):
        pass

    def cleanup(self):
        pass


class FilterStep(PipelineStep):

    def __init__(self, labels, remove=True):
        super().__init__([])
        self.labels = labels
        self.remove = remove

    def process(self, item: PipelinePacket):
        if self.remove and item.label not in self.labels:
            yield item
        elif not self.remove and item.label in self.labels:
            yield item

    def setup(self):
        pass

    def cleanup(self):
        pass


class EndStep(PipelineStep):

    def __init__(self, labels=None):
        super().__init__(labels)
        self.__results = []

    @property
    def results(self):
        return self.__results

    def process(self, item: PipelinePacket):
        self.__results.append(item.data)

    def setup(self):
        pass

    def cleanup(self):
        pass
