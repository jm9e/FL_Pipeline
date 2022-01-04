import os
from datetime import datetime

from pipeline.pipeline import PipelineStep, PipelinePacket


class WriteResultsStep(PipelineStep):

    def __init__(self, columns, labels=None):
        super().__init__(labels)
        self.__results = []
        self.columns = columns

    def process(self, item: PipelinePacket):
        filename = item.data['name'] + '.csv'
        if not os.path.isfile(filename):
            with open(filename, "w") as f:
                f.write(";".join(['time'] + self.columns))
                f.write("\n")
        with open(filename, "a") as f:
            f.write(";".join([f'{datetime.now()}'] + [f'{item.data[col]}' for col in self.columns]))
            f.write("\n")
        yield None

    def setup(self):
        pass

    def cleanup(self):
        pass
