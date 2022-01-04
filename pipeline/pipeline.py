import queue
import threading
from abc import abstractmethod, ABC
from typing import Generator, List


class PipelinePacket:

    def __init__(self, label, data):
        self.label = label
        self.data = data


class PipelineStep(ABC):

    def __init__(self, labels: List[str] or None):
        self.q_in = None
        self.q_out = None
        self.t = None
        self.labels = labels

    def set_q_in(self, q: queue.Queue):
        self.q_in = q

    def set_q_out(self, q: queue.Queue):
        self.q_out = q

    def __worker(self):
        if self.q_in is None and self.q_out is None:
            return

        self.setup()

        if self.q_in is None:
            # We are probably the first step in the pipeline
            for i_out in self.process(None):
                self.q_out.put(i_out)

        else:
            # We have a preceding step whose items we need to process
            while True:
                item: PipelinePacket = self.q_in.get()
                if item is None:
                    break

                if self.labels is not None and item.label not in self.labels:
                    if self.q_out is not None:
                        self.q_out.put(item)
                    continue

                if self.q_out is None:
                    # We are probably the last step in the pipeline so we ignore the output
                    for _ in self.process(item):
                        pass

                else:
                    # We have a succeeding step to which we need to push the items
                    for i_out in self.process(item):
                        self.q_out.put(i_out)

                self.q_in.task_done()

        if self.q_out is not None:
            self.q_out.put(None)

        self.cleanup()

    def start(self):
        self.t = threading.Thread(target=self.__worker, daemon=True)
        self.t.start()

    def join(self):
        self.t.join()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def process(self, item: PipelinePacket) -> Generator[PipelinePacket, None, None]:
        return
        yield


class Pipeline:

    def __init__(self):
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def compile(self):
        q = None
        for s in self.steps[:-1]:
            s.set_q_in(q)
            q = queue.Queue()
            s.set_q_out(q)
        self.steps[-1].set_q_in(q)

    def start(self):
        for s in self.steps:
            s.start()

    def join(self):
        for s in self.steps:
            s.join()
