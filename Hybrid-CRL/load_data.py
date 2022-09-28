import json
import pickle
from pathlib import Path

from tqdm import tqdm


class InputExample:
    '''Data structure for input examples.
    '''
    def __init__(self, guid, code, label=None):
        '''
        '''
        self.guid = guid
        self.code = code
        self.label = label


class RawDataLoader:
    '''Loads data from raw data files.
    '''
    def __init__(self, task, stage, data_dir):
        '''Initalizes :class: ``RawDataLoader``.
        '''
        self.task = task
        self.stage = stage
        self.data_dir = data_dir

    def load_data(self):
        '''Data generator, yields code snippet and ground truth label as
        a tuple.
        '''
        data_path = Path(self.data_dir) / f"{self.task}_{self.stage}.txt"

        with open(str(data_path), 'r') as text_file:
            data = text_file.readlines()

        for row in data:
            row_contents = row.split('\t')
            id, code, label = row_contents[0], row_contents[1], int(row_contents[2])
            yield id, code, label


class DataProcessor:
    '''Processes data and creates examples for both tasks.
    '''
    def __init__(self, data_dir='./datasets'):
        '''Initialize :class: ``DataProcessor``.
        '''
        self.data_dir = data_dir

    def get_train_examples(self, task):
        '''Retrieve examples for train partition.
        '''
        examples = []
        try:
            examples = self.load_examples(task, 'train')
        except FileNotFoundError:
            examples = self.create_examples(task, 'train')
        return examples


    def get_val_examples(self, task):
        '''Retrieve examples for validation partition.
        '''
        try:
            examples = self.load_examples(task, 'val')
        except FileNotFoundError:
            examples = self.create_examples(task, 'val')
        return examples

    def get_test_examples(self, task):
        '''Retrieve examples for test partition.
        '''
        try:
            examples = self.load_examples(task, 'test')
        except FileNotFoundError:
            examples = self.create_examples(task, 'test')
        return examples

    def create_examples(self, task, stage):
        '''Create ``InputExample`` objects for both tasks.
        '''
        rdl = RawDataLoader(task, stage, self.data_dir)
        examples = []

        for (id, code, label) in tqdm(rdl.load_data()):
            guid = f'{task}-{stage}-{id}'

            if stage == 'test':
                label = None

            examples.append(
                InputExample(guid=guid,
                             code=code,
                             label=label)
            )

        self.save(examples, task, stage)

        if stage == 'test':
            instances = list(rdl.load_data())
            return examples, instances
        else:
            return examples

    def load_examples(self, task, stage):
        '''
        '''
        path_to_file = Path(self.data_dir) / f"{task}_examples_{stage}.pkl"

        with open(str(path_to_file), 'rb') as handler:
            examples = pickle.load(handler)

        return examples

    def save(self, examples, task, stage):
        '''
        '''
        path_to_save = Path(self.data_dir)
        path_to_save.mkdir(exist_ok=True, parents=True)
        path_to_file = path_to_save / \
            f"{task}_examples_{stage}.pkl"

        with open(str(path_to_file), 'wb') as handler:
            pickle.dump(examples, handler)
