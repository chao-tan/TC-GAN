import importlib
import torch.utils.data
from datasets.cdataset import cdataset


def find_dataset_using_name(dataset_name):
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, cdataset):   # not case sensetive
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."%(dataset_filename,target_dataset_name))
    return dataset


def create_dataset(config):
    data_loader = CustomDatasetDataLoader(config)
    dataset = data_loader.load_data()
    return dataset



class CustomDatasetDataLoader(object):
    def __init__(self, config):

        # set different batchsize during training and testing_usr
        if config['status'] == "train":
            self.batchsize = int(config['train_batch_size'])
        elif config['status'] == 'test':
            self.batchsize = int(config['test_batch_size'])
        else:
            raise NotImplementedError("status is avaliable in train/test, but get %s in configs"%(config['status']))

        dataset_class = find_dataset_using_name(config['dataset_mode'])
        self.dataset = dataset_class(config)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=self.batchsize,
                                                      shuffle=True,
                                                      num_workers=int(config['num_threads']))


    def load_data(self):
        return self


    def __len__(self):
        return len(self.dataset)


    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data