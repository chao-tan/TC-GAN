import importlib
from modules.comdel import cmodel


def find_model_using_name(model_name):
    model_filename = "modules.models." + model_name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = "network"
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, cmodel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of abstract with class name that matches %s."%(model_filename,target_model_name))
        exit(0)

    return model



def create_model(config):
    model = find_model_using_name(config['model'])
    instance = model(config)
    print("module [%s] was created" % type(instance).__name__)
    return instance


