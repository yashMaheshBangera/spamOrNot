import pandas as pd

def get_spambase(dataset):
    dictionary = {}
    dictionary['features'] = dataset.data.features
    dictionary['targets'] = dataset.data.targets
    dictionary['data'] = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    dictionary['metadata'] = dataset.metadata
    dictionary['variables'] = dataset.variables
    return dictionary

