class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

DATASET_MAP = {
    'binary': "{}_binary_class.csv",
    'multi_class': "{}_multi_class.csv",
    'multi_label': "{}_multi_label.csv"
}