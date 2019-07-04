import os
import matplotlib.pyplot as plt
import torch
import numpy as np


LABELS = ['SGD', 'Adam', 'AMSGrad', 'AdaBound', 'AMSBound']
Model_Name = 'SMLP' # 'SLP, 'Densenet', 'Resnet'

def get_folder_path(use_pretrained=True):
    path = 'curve'
    if use_pretrained:
        path = os.path.join(path)
    return path


def get_curve_data(use_pretrained=True, model=Model_Name):
    folder_path = get_folder_path(use_pretrained)
    filenames = [name for name in os.listdir(folder_path) if name.startswith(model.lower())]
    print(filenames)
    paths = [os.path.join(folder_path, name) for name in filenames]
    keys = [name.split('-')[1] for name in filenames]
    return {key: torch.load(fp) for key, fp in zip(keys, paths)}


def plot(use_pretrained=True, model=Model_Name, optimizers=None, curve_type='train'):
    assert model in [Model_Name], 'Invalid model name: {}'.format(model)
    assert curve_type in ['train', 'test'], 'Invalid curve type: {}'.format(curve_type)
    assert all(_ in LABELS for _ in optimizers), 'Invalid optimizer'

    curve_data = get_curve_data(use_pretrained, model=model)
    print(curve_data)

    plt.figure()
    plt.title('{} Accuracy for {} on CIFAR'.format(curve_type.capitalize(), model))
    plt.xlabel('Epoch')
    plt.ylabel('{} Accuracy %'.format(curve_type.capitalize()))
    # plt.ylim(83,93 if curve_type == 'train' else 96)
    # plt.ylim(75, 93 if curve_type == 'train' else 96)

    for optim in optimizers:
        linestyle = '--' if 'Bound' in optim else '-'
        accuracies = np.array(curve_data[optim.lower()]['{}_acc'.format(curve_type)])
        plt.plot(accuracies, label=optim, ls=linestyle)

    plt.grid(ls='--')
    plt.legend()
    plt.show()

plot(use_pretrained=True, model=Model_Name, optimizers=LABELS, curve_type='train')
plot(use_pretrained=True, model=Model_Name, optimizers=LABELS, curve_type='test')
