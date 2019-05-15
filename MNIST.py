"""Train MNIST with PyTorch."""
from __future__ import print_function

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os

from models import *
from adabound import AdaBound


learning_rate = 0.05
final_learning_rate = 0.05
model_choice = 'SLP_Model'  # 'resnet', 'densenet', 'SLP_Model'
optim_choice = 'sgd'  # 'sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'amsbound'
momentum_choice = 0.9
beta_1 = 0.99
beta_2 = 0.999
resumed = '-r'
weights = 0.0001
gamma_choice = 0.1
epochs = 100



def build_dataset(): 
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    return train_loader, test_loader


def get_ckpt_name(model=model_choice, optimizer=optim_choice, lr=learning_rate, final_lr=final_learning_rate, momentum=momentum_choice,
                  beta1=beta_1, beta2=beta_2, gamma=gamma_choice):
    name = {
        'sgd': 'lr{}-momentum{}'.format(lr, momentum),
        'adagrad': 'lr{}'.format(lr),
        'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'amsgrad': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
        'amsbound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
    }[optimizer]
    return '{}-{}-{}'.format(model, optimizer, name)


def build_model(device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet': ResNet34,
        'densenet': DenseNet121,
        'Simple_MLP': MLP,
        'MLP_Dropout': MLP_drop,
        'SLP_model': S_L_P,
    }[model_choice]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(model_params):
    if optim_choice == 'sgd':
        return optim.SGD(model_params, learning_rate, momentum=momentum_choice,weight_decay=weights)
    elif optim_choice == 'adagrad':
        return optim.Adagrad(model_params, learning_rate,  weight_decay=0.0)
    elif optim_choice == 'adam':
        return optim.Adam(model_params, learning_rate, betas=(beta_1, beta_2),
                          weight_decay=weights)
    elif optim_choice == 'amsgrad':
        return optim.Adam(model_params, learning_rate, betas=(beta_1, beta_2),
                          weight_decay=weights, amsgrad=True)
    elif optim_choice == 'adabound':
        return AdaBound(model_params, learning_rate, betas=(beta_1, beta_2),
                        final_lr=final_learning_rate, gamma=gamma_choice,
                        weight_decay=weights)
    else:
        assert optim_choice == 'amsbound'
        return AdaBound(model_params, learning_rate, betas=(beta_1, beta_2),
                        final_lr=final_learning_rate, gamma=gamma_choice,
                        weight_decay=weights, amsbound=True)


def train(net, epoch, device, data_loader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)

    return accuracy


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)

    return accuracy



train_loader, test_loader = build_dataset()
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

ckpt_name = get_ckpt_name(model=model_choice, optimizer=optim_choice, lr=learning_rate,
                              final_lr=final_learning_rate, momentum=momentum_choice,
                              beta1=beta_1, beta2=beta_2, gamma=gamma_choice)

ckpt = None
best_acc = 0
start_epoch = -1

net = build_model(device, ckpt=ckpt)
criterion = nn.CrossEntropyLoss()
optimizer = create_optimizer(net.parameters())

train_accuracies = []
test_accuracies = []

for epoch in range(start_epoch + 1, epochs):
    train_acc = train(net, epoch, device, train_loader, optimizer, criterion)
    test_acc = test(net, device, test_loader, criterion)

# Save checkpoint.

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': test_acc,
        'epoch': epoch,
    }

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    if not os.path.isdir('curve'):
        os.mkdir('curve')
    torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                os.path.join('curve', ckpt_name))
