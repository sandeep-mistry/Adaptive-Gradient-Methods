# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus
from models.One_Layer_LSTM import RNNLM
from models.Two_Layer_LSTM import RNNLM_2
from models.Three_Layer_LSTM import RNNLM_3
from adabound import AdaBound
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Training')
    parser.add_argument('--model', default='One_Layer_LSTM', type=str, help='model',
                        choices=['One_Layer_LSTM','Two_Layer_LSTM','Three_Layer_LSTM'])
    parser.add_argument('--optim', default='adam', type=str, help='optimizer',
                        choices=['sgd','adam','adabound', 'amsbound'])
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.001, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.99, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    return parser


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Running with GPU Acceleration")
else:
    device = torch.device('cpu')
    print("Running on CPU")

# Hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 50
num_samples = 1000  # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.001
final_learning_rate = 0.0001
step_size = 37
beta_1 = 0.9
beta_2 = 0.99

# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data('data/train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length

def get_ckpt_name(model='One_Layer_LSTM', optimizer='adam', lr=learning_rate, final_lr=final_learning_rate, momentum=0.9,
                  beta1=beta_1, beta2=beta_2, gamma=0.1):
    name = {
        'sgd': 'lr{}-momentum{}'.format(lr, momentum),
        'adagrad': 'lr{}'.format(lr),
        'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'amsgrad': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
        'amsbound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
    }[optimizer]
    return '{}-{}-{}'.format(model, optimizer, name)


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(ckpt_name)

def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'One_Layer_LSTM': RNNLM,
        'Two_Layer_LSTM': RNNLM_2,
        'Thre_Layer_LSTM': RNNLM_3,
    }[args.model](vocab_size, embed_size, hidden_size, num_layers)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net

# model = RNNLM_3(vocab_size, embed_size, hidden_size, num_layers).to(device)

# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum)
    elif args.optim == 'adam':
        return optim.Adam(model_params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma)

    else:
        assert args.optim == 'amsbound'
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                         amsbound=True)



# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]

parser = get_parser()
args = parser.parse_args()

ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              final_lr=args.final_lr, momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, gamma=args.gamma)
if args.resume:
    ckpt = load_checkpoint(ckpt_name)
    best_acc = ckpt['acc']
    start_epoch = ckpt['epoch']
else:
    ckpt = None
    best_acc = 0
    start_epoch = -1

net = build_model(args, device, ckpt=ckpt)
criterion = nn.CrossEntropyLoss()
optimizer = create_optimizer(args, net.parameters())
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1,
#                                           last_epoch=start_epoch)

train_perplexities = []

# Train the model
for epoch in range(start_epoch + 1, num_epochs):
    # scheduler.step()
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))

    for i in range(0, ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i + seq_length].to(device)
        targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)

        # Forward pass
        states = detach(states)
        outputs, states = net(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        # Backward and optimize
        net.zero_grad()
        loss.backward()
        clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        # perplexity = np.exp(loss.item())

        step = (i + 1) // seq_length
        if step % 100 == 0:
            # perplexity = np.exp(loss.item())
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                  .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))
# Save checkpoint.
    perplexity = np.exp(loss.item())
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': perplexity,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
         os.mkdir('checkpoint')
    torch.save(state, os.path.join('checkpoint', ckpt_name))
    best_acc = perplexity

    train_perplexities.append(perplexity)
    # test_accuracies.append(test_acc)
    if not os.path.isdir('curve'):
        os.mkdir('curve')
    torch.save({'train_acc': train_perplexities},
                os.path.join('curve', ckpt_name))
# Test the model
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # Select one word id randomly
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # Forward propagate RNN
            output, state = net(input, state)

            # Sample a word id
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)

            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i + 1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))

# Save the model checkpoints
torch.save(net.state_dict(), 'model.ckpt')