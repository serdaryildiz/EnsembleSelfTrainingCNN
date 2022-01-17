import argparse
import logging
import numpy

import torch
from torchvision import transforms
from torchvision.datasets import STL10

from utills import newExp, getPredictionDist
from AutoAugment.autoaugment import ImageNetPolicy

from BaseLearner.Models import BasicCNN
from BaseLearner.trainBaseLearner import trainBaseLearner

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args):
    # create new experiment dir
    expPath, expName = newExp()
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(filename=expPath + '/log.txt', mode='a'),
                                  logging.StreamHandler()]
                        )

    log = logging.getLogger(expName)
    log.info(f" \t New experiment directory: {expPath}")
    log.info(f' \t Device : {device}')

    test_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    testLoader = torch.utils.data.DataLoader(STL10(root='Data/STL10',
                                                   split='test',
                                                   transform=test_transforms),
                                             batch_size=128,
                                             shuffle=False,
                                             num_workers=0)

    unlabeled_loader = torch.utils.data.DataLoader(STL10(root='Data/STL10',
                                                         split='unlabeled',
                                                         transform=test_transforms),
                                                   batch_size=128,
                                                   shuffle=False,
                                                   num_workers=0)

    # get base learners
    baseLearners = []

    # get train data sets
    trainDatasets, valDatasets = getTrainValDataset_window(args.num_base_learner)
    for trainData, valData in zip(trainDatasets, valDatasets):
        log.info(' \t New Base Learner Training : ')
        trainLoader = torch.utils.data.DataLoader(trainData,
                                                  batch_size=128,
                                                  shuffle=True,
                                                  num_workers=0)
        valLoader = torch.utils.data.DataLoader(valData,
                                                batch_size=128,
                                                shuffle=False,
                                                num_workers=0)

        log.info(f' \t Train Dataset Len:{len(trainLoader.dataset)}')
        log.info(f' \t   Val Dataset Len:{len(valLoader.dataset)}')

        # get model
        log.info(f' \t Base Learner : {args.base_learner}')
        if args.base_learner == 'BasicCNN':
            model = BasicCNN(num_classes=10)
        model = model.to(device)

        model = trainBaseLearner(model, trainLoader, valLoader, expPath, expName, args)
        baseLearners.append(model)
        evalModel(model, testLoader, args, log)
        evalModel(model, valLoader, args, log, evalDataset='Val')
        evalModel(model, trainLoader, args, log, evalDataset='Train')

    getPredictionDist(baseLearners, testLoader, device=device, log=log)
    logging.shutdown()
    return baseLearners





def evalModel(model, test_loader, args, log=None, evalDataset='Test'):
    """
     evaluate model
    :param evalDataset: type of dataset
    :param model: model
    :param test_loader: test data loader
    :param args: arguments
    :param log: log for console information
    """
    if log is not None:
        log.info(f' \t \t Evaluation on {evalDataset} dataset  ')
    # criterion
    if args.criterion_BL == 'CEL':
        criterion = torch.nn.CrossEntropyLoss().to(device)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100. * correct / total
    if log is not None:
        log.info(f' \t \t  Model Acc : {acc} ')
    return


def getTrainValDataset_window(numBaseLearner, shuffle=False):
    """
        load 10-fold STL dataset and split with number of base learner
    :param numBaseLearner: number of base learner
    :param shuffle: if true, shuffle datasets
    :return: train and validation datasets
    """
    # dataset transforms
    train_transforms = transforms.Compose([ImageNetPolicy(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    # get 10 folds
    folds = []
    folds_val = []
    for f in range(10):
        folds.append(STL10(root='Data/STL10',
                           split='train',
                           folds=f,
                           transform=train_transforms))
        folds_val.append(STL10(root='Data/STL10',
                               split='train',
                               folds=f,
                               transform=val_transforms))

    if shuffle:
        idx = list(range(len(folds)))
        numpy.random.shuffle(idx)
        folds = folds[idx]
        folds_val = folds_val[idx]

    windowSize = 10 - numBaseLearner + 1
    trainDatasets = []
    valDataset = []
    for i in range(numBaseLearner):
        trainDatasets.append(torch.utils.data.ConcatDataset(folds[i:windowSize + i]))
        valDataset.append(torch.utils.data.ConcatDataset(folds_val[:i] + folds_val[windowSize + i:]))

    return trainDatasets, valDataset


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='STL10', help='dataset')
    parser.add_argument('--base-learner', type=str, default='BasicCNN', help='base learner')
    parser.add_argument('--num-base-learner', type=int, default=2, help='number of base learner')

    # Base Learner Parameters
    parser.add_argument('--optim-BL', type=str, default='Adam', help='optimizer')
    parser.add_argument('--lr-BL', type=int, default=0.001, help='learning rate')
    parser.add_argument('--scheduler-BL', type=str, default='MSLR', help='')
    parser.add_argument('--criterion-BL', type=str, default='CEL', help='learning rate')
    parser.add_argument('--epoch-BL', type=int, default=2, help='learning rate')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_opt()
    train(args)
