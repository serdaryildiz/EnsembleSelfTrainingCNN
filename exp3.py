"""
Experiment 3 :
    1 - Create N CNN base-learner
    2 - Calculate pseudo labels using the N base learners
    3 - Train Base Learners using pseudo labeled data
    4 - Unify decisions

using different validation datasets
"""
import argparse
import logging
import numpy

import torch
from torchvision import transforms
from torchvision.datasets import STL10

from exp3Utils import getDataLoaders, getPseudoLabels, getTrainDataForBaseLearners, getPretrain
from utills import newExp, plotKappaErrorGraph, evalModel
from AutoAugment.autoaugment import ImageNetPolicy

from BaseLearner.Models import BasicCNN
from BaseLearner.trainBaseLearner import trainBaseLearner

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seeds = [42, 611, 432, 114, 739, 642, 923, 445, 316, 889, 128, 775, 66]


def main(args):
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

    # get base learners
    baseLearners = []
    for i in range(args.num_base_learner):

        # get dataset iterators
        trainLoader, valLoader, unlabeledLoader, testLoader = getDataLoaders(i, args)

        log.info(f' \t {i + 1}. base learner is training...')
        torch.manual_seed(seeds[i])
        model = BasicCNN()
        model = model.to(device)


        model = getPretrain(model, i, log=log)

        # get pseudo labels using base-learner
        pseudoLabels = getPseudoLabels(model, unlabeledLoader, log=log)

        # get new train dataset for new base learners
        trainLoader = getTrainDataForBaseLearners(pseudoLabels, i, args, log)

        # args.lr_BL = args.lr_BL *.01
        # args.label_smoothing = 0.3

        # train base learners
        model = trainBaseLearner(model, trainLoader, valLoader, expPath, expName, args)

        # evaluate model
        evalModel(model, testLoader, args, log)
        evalModel(model, valLoader, args, log, evalDataset='Validation')
        baseLearners.append(model)

    # plot kappa-error graph
    plotKappaErrorGraph(baseLearners, testLoader, expPath, log)

    logging.shutdown()
    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='STL10', help='dataset')
    parser.add_argument('--base-learner', type=str, default='BasicCNN', help='base learner')
    parser.add_argument('--num-base-learner', type=int, default=10, help='number of base learner')

    # Base Learner Parameters
    # parser.add_argument('--optim-BL', type=str, default='SGD', help='optimizer')
    parser.add_argument('--optim-BL', type=str, default='Adam', help='optimizer')
    # parser.add_argument('--lr-BL', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-BL', type=float, default=0.0001, help='learning rate')
    # parser.add_argument('--scheduler-BL', type=str, default='MSLR', help='')
    parser.add_argument('--scheduler-BL', type=str, default='CyclicLR', help='')
    parser.add_argument('--criterion-BL', type=str, default='CEL', help='learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0, help='')
    parser.add_argument('--epoch-BL', type=int, default=20, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_opt()
    main(args)
