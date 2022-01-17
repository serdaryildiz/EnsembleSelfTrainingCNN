"""

"""
import argparse
import logging
import numpy
import tqdm

import torch.nn.functional as F
import torch
from torchvision import transforms
from torchvision.datasets import STL10

from utills import newExp, evalModel
from AutoAugment.autoaugment import ImageNetPolicy

from BaseLearner.Models import BasicCNN

import matplotlib.pyplot as plt

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

    # get dataset iterators
    _, __, unlabeledLoader, testLoader = getDataLoaders(0, args)

    # get base learners
    baseLearners = []
    acc = []
    score_min = []
    score_max = []
    for i in range(args.num_base_learner):

        log.info(f' \t {i + 1}. base learner is training...')
        torch.manual_seed(seeds[i])
        model = BasicCNN()
        model = model.to(device)

        # get pretrain
        model = getPretrain(model, i, log=log)

        # evaluate model
        evalModel(model, testLoader, args, log)
        # evalModel(model, valLoader, args, log, evalDataset='Validation')

        # get pseudo labels using base-learner
        # pseudoLabels = getPseudoLabels(model, valLoader, log=log)

        # pseudoLabels = getPseudoLabels(model, unlabeledLoader, log=log)
        # pseudoScores, pseudoLabels = pseudoLabels.max(1)
        # scores_unlabeled, indexes = torch.sort(pseudoScores)

        pseudoLabels = getPseudoLabels(model, testLoader, log=log)
        pseudoScores, pseudoLabels = pseudoLabels.max(1)
        scores, indexes = torch.sort(pseudoScores)

        # n_bins = 100
        # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        # axs[0].hist(scores.detach().cpu().numpy(), bins=n_bins, range=(0, 1))
        # axs[0].set_title('Test Data Probability Histogram', fontdict={'fontsize': 8})
        # axs[1].hist(scores_unlabeled.detach().cpu().numpy(), bins=n_bins, range=(0, 1))
        # axs[1].set_title('Unlabeled Data Probability Histogram', fontdict={'fontsize': 8})
        # fig.savefig(expPath + f'/dist-{i}.png')
        # return

        p = 10
        step_size = len(indexes) // p
        baseLearnerAcc = []
        baseLearnerScore_min = []
        baseLearnerScore_max = []
        for s in range(p):
            # y_true = torch.from_numpy(numpy.array([data.labels for data in valLoader.dataset.datasets]).reshape(-1)[indexes[s*step_size:(s+1)*step_size]])
            y_true = torch.from_numpy(testLoader.dataset.labels[indexes[s * step_size:(s + 1) * step_size]])
            y_pred = pseudoLabels[indexes[s * step_size:(s + 1) * step_size]]
            baseLearnerScore_min.append(min(scores[s * step_size:(s + 1) * step_size]))
            baseLearnerScore_max.append(max(scores[s * step_size:(s + 1) * step_size]))

            correct = y_pred.eq(y_true).sum().item()
            baseLearnerAcc.append(correct / step_size)

        acc.append(baseLearnerAcc)
        score_min.append(baseLearnerScore_min)
        score_max.append(baseLearnerScore_max)
        baseLearners.append(model)

    print(acc)
    acc = numpy.mean(acc, axis=0)
    score_min = numpy.mean(score_min, axis=0)
    score_max = numpy.mean(score_max, axis=0)

    names = ["%.4f - %.4f" % (score_max[i] * 100, score_min[i] * 100) for i in range(p)]
    print(acc)
    print(names)
    plt.barh(names, acc)
    plt.show()

    logging.shutdown()
    return


# def getPretrain(model, num, log=None, root='rapor/Deney3-4/tmp/2/exp2/weights/'):
def getPretrain(model, num, log=None, root='rapor/Deney5/tmp/exp2/weights/'):
    """
     load pretrained model
    :param model: model
    :param num: which model number
    :param log: log
    :param root: root
    :return: model
    """
    path = f'{root}baseLearner-{num * 2}.pth'
    checkpoint = torch.load(path, map_location=device)
    if log is not None:
        log.info(f' \t\t Pretrain Model Acc : {checkpoint["acc"]}')
    model.load_state_dict(checkpoint['net'])
    return model


def getPseudoLabels(model, dataLoader, log=None):
    """
     using model obtain pseudo labels
    :param model: model
    :param dataLoader: unlabeled data loader
    :return: pseudo labels
    """
    if log is not None:
        log.info(f' \t \t Pseudo-Labels are calculating... ')
    dataSize = len(dataLoader.dataset)
    pseudoLabels = torch.zeros((dataSize, 10))
    last = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in tqdm.tqdm(enumerate(dataLoader)):
            batchSize = inputs.size(0)
            inputs = inputs.to(device)
            outputs = F.softmax(model(inputs), dim=1)
            pseudoLabels[last:last + batchSize] = outputs.detach().cpu()
            last += batchSize
    return pseudoLabels


def getDataLoaders(modelNo, args):
    """
     take arguments and return dataset iterators
    :param modelNo: which model?
    :param args: arguments
    :return: dataset iterators
    """
    if modelNo > 10:
        modelNo = modelNo % 10

    # dataset transforms
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    train_transforms = transforms.Compose([ImageNetPolicy(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    # dataset Loaders
    # get the first 9 one's folds for training and one validation
    folds = []
    for f in range(10):
        if f == modelNo:
            train = STL10(root='Data/STL10',
                          split='train',
                          folds=f,
                          transform=train_transforms)
        else:
            folds.append(STL10(root='Data/STL10',
                               split='train',
                               folds=f,
                               transform=val_transforms))

    val_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(folds),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=0)

    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=0)

    testLoader = torch.utils.data.DataLoader(STL10(root='Data/STL10',
                                                   split='test',
                                                   transform=test_transforms),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=0)

    unlabeled_loader = torch.utils.data.DataLoader(STL10(root='Data/STL10',
                                                         split='unlabeled',
                                                         transform=test_transforms),
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=0)
    return train_loader, val_loader, unlabeled_loader, testLoader


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
