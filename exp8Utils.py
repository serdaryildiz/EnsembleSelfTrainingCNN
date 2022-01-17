import torch
from torchvision import transforms
from torchvision.datasets import STL10

from AutoAugment.autoaugment import ImageNetPolicy

import tqdm
import copy
import numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    train_transforms = transforms.Compose([transforms.Resize(224),
                                          ImageNetPolicy(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
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
            outputs = model(inputs)
            pseudoLabels[last:last + batchSize] = outputs.detach().cpu()
            last += batchSize
    return pseudoLabels


def getTrainDataForBaseLearners(pseudoLabels, modelNo, args, expPath, log=None):
    """
     obtain train datasets for base learners
    :param modelNo: which model?
    :param pseudoLabels: pseudo-labels
    :param args: arguments
    :param log: log
    :return: list of data loaders
    """
    if modelNo > 10:
        modelNo = modelNo % 10

    pseudoScores, pseudoLabels = pseudoLabels.max(1)
    train_transforms = transforms.Compose([transforms.Resize(224),
                                          ImageNetPolicy(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    unlabeledData = STL10(root='Data/STL10',
                          split='unlabeled',
                          transform=train_transforms)
    unlabeledDataLen = len(pseudoLabels)
    assert len(unlabeledData) == unlabeledDataLen

    unlabeledData.labels = pseudoLabels.detach().cpu().numpy()
    # scores, indexes = torch.sort(pseudoScores)
    # th = int(unlabeledDataLen * 0.2)
    # # indexes = indexes[th:-th]
    # # indexes = indexes[:-th]
    # indexes = indexes[th:]
    # unlabeledData.data = unlabeledData.data[indexes]
    # unlabeledData.labels = unlabeledData.labels[indexes]

    # unlabeledDataLen = len(unlabeledData)

    # with open(f'{expPath}/indexes-{modelNo}.txt', 'w') as f:
    #     line = f'{indexes[0]}'
    #     for i in indexes[1:]:
    #         line += f', {i}'
    #     line += '\n'
    #     f.writelines(line)
    # f.close()


    folds = [STL10(root='Data/STL10',
                   split='train',
                   folds=modelNo,
                   transform=train_transforms)]

    folds = torch.utils.data.ConcatDataset(folds)

    ratio = unlabeledDataLen // len(folds)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([folds] * ratio + [unlabeledData]),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)
    if log is not None:
        log.info(f' \t \t Train Data with Pseudo-Labeled Data Size : {len(train_loader.dataset)}')
    return train_loader


def getPretrain(model, num, log=None, root='rapor/Deney5/EXP/exp2/weights/'):
    """
     load pretrained model
    :param model: model
    :param num: which model number
    :param log: log
    :param root: root
    :return: model
    """
    path = f'{root}baseLearner-{num*2}.pth'
    checkpoint = torch.load(path, map_location=device)
    if log is not None:
        log.info(f' \t\t Pretrain Model Acc : {checkpoint["acc"]}')
    model.load_state_dict(checkpoint['net'])
    return model
