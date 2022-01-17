import os
import numpy
from pathlib import Path
import tqdm
import glob
import logging
import torch
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import matplotlib.cm as cm
from sklearn.manifold import TSNE

def newExp():
    """
        Create new experiment dir
    :return: Experiment dir path
    """
    baseDir = 'run'
    os.makedirs(baseDir, exist_ok=True)
    subDirs = os.listdir(baseDir + '/')
    last = len(subDirs)
    newDirPath = f'{baseDir}/exp{last + 1}'
    assert os.path.exists(newDirPath) == False
    os.makedirs(newDirPath)
    os.makedirs(newDirPath + '/weights')
    expName = f'Exp-{last + 1}'
    return newDirPath, expName


def visualizeLoss(lossArr, exp_path, valAccArr=None, epoch=None):
    """
        Plot Train Loss Graph and validation accuracy graph
    :param lossArr:
    :param exp_path:
    :return:
    """
    _ = plt.figure()
    plt.grid()
    plt.plot(list(range(len(lossArr))), lossArr, linewidth=.3)

    if valAccArr is not None and epoch is not None:
        maxVal = max(lossArr)
        minVal = min(lossArr)
        for i, acc in enumerate(valAccArr):
            point_y = minVal + (acc / 100) * (maxVal - minVal)
            point_x = (i + 1) * (len(lossArr) / epoch)
            plt.text(point_x, point_y, f' %{acc}', color='red', fontsize='x-small')

    plt.title('Train Loss Graph')
    plt.xlabel(' Iter ')
    plt.ylabel(' Loss ')

    num = 0
    path = Path(exp_path + f'/TrainLoss-{num}.png')
    while path.exists():
        num += 1
        path = Path(exp_path + f'/TrainLoss-{num}.png')

    plt.savefig(path)
    plt.close()
    return


def plotKappaErrorGraph(baseLearners, dataLoader, exp_path, log=None):
    """
        plot kappa-error diagram
    :param baseLearners: list of base-learners
    :param dataLoader: data loader
    :param exp_path: experiment path
    :param log: log
    """
    if log is not None:
        log.info(f' \t \t Kappa-Error Graph is calculating...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preds = []
    for learner in baseLearners:
        pred = []
        actualLabels = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataLoader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = learner(inputs)
                _, predicted = outputs.max(1)
                pred += list(predicted.detach().cpu().numpy())
                actualLabels += list(targets.detach().cpu().numpy())
        preds.append(pred)

    x = []
    y = []
    index = numpy.arange(0, len(baseLearners))
    for (idx1, idx2) in itertools.combinations(index, 2):
        pred1 = preds[idx1]
        pred2 = preds[idx2]
        kappa = cohen_kappa_score(pred1, pred2)
        error = 1 - (accuracy_score(actualLabels, pred1) + accuracy_score(actualLabels, pred2)) / 2
        x.append(kappa)
        y.append(error)

    _ = plt.figure()
    plt.grid()

    plt.scatter(x, y, s=10)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.title(' Kappa-Error Graph')
    plt.xlabel(' Kappa ')
    plt.ylabel(' Error ')

    num = 0
    path = Path(exp_path + f'/KappaErrorGraph-{num}.png')
    while path.exists():
        num += 1
        path = Path(exp_path + f'/KappaErrorGraph-{num}.png')
    plt.savefig(path)
    plt.close()

    # save points
    with open(exp_path + '/KappaErrorGraphPoints.txt', 'w') as f:
        line = f'{x[0]}'
        for p in x[1:]:
            line += f', {p}'
        line += '\n'
        f.writelines(line)

        line = f'{y[0]}'
        for p in y[1:]:
            line += f', {p}'
        line += '\n'
        f.writelines(line)
    f.close()
    return


def evalModel(model, test_loader, args, log=None, evalDataset='Test'):
    """
     evaluate model
    :param evalDataset: type of dataset
    :param model: model
    :param test_loader: test data loader
    :param args: arguments
    :param log: log for console information
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

















# def getPredictionDist(baseLearners, dataLoader, device, log=None):
#
#     log.info(' \t \t Distribution is calculating... ')
#     colors = cm.rainbow(numpy.linspace(0, 1, len(baseLearners)))
#     pred = numpy.zeros((len(baseLearners) * len(dataLoader.dataset), 10))
#     last = 0
#     for j, model in enumerate(baseLearners):
#         for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(dataLoader)):
#             batchSize = inputs.size(0)
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             pred[last:last+batchSize] = outputs.cpu().detach().numpy()
#             last += batchSize
#     Nodes_embedded = TSNE(n_components=2, learning_rate='auto', init='random', n_iter=10000).fit_transform(pred)
#
#     Nodes_embedded = Nodes_embedded.reshepe(len(baseLearners), -1, 2)
#
#     for i in range(len(baseLearners)):
#         plt.scatter(Nodes_embedded[i][:, 0], Nodes_embedded[i][:, 1], color=colors[i])
#     plt.show()


