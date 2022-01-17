import argparse
import logging
import tqdm
import os
from pathlib import Path

import torch
from torch import nn

from utills import visualizeLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def trainBaseLearner(model, trainLoader, valLoader, expPath, expName, args):
    log = logging.getLogger(expName)
    trainDataLen = len(trainLoader.dataset)

    # get optimizer
    log.info(f' \t       Optimizer : {args.optim_BL}')
    log.info(f' \t       Scheduler : {args.scheduler_BL}')
    log.info(f' \t   Learning Rate : {args.lr_BL}')
    log.info(f' \t       Criterion : {args.criterion_BL}')
    log.info(f' \t Label Smoothing : {args.label_smoothing}')
    log.info(f' \t Train Data Size : {trainDataLen}')
    log.info(f' \t      Batch Size : {args.batch_size}')

    if args.optim_BL == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_BL,
                                    momentum=0.9,
                                    weight_decay=1e-4)
    elif args.optim_BL == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_BL)

    # criterion
    if args.criterion_BL == 'CEL':
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # schedular
    if args.scheduler_BL == 'MSLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12])
    elif args.scheduler_BL == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_BL, max_lr=0.001,
                                                      step_size_up=trainDataLen // args.batch_size,
                                                      mode='triangular2', cycle_momentum=False)

    # train
    train_losses = []
    val_acc = []
    best_acc = 0
    for epoch in range(1, args.epoch_BL + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainLoader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if train_loss is None:
                exit(0)

            if args.scheduler_BL == 'CyclicLR':
                scheduler.step()

        # train_losses.append(train_loss)
        log.info(
            ' \t Epoch: %d Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            epoch, train_loss, 100. * correct / total, correct, total))

        if args.scheduler_BL == 'MSLR':
            scheduler.step()
        best_acc, acc = val(model, valLoader, criterion, expPath, best_acc, log)
        val_acc.append(acc)

    model = saveLastBestModel(model, expPath, log)
    visualizeLoss(train_losses, expPath, valAccArr=val_acc, epoch=args.epoch_BL)
    return model


def val(model, test_loader, criterion, exp_path, best_acc, log=None):
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
        log.info(f' \t \t  Best Test Acc: {best_acc} Current Test Acc : {acc} ')

    if acc > best_acc:
        if log is not None:
            log.info(' \t \t Saving..')

        with open(exp_path + '/BaseLearnerModel.txt', 'w') as f:
            f.writelines(model.__str__())
        f.close()
        state = {
            'net': model.state_dict(),
            'acc': acc,
        }
        torch.save(state, exp_path + '/weights/best-baseLearner.pth')
        best_acc = acc
    return best_acc, acc


def saveLastBestModel(model, exp_path, log=None):
    """
    change best base learner checkpoint file to new base learner checkpoint file
    :param model: model
    :param exp_path: experiment path
    :param log: log
    :return: best model
    """
    exist = exp_path + '/weights/best-baseLearner.pth'
    checkpoint = torch.load(exist, map_location=device)
    if log is not None:
        log.info(f' \t \t BEST MODEL ACC: {checkpoint["acc"]}')

    model.load_state_dict(checkpoint['net'])

    num = 0
    newPath = Path(exp_path + f'/weights/baseLearner-{num}.pth')
    while newPath.exists():
        num += 1
        newPath = Path(exp_path + f'/weights/baseLearner-{num}.pth')
    os.rename(exist, newPath)
    return model


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='STL10', help='dataset')
    parser.add_argument('--base-learner', type=str, default='BasicCNN', help='base learner')
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')

    parser.add_argument('--scheduler', type=str, default='MSLR', help='')
    parser.add_argument('--criterion', type=str, default='CEL', help='learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='learning rate')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_opt()
    trainBaseLearner(args)
