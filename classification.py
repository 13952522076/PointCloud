"""
Re-organized codes for point cloud training
Usage:
python classification.py --use_normals --use_uniform_sample
"""
import argparse
import os
import datetime
import torch
import random
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args
from datasets import ModelNet40DataSet
from losses import PointNetLoss, CELoss
import provider

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-d', '--data_path', default='data/modelnet40_normal_resampled/', type=str)
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointNet', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--step_size', type=int, default=20, help='LR step size')
    parser.add_argument('--gamma', type=float, default=0.7, help='LR scheduler gamma lr*gamma')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals besides x,y,z')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--randseed', type=int, help='rand seed')
    return parser.parse_args()


def main():
    args = parse_args()
    # for random seed
    if args.randseed:
        print(f"Random seed is fixed to {args.randseed} （Pytorch, python, numpy） for reproducibility...")
        torch.manual_seed(args.randseed)
        random.seed(args.randseed)
        np.random.seed(args.randseed)

    else:
        print("Random seed is not fixed Pytorch, python, numpy）...")

    if args.checkpoint is None:
        time_stamp = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
        args.checkpoint = 'checkpoints/' + args.model + time_stamp
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate', 'Train-Loss', 'Train-acc', 'Valid-Loss', 'Valid-acc'])

    print('==> Preparing data..')
    train_dataset = ModelNet40DataSet(root=args.data_path, train=True, points=args.num_points,
                                      use_uniform_sample=args.use_uniform_sample,
                                      process_data=args.process_data)
    test_dataset = ModelNet40DataSet(root=args.data_path, train=False, points=args.num_points,
                                     use_uniform_sample=args.use_uniform_sample,
                                     process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=32, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=32)

    # Model
    print('==> Building model..')
    try:
        net = models.__dict__[args.model](num_classes=args.num_classes,
                                          use_normals=args.use_normals, num_points=args.num_points)
    except:
        net = models.__dict__[args.model]()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    criterion = CELoss()
    if args.model == "PointNet":
        criterion = PointNetLoss()
    net = net.to(device)
    criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9,)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    for epoch in range(start_epoch, args.epoch):
        print('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, trainDataLoader, optimizer, criterion, device)  # {"loss", "acc", "time"}
        test_out = validate(net, testDataLoader, criterion, device)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best)
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc"],
                       test_out["loss"], test_out["acc"]])
        print(f"Training loss:{train_out['loss']} acc:{train_out['acc']}% time:{train_out['time']}s) | "
              f"Testing loss:{test_out['loss']} acc:{test_out['acc']}% time:{test_out['time']}s) \n\n")
    logger.close()


    print(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    print(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    print(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    print(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    print(f"++++++++" * 5)


def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    time_cost = datetime.datetime.now()
    for batch_idx, (points, targets) in enumerate(trainloader):
        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        points, targets = points.to(device), targets.to(device).long()
        optimizer.zero_grad()
        out = net(points)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out["logits"].max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * correct / total)),
        "time": time_cost
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (points, targets) in enumerate(testloader):
            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, targets = points.to(device), targets.to(device).long()
            out = net(points)
            loss = criterion(out, targets)
            test_loss += loss.item()
            _, predicted = out["logits"].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * correct / total)),
        "time": time_cost
    }


if __name__ == '__main__':
    main()
