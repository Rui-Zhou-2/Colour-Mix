import random
import time
import warnings
import sys
import argparse
from PIL import Image
import numpy as np
import os
import math
import shutil
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.append('..')
import common.vision.models.segmentation as models
import common.vision.datasets.segmentation as datasets
import common.vision.transforms.segmentation as T
from common.vision.transforms import DeNormalizeAndTranspose
from common.utils.data import ForeverDataIterator
from common.utils.metric import ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter, Meter
from common.utils.logger import CompleteLogger
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from self_training.mean_teacher import EMATeacher2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(label)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        chosen_classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, chosen_classes).unsqueeze(0))
    return class_masks

def generate_class_mask(label, classes):
    
    label, classes = torch.broadcast_tensors(label, classes.unsqueeze(1).unsqueeze(2))
    # print(label.shape) #[1, 540, 810])
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask

def one_mix(mask, data=None, target=None):
    # print(mask.shape)

    if data is not None:
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] + (1 - stackedMask0) * data[1])
    if target is not None:
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] + (1 - stackedMask0) * target[1])
    # print(data.shape)torch.Size([1, 3, 540, 810])
    # print(target.shape)torch.Size([1, 1, 540, 810])
    return data, target

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    target_dataset = datasets.__dict__[args.target]
    train_target_dataset = target_dataset(
        root=args.target_root,
        transforms=T.Compose([
            T.Resize(image_size=args.train_size),
            # T.RandomResizedCrop(size=args.train_size, ratio=args.resize_ratio, scale=(0.5, 1.)),
            T.NormalizeAndTranspose(),
        ]),
    )
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_target_dataset = target_dataset(
        root=args.target_root, split='val',
        transforms=T.Compose([
            # T.RandomResizedCrop(size=args.test_input_size, ratio=args.resize_ratio, scale=(0.5, 1.)),
            T.Resize(image_size=args.test_input_size, label_size=args.test_output_size),
            T.NormalizeAndTranspose(),
        ])
    )
    val_target_loader = DataLoader(val_target_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # collect the absolute paths of all images in the target dataset
    target_image_list = train_target_dataset.collect_image_paths()

    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(
        root=args.source_root,
        transforms=T.Compose([
            T.Resize(image_size=args.train_size),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomHorizontalFlip(),
            T.NormalizeAndTranspose(),
        ]),
    )
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)


    model = models.__dict__[args.arch](num_classes=2).to(device)
    t_model = EMATeacher2(model, alpha=args.alpha)

    optimizer = SGD(model.get_parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. - float(x) / args.epochs / args.iters_per_epoch)
                                                 ** (args.lr_power))

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        t_model.load_state_dict(checkpoint['t_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        
    weight = torch.Tensor([1,20])
    criterion = torch.nn.CrossEntropyLoss(weight= weight, ignore_index=args.ignore_label).to(device)
    interp_train = lambda x: F.interpolate(x, size=args.train_size[::-1], mode='bilinear', align_corners=True)
    interp_val = lambda x: F.interpolate(x, size=args.test_output_size[::-1], mode='bilinear', align_corners=True)
    # define visualization function
    decode = train_source_dataset.decode_target
    def visualize(image, pred, label, prefix):

        image = image.detach().cpu().numpy()
        pred = pred.detach().max(dim=0)[1].cpu().numpy()
        label = label.cpu().numpy()
        for tensor, name in [
            (Image.fromarray(np.uint8(DeNormalizeAndTranspose()(image))), "image"),
            (decode(label), "label"),
            (decode(pred), "pred")
        ]:
            tensor.save(logger.get_image_path("{}_{}.png".format(prefix, name)))

    if args.phase == 'test':
        confmat = validate(val_target_loader, model, interp_val, criterion, visualize, args)
        print(confmat)
        return

    # start training
    best_iou = 0.
    for epoch in range(args.start_epoch, args.epochs):

        
        logger.set_epoch(epoch)
        print(lr_scheduler.get_lr())

        # train for one epoch
        train(train_source_iter, train_target_iter, model, t_model,interp_train, criterion, optimizer,
              lr_scheduler, epoch, visualize if args.debug else None, args)

        # evaluate on validation set
        confmat = validate(val_target_loader, t_model, interp_val, criterion, visualize if args.debug else None, args)
        print(confmat.format(train_source_dataset.classes))
        acc_global, acc, iu = confmat.compute()

        # calculate the mean iou over partial classes
        indexes = [train_source_dataset.classes.index(name) for name
                   in train_source_dataset.evaluate_classes]
        iu = iu[indexes]
        mean_iou = iu.mean()

        # remember best acc@1 and save checkpoint
        torch.save(
            {
                'model': model.state_dict(),
                't_model': t_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }, logger.get_checkpoint_path(epoch)
        )
        if mean_iou > best_iou:
            shutil.copy(logger.get_checkpoint_path(epoch), logger.get_checkpoint_path('best'))
        best_iou = max(best_iou, mean_iou)
        print("Target: {} Best: {}".format(mean_iou, best_iou))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model,t_model, interp, criterion,optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, visualize, args: argparse.Namespace):

    losses_s = AverageMeter('Loss (s)', ':3.2f')

    losses_m = AverageMeter('Loss (m)', ':3.2f')
    accuracies_s = Meter('Acc (s)', ':3.2f')
    accuracies_t = Meter('Acc (t)', ':3.2f')

    iou_s = Meter('IoU (s)', ':3.2f')
    iou_t = Meter('IoU (t)', ':3.2f')


    confmat_s = ConfusionMatrix(2)#model.num_classes
    confmat_t = ConfusionMatrix(2)

    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses_s,losses_m,
         accuracies_s, accuracies_t,iou_s,  iou_t],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    t_model.train()       

    for i in range(args.iters_per_epoch):

        optimizer.zero_grad()

        x_s, label_s = next(train_source_iter)
        x_t, label_t = next(train_target_iter)

        x_s = x_s.to(device)
        label_s = label_s.long().to(device)
        x_t = x_t.to(device)
        label_t = label_t.long().to(device)

        # compute output
        y_s = model(x_s)
        pred_s = interp(y_s)
        del y_s
        loss_cls_s = criterion(pred_s, label_s)

        loss_cls_s.backward(retain_graph=True)

        y_t = t_model(x_t)
         
        pred_t = interp(y_t)
        ema_softmax = torch.softmax(pred_t.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(0.9).long() == 1
        new_pseudo_label = pseudo_label * ps_large_p

        del pseudo_prob, pseudo_label, ema_softmax
        mixed_x_t, mixed_label = [None] * args.batch_size, [None] * args.batch_size
        mix_masks = get_class_masks(label_s)
        

        for j in range(args.batch_size):
            # Clone the labels for manipulation
            label_s_c = label_s[j].clone()
            label_t_c = new_pseudo_label[j].clone()

            mixed_x_t[j], mixed_label[j] = one_mix(mix_masks[j], data=[x_s[j], x_t[j]], target=[label_s_c, label_t_c])


        mixed_label = torch.stack(mixed_label)
        mixed_label = mixed_label.long()
        mixed_x_t = mixed_x_t.squeeze(1)
        mixed_label = mixed_label.squeeze(1)
        y_m = model(mixed_x_t)
        pred_mix = interp(y_m)
        del y_m

        loss_m = criterion(pred_mix, mixed_label)
        loss_m.backward()

        optimizer.step()
        lr_scheduler.step()

        # measure accuracy and record loss
        losses_s.update(loss_cls_s.item(), x_s.size(0))

        losses_m.update(loss_m.item(), x_s.size(0))

        confmat_s.update(label_s.flatten(), pred_s.argmax(1).flatten())
        confmat_t.update(label_t.flatten(), pred_t.argmax(1).flatten())

        acc_global_s, acc_s, iu_s = confmat_s.compute()
        acc_global_t, acc_t, iu_t = confmat_t.compute()
        # acc_global_m, acc_m, iu_m = confmat_m.compute()
        accuracies_s.update(acc_s.mean().item())
        accuracies_t.update(acc_t.mean().item())
        # accuracies_m.update(acc_m.mean().item())
        iou_s.update(iu_s.mean().item())
        iou_t.update(iu_t.mean().item())

        global_step = epoch * args.iters_per_epoch + i + 1
  
        alpha = min(args.alpha, 1 - 1 / global_step)
        
        t_model.set_alpha(alpha)

        t_model.update()

        if i % 200 == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model, interp, criterion, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = Meter('Acc', ':3.2f')
    iou = Meter('IoU', ':3.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc, iou],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    confmat = ConfusionMatrix(2)#model.num_classes

    with torch.no_grad():
        # end = time.time()
        for i, (x, label) in enumerate(val_loader):
            x = x.to(device)
            label = label.long().to(device)

            # compute output
            output = interp(model(x))
            loss = criterion(output, label)

            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))
            confmat.update(label.flatten(), output.argmax(1).flatten())
            acc_global, accs, iu = confmat.compute()
            acc.update(accs.mean().item())
            iou.update(iu.mean().item())

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            if i % 100 == 0:

                progress.display(i)
                # if visualize is not None:
                #     visualize(x[0], output[0], label[0], "val_{}".format(i))

    return confmat


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='FDA for Segmentation Domain Adaptation')
    # dataset parameters
    parser.add_argument('source_root', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--resize-ratio', nargs='+', type=float, default=(1.5, 8 / 3.),
                        help='the resize ratio for the random resize crop')
    parser.add_argument('--train-size', nargs='+', type=int, default=(810, 540),
                        help='the input and output image size during training')
    parser.add_argument('--test-input-size', nargs='+', type=int, default=(810, 540),
                        help='the input image size during test')
    parser.add_argument('--test-output-size', nargs='+', type=int, default=(810, 540),
                        help='the output image size during test')
    # model parameters
    parser.add_argument('--alpha', default=0.9, type=float,
                        help='ema decay factor')    
    parser.add_argument('-a', '--arch', metavar='ARCH', default='deeplabv2_resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: deeplabv2_resnet101)')
    parser.add_argument("--entropy-weight", type=float, default=0., help="weight for entropy")
    parser.add_argument("--ita", type=float, default=2.0, help="ita for robust entropy")
    parser.add_argument("--beta", type=int, default=1, help="beta for FDA")
    parser.add_argument("--resume", type=str, default=None,#'./rgb/dacs_ut/ckp/best.pth'
                        help="Where restore model parameters from.")
    # training parameters
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N',
                        help='mini-batch size (default: 2)')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')

    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--lr-power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate (only for deeplab).")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warm-up-epochs', default=1, type=int,
                        help='number of epochs to warm up (default: 10)')    
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=18, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--ignore-label", type=int, default=3,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--log", type=str, default='fda',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--debug', action="store_true",
                        help='In the debug mode, save images and predictions during training')
    args = parser.parse_args()
    main(args)