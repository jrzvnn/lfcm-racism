import glob
import os
import time 
import torch


def train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_racist = AverageMeter()
    acc_notRacist = AverageMeter()
    acc_avg = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, image_text, tweet, comment, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(gpu, non_blocking=True)
        image_var = torch.autograd.Variable(image)
        image_text_var = torch.autograd.Variable(image_text)
        tweet_var = torch.autograd.Variable(tweet)
        comment_var = torch.autograd.Variable(comment)
        target_var = torch.autograd.Variable(target).squeeze(1)

        # compute output
        output = model(image_var, image_text_var, tweet_var, comment_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        prec1 = accuracy(output.data, target.long().cuda(gpu), topk=(1,))
        cur_acc_racist, cur_acc_notRacist = accuracy_per_class(output.data, target.long().cuda(gpu))
        acc_racist.update(cur_acc_racist, image.size()[0])
        acc_notRacist.update(cur_acc_notRacist, image.size()[0])
        acc_avg.update((cur_acc_racist + cur_acc_notRacist) / 2, image.size()[0])
        # print image.size()[0]
        losses.update(loss.data.item(), image.size()[0])
        acc.update(prec1[0], image.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'GPU: {gpu}\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Acc {acc.val.data[0]:.3f} ({acc.avg.data[0]:.3f})\t'
        #           'Acc Hate {acc_racist.val:.3f} ({acc_racist.avg:.3f})\t'
        #           'Acc NotHate {acc_notRacist.val:.3f} ({acc_notRacist.avg:.3f})\t'
        #           'Acc Avg {acc_avg.val:.3f} ({acc_avg.avg:.3f})\t'.format(
        #            epoch, i, len(train_loader), gpu=str(gpu), batch_time=batch_time,
        #            data_time=data_time, loss=losses, acc=acc, acc_racist=acc_racist, acc_notRacist=acc_notRacist, acc_avg=acc_avg))

    print('TRAIN:: Acc: ' + str(acc.avg.data[0].item()) + 'Acc Avg: ' + str(acc_avg.avg) + ' Racist Acc: ' + str(acc_racist.avg) + ' - Not Racist Acc: ' + str(
        acc_notRacist.avg))

    # disable temporarily
    plot_data['train_loss'][plot_data['epoch']] = losses.avg
    plot_data['train_acc'][plot_data['epoch']] = acc.avg
    plot_data['train_acc_racist'][plot_data['epoch']] = acc_racist.avg
    plot_data['train_acc_notRacist'][plot_data['epoch']] = acc_notRacist.avg
    plot_data['train_acc_avg'][plot_data['epoch']] = acc_avg.avg


    return plot_data


def validate(val_loader, model, criterion, print_freq, plot_data, gpu):
    with torch.no_grad():

        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        acc_racist = AverageMeter()
        acc_notRacist = AverageMeter()
        acc_avg = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (image, image_text, tweet, comment, target) in enumerate(val_loader):

            target = target.cuda(gpu, non_blocking=True)
            image_var = torch.autograd.Variable(image)
            image_text_var = torch.autograd.Variable(image_text)
            tweet_var = torch.autograd.Variable(tweet)
            comment_var = torch.autograd.Variable(comment)
            target_var = torch.autograd.Variable(target).squeeze(1)


            # compute output
            output = model(image_var, image_text_var, tweet_var, comment_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            prec1 = accuracy(output.data, target.long().cuda(gpu), topk=(1,))
            cur_acc_racist, cur_acc_notRacist = accuracy_per_class(output.data, target.long().cuda(gpu))
            acc_racist.update(cur_acc_racist, image.size()[0])
            acc_notRacist.update(cur_acc_notRacist, image.size()[0])
            acc_avg.update((cur_acc_racist + cur_acc_notRacist) / 2, image.size()[0])

            losses.update(loss.data.item(), image.size()[0])
            acc.update(prec1[0], image.size()[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc {acc.val.data[0]:.3f} ({acc.avg.data[0]:.3f})'
            #           'Acc Hate {acc_racist.val:.3f} ({acc_racist.avg:.3f})\t'
            #           'Acc NotHate {acc_notRacist.val:.3f} ({acc_notRacist.avg:.3f})\t'
            #           'Acc Avg {acc_avg.val:.3f} ({acc_avg.avg:.3f})\t'.format(
            #            i, len(val_loader), batch_time=batch_time, loss=losses,
            #            acc=acc, acc_racist=acc_racist, acc_notRacist=acc_notRacist, acc_avg=acc_avg))

        print('VALIDATION: Acc: ' + str(acc.avg.data[0].item()) + 'Acc Avg: ' + str(acc_avg.avg) + ' Racist Acc: ' + str(acc_racist.avg) + ' - Not Racist Acc: ' + str(acc_notRacist.avg ))

        plot_data['val_loss'][plot_data['epoch']] = losses.avg
        plot_data['val_acc'][plot_data['epoch']] = acc.avg
        plot_data['val_acc_racist'][plot_data['epoch']] = acc_racist.avg
        plot_data['val_acc_notRacist'][plot_data['epoch']] = acc_notRacist.avg
        plot_data['val_acc_avg'][plot_data['epoch']] = acc_avg.avg


    return plot_data


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_per_class(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()

    correct_racist = 0
    correct_notRacist = 0
    total_racist = 0
    total_notRacist = 0
    pred = pred[0]

    for i, cur_target in enumerate(target):

        if cur_target == 1:
            total_racist += 1
            if cur_target == pred[i]: correct_racist += 1
        else:
            total_notRacist += 1
            if cur_target == pred[i]: correct_notRacist += 1

    if total_racist == 0 : total_racist = 1
    if total_notRacist == 0 : total_notRacist = 1

    acc_racist = 100 * float(correct_racist) / total_racist
    acc_notRacist = 100 * float(correct_notRacist) / total_notRacist


    return acc_racist, acc_notRacist


def save_checkpoint(model, is_best, filename='checkpoint.pth.tar'):
    print("Saving Checkpoint")
    # prefix = 16
    # if '_ValLoss_' in filename:
    #     prefix = 30
    # for cur_filename in glob.glob(filename[:-prefix] + '*'):
    #     print(cur_filename)
    #     os.remove(cur_filename)
    torch.save(model.state_dict(), filename + '.pth')
    # if is_best:
    #     shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')