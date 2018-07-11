import os
import torch
import numpy as np
import pickle
import time

from torch.autograd import Variable
from layers import *

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def voc_ap(rec, prec, use_07_metric=True):  ###???
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:   #############invalid value encountered in greater_equal
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(gt_class_recs, pred_class_recs, npos, ovthresh=0.5, use_07_metric=True):
    image_ids = []
    confidence = []
    BB = []

    for im_idx in range(len(pred_class_recs)):
        dets = pred_class_recs[im_idx]
        if dets == []: continue
        for k in range(dets.shape[0]):
            image_ids.append(im_idx)
            confidence.append(dets[k, -1])
            BB.append([dets[k, 0] + 1, dets[k, 1] + 1,
                        dets[k, 2] + 1, dets[k, 3] + 1])
        # format: box_coord, conf
    if len(image_ids) == 0:
        return -1, -1, -1
    
    confidence = np.array(confidence)    #conf
    BB = np.array(BB)  #bbox

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd): #
        R = gt_class_recs[image_ids[d]]    #gt with this classname in this image
        bb = BB[d, :].astype(float)     #predict bbox
        ovmax = -np.inf #init min_val
        BBGT = R['bbox'].astype(float)  #gt box
        if BBGT.size > 0:   #if exist bbox
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih    #a^b
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                    (BBGT[:, 2] - BBGT[:, 0]) *
                    (BBGT[:, 3] - BBGT[:, 1]) - inters)  #a v b
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:    #difficult bbox not in calc
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)  #cumulative sum
    tp = np.cumsum(tp)
    rec = tp / float(npos)  #recall             #invalid value encountered in true_divide
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)   #precision
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def parse_rec(targets): #all objs in a img
    objects = []
    for obj in targets:
        obj_struct = {}
        obj_struct['cls_id'] = int(obj[-1])  #start from 0
        obj_struct['difficult'] = 0 #in voc0712.py, have filtered difficult=1
        obj_struct['bbox'] = [int(obj[0]), int(obj[1]),
                              int(obj[2]), int(obj[3])]
        objects.append(obj_struct)

    return objects

class EvalSolver(object):
    def __init__(self, data_loader, test_size,cachedir, cfg):
        self.detector = Detect(cfg['num_classes'], 0, 200, 0.01, 0.45)
        self.data_loader = data_loader
        self.test_size = test_size
        self.cfg = cfg

        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.num_priors = self.priors.size(0)

        self.cachefile = os.path.join(cachedir, '{}_annots.pkl'.format(cfg['dataset_name']))
        self.gt_recs, is_readed = self.read_cachefile(cachedir, self.cachefile)
        self.is_readed = is_readed
    
    def read_cachefile(self, cachedir, cachefile):
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        #cachefile = os.path.join(cachedir, file_name)
        #record gt box
        if os.path.isfile(cachefile):
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)
            return recs, True
        else:
            return {}, False

    def validate(self, net, criterion=None, use_cuda=True):

        all_boxes = [[[] for _ in range(self.test_size)]
                 for _ in range(self.cfg['num_classes'])]

        conf_t = ()#torch.FloatTensor(self.test_size, self.num_priors, self.cfg['num_classes'])
        loc_t = ()#torch.FloatTensor(self.test_size, self.num_priors, 4)

        #print('all_boxes----', len(self.data_loader)) #10????
        img_idx = 0
        # timers = {'pred_time': Timer(), 'detect_time': Timer(),
        #         'allbox_time': Timer(), 'ap_time': Timer()}
        for (images, targets, heights, widths, _, _) in self.data_loader:
            if use_cuda:
                images = Variable(images.cuda())
                targets = [anno.numpy() for anno in targets]  #Variable(anno.cuda(), volatile=True) 
                self.priors.cuda()
            else:
                images = Variable(images)
                targets = [anno.numpy() for anno in targets]
            
            #timers['pred_time'].tic()
            """
            loc, conf is Variable    "bug priors" torch.Size([17464, 4])
            #torch.Size([32, 8732, 4]) torch.Size([32, 8732, 21])
            """
            loc, conf= net(images, phase='eval')
            #timers['pred_time'].toc()

            # loss
            #loss_l, loss_c = criterion(out, targets)

            #timers['detect_time'].tic()
            #Variable  .data  convert to Tensor
            detections = self.detector(loc, conf, self.priors)  #run on a gpu
            detections = detections.data    #Shape(8,21,200,5)
            #timers['detect_time'].toc()
            
            #print('debug detections----', detections.size(), images.size())
            #print("debug p d time1: %.4f %.4f" % (timers['pred_time'].diff, timers['detect_time'].diff))

            #timers['allbox_time'].tic()
            for batch_idx in range(images.size(0)): #imgs in batch
                for cls_idx in range(1, detections.size(1)):  #skip bg class
                    dets = detections[batch_idx, cls_idx, :]
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.dim() == 0:
                        continue
                    boxes = dets[:, 1:]
                    #print('debug boxes----', boxes)
                    boxes[:, 0] *= widths[batch_idx]
                    boxes[:, 2] *= widths[batch_idx]
                    boxes[:, 1] *= heights[batch_idx]
                    boxes[:, 3] *= heights[batch_idx]
                    scores = dets[:, 0].cpu().numpy()
                    cls_dets = np.hstack((boxes.cpu().numpy(),
                                        scores[:, np.newaxis])).astype(np.float32,
                                                                        copy=False)
                    all_boxes[cls_idx][img_idx] = cls_dets  #is ok
                #record gt box
                if not self.is_readed:
                    self.gt_recs[img_idx] = parse_rec(targets[batch_idx])
                img_idx += 1    #img index
            #timers['allbox_time'].toc()
            # print("debug time2: ave=%.4f ave=%.4f %.4f" % (timers['pred_time'].average_time, timers['detect_time'].average_time, 
            #         timers['allbox_time'].diff))
        
        #timers['ap_time'].tic()
        aps = []
        for cls_idx in range(1, self.cfg['num_classes']):
            gt_class_recs = {}
            npos = 0    # indifficult num_obj in a img
            for imagename in range(self.test_size):
                R = [obj for obj in self.gt_recs[imagename] if obj['cls_id'] == cls_idx - 1]
                bbox = np.array([x['bbox'] for x in R])
                difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
                det = [False] * len(R)  #detect flag,0:undetected 1:detected
                npos = npos + sum(~difficult)   #only think difficult=0
                gt_class_recs[imagename] = {'bbox': bbox,
                                        'difficult': difficult,
                                        'det': det}
            rec, prec, ap = voc_eval(gt_class_recs, all_boxes[cls_idx], npos)
            aps += [ap]
            #print('eval class {}: {}'.format(cls_idx - 1, ap))
        #timers['ap_time'].toc()
        #print("debug mAP time3: %.4f" % (timers['ap_time'].diff))
        
        # save gt cachefile
        if (not self.is_readed) and len(self.gt_recs) != 0:
            with open(self.cachefile, 'wb') as f:
                pickle.dump(self.gt_recs, f)
            self.is_readed = True
        
        return np.mean(aps)
