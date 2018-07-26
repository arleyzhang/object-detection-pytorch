import os
import pickle
import time
import numpy as np

import torch
from torch.autograd import Variable

from lib.data.voc_eval import get_output_dir, evaluate_detections
from lib.layers import DetectOut
from lib.utils import visualize_utils


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


class EvalBase(object):
    def __init__(self, data_loader, cfg):
        self.detector = DetectOut(cfg['num_classes'], 0, 200, 0.01, 0.45)
        self.data_loader = data_loader
        self.dataset = self.data_loader.dataset
        self.name = self.dataset.name
        self.cfg = cfg
        self.results = None  # dict for voc and list for coco
        self.image_sets = self.dataset.image_sets
        # self.test_set = osp.join(self.name, str(self.dataset.image_sets))

    def result_init(self):
        raise NotImplementedError

    def evaluate_stats(self, tb_writer):
        return None

    def post_proc(self, det, img_idx, id):
        raise NotImplementedError

    # @profile
    def validate(self, net, priors, use_cuda=True, tb_writer=None):
        print('start vali')
        img_idx = 0
        _t = {'im_detect': Timer(), 'misc': Timer()}
        _t['misc'].tic()
        self.result_init()
        for batch_idx, (images, targets, extra) in enumerate(self.data_loader):
            print('im', img_idx)
            if use_cuda:
                images = Variable(images.cuda(), volatile=True)
                extra = extra.cuda()
            else:
                images = Variable(images, volatile=True)

            _t['im_detect'].tic()
            loc, conf = net(images, phase='eval')
            # image, cls, #box, [score, xmin, ymin, xmax, ymax]
            detections = self.detector(loc, conf, priors)
            _t['im_detect'].toc(average=False)

            det = detections.data
            h = extra[:, 0].unsqueeze(-1).unsqueeze(-1)
            w = extra[:, 1].unsqueeze(-1).unsqueeze(-1)
            det[:, :, :, 1] *= w  # xmin
            det[:, :, :, 3] *= w  # xmax
            det[:, :, :, 2] *= h  # ymin
            det[:, :, :, 4] *= h  # ymax

            # append image id and class to the detection results
            id = torch.arange(0, det.shape[0]).unsqueeze(-1).expand(list(det.shape[:2])) \
                .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
            cls = torch.arange(0, det.shape[1]).expand(list(det.shape[:2])).unsqueeze(-1) \
                .expand(list(det.shape[:3])).unsqueeze(-1)
            if self.name == 'MS COCO':
                coco_id = torch.Tensor(self.dataset.ids[img_idx: img_idx + det.shape[0]])
                coco_id = coco_id.unsqueeze(-1).expand(list(det.shape[:2])) \
                    .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
                det = torch.cat((det, id, cls, coco_id), 3)
            elif self.name == 'VOC0712':
                det = torch.cat((det, id, cls), 3)
            else:
                raise Exception("wrong dataset name")
            mymask = det[:, :, :, 0].gt(0.).unsqueeze(-1).expand(det.size())

            if self.name == 'MS COCO':
                det = torch.masked_select(det, mymask).view(-1, 8)
                # xmin, ymin, xmax, ymax, score, image, cls, cocoid
                det = det[:, [1, 2, 3, 4, 0, 5, 6, 7]]
                # det[:, [4]] += 1
            elif self.name == 'VOC0712':
                det = torch.masked_select(det, mymask).view(-1, 7)
                # xmin, ymin, xmax, ymax, score, image, cls
                det = det[:, [1, 2, 3, 4, 0, 5, 6]]
            if tb_writer is not None:
                self.visualize_box(images, targets, h, w, det, img_idx, tb_writer)

            img_idx = self.post_proc(det, img_idx, id)

        _t['misc'].toc(average=False)
        print(_t['im_detect'].total_time, _t['misc'].total_time)
        self.evaluate_stats(tb_writer)

    def visualize_box(self, images, targets, h, w, det, img_idx, tb_writer):
        det_ = det.cpu().numpy()
        # det_ = det_[det_[:, 4] > 0.5]
        images = images.permute(0, 2, 3, 1)
        images = images.data.cpu().numpy()
        for idx in range(len(images)):
            img = images[idx].copy()
            img = img[:, :, (2, 1, 0)]
            img += np.array((104., 117., 123.), dtype=np.float32)

            det__ = det_[det_[:, 5] == idx]
            w_ = w[idx, :].cpu().numpy()
            h_ = h[idx, :].cpu().numpy()
            w_r = 1000  # resize to 1000, h
            h_r = w_r / w_ * h_

            det__[:, 0:4:2] = det__[:, 0:4:2] / w_ * w_r
            det__[:, 1:4:2] = det__[:, 1:4:2] / h_ * h_r

            t = targets[idx].numpy()
            t[:, 0:4:2] = t[:, 0:4:2] * w_r
            t[:, 1:4:2] = t[:, 1:4:2] * h_r
            t[:, 4] += 1  # TODO because of the traget transformer

            boxes = {'gt': t, 'pred': det__}
            tb_writer.cfg['steps'] = img_idx + idx
            if self.name == 'MS COCO':
                tb_writer.cfg['img_id'] = int(det__[0, 7]) if det__.size != 0 else 'no_detect'
            if self.name == 'VOC0712':
                tb_writer.cfg['img_id'] = int(det__[0, 5]) if det__.size != 0 else 'no_detect'
            tb_writer.cfg['thresh'] = 0.5
            visualize_utils.vis_img_box(img, boxes, (h_r, w_r), tb_writer)


class EvalVOC(EvalBase):
    def __init__(self, data_loader, cfg):
        super(EvalVOC, self).__init__(data_loader, cfg)
        self.test_set = self.image_sets[0][1]

    def result_init(self):
        self.results = [[[] for _ in range(len(self.dataset))] for _ in range(self.cfg['num_classes'])]

    def post_proc(self, det, img_idx, id):
        # manually broadcast
        det = det.cpu().numpy()
        # det_tensors.append(det)
        for b_idx in range(id.shape[0]):
            det_ = det[det[:, 5] == b_idx]
            for cls_idx in range(1, id.shape[1]):  # skip bg class
                det__ = det_[det_[:, 6] == cls_idx]
                if det__.size == 0:
                    continue
                self.results[cls_idx][img_idx] = det__[:, 0:5].astype(np.float32, copy=False)
            img_idx += 1
        return img_idx

    def evaluate_stats(self, tb_writer):
        output_dir = get_output_dir('ssd300_120000', self.test_set)
        det_file = os.path.join(output_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(self.results, f, pickle.HIGHEST_PROTOCOL)
        print('Evaluating detections')
        res = evaluate_detections(self.results, output_dir, self.data_loader.dataset, test_set=self.test_set)
        if tb_writer is not None:
            visualize_utils.viz_pr_curve(res, tb_writer)


class EvalCOCO(EvalBase):
    def __init__(self, data_loader, cfg):
        super(EvalCOCO, self).__init__(data_loader, cfg)

    def result_init(self):
        self.results = []

    # @profile
    def post_proc(self, det, img_idx, id):
        det[:, 2] -= det[:, 0]  # w
        det[:, 3] -= det[:, 1]  # h
        det = det[:, [7, 0, 1, 2, 3, 4, 6]]
        det_ = det.cpu().numpy()
        det__ = det_[det_[:, 5] > 0.5]
        self.results.append(det__)
        img_idx += id.shape[0]
        return img_idx

    def evaluate_stats(self, tb_writer):
        from pycocotools.cocoeval import COCOeval
        res = np.vstack(self.results)
        coco = self.dataset.cocos[0]['coco']
        coco_pred = coco.loadRes(res)
        cocoEval = COCOeval(coco, coco_pred, 'bbox')
        # cocoEval.params.imgIds = self.dataset.ids
        # cocoEval.params.catIds = [24, 7, 5, 6]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        # gts=coco.loadAnns(coco.getAnnIds(imgIds=self.dataset.ids,
        #                                  catIds=coco.getCatIds()))
        # dts=coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=self.dataset.ids,
        #                                            catIds=coco_pred.getCatIds()))
