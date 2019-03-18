# -*- coding: UTF-8 -*-
import os
import os.path as osp
import sys
import pickle
import numpy as np
import mmcv

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '.'))
sys.path.insert(0, osp.join(cur_dir, '..'))
sys.path.insert(0, osp.join(cur_dir, '../..'))


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    ap11_cls = []  # this is where changed
    if use_07_metric:
        # 11 point metric
        # average precision when recall is 0.0,0.1,0.2...0.9，1.0 (11 points)
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
            ap11_cls.append(p)
    #  if use_07_metric is false, use area mode
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        ap11_cls = np.zeros(11)
    return ap, ap11_cls


def parse_tencent_youtu_cache(annopath):
    ann = mmcv.load(annopath)
    recs = {}
    for image_path in ann.keys():
        image_name = '/'.join(image_path.split('/')[-2:])
        instances = ann[image_path]
        objects = []
        for ins in instances:
            cls_name = ins.class_label
            x1 = ins.x_top_left
            y1 = ins.y_top_left
            x2 = x1 + ins.width - 1
            y2 = y1 + ins.height - 1
            difficult = ins.difficult  # bool
            # truncated = ins.truncated # bool
            obj_struct = dict(name=cls_name, difficult=difficult, bbox=[x1, y1, x2, y2])
            objects.append(obj_struct)
        recs[image_name] = objects
    return recs


def get_gt(annopath, cachedir):
    # load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots_from_tencent_youtu_cache.pkl")

    # load annotations --------------------------------
    if not os.path.isfile(cachefile):
        # load annot from tencent youtu cached format
        # parse the cached annotation =============
        recs = parse_tencent_youtu_cache(annopath)
        # save
        print("Saving cached annotations to {:s}".format(cachefile))
        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)
    else:
        # load cached annotations
        print("****load cached annotaion: {}****".format(cachefile))
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)
    return recs


# calculate recall and precision for one class
def voc_eval(
        detpath,
        annots,
        classname,
        ovthresh=0.5,
        use_07_metric=False,
):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections # results/<comp_id>_det_test_xxx.txt
        detpath.format(classname) should produce the detection results file.
    annopath: Path to cached annotation, tencent youtu cached format (brambox)
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    imagenames = ['/'.join(key.split('/')[-2:]) for key in annots.keys()]
    imagenames = sorted(imagenames)

    # extract gt objects for this class
    #
    # annotations of current class
    class_recs = {}
    # npos: number of objects
    npos = 0
    for imagename in imagenames:
        # only get the annotations of the specific class in recs
        R = [obj for obj in annots[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        # if no difficult, all are 0.
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # len(R) number of gt_bboxes for this class，det: whether is detected, initialized to False。
        det = [False] * len(R)
        # add number of non-difficult gt bbox
        npos = npos + sum(~difficult)
        _imageid = imagename.split(r'/')[-1][:-4]
        class_recs[_imageid] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets (read detection results) -------------------------------
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    # TODO: fix this
    image_ids = ['/'.join(x[0].split('/')[-2:]) for x in splitlines]
    #     image_ids = [os.path.basename(x[0]).split('.')[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                    (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                    - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    print(f"miss rate for {classname} = {round(1-sum(tp)/npos,4)}")
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap, ap11_cls = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, ap11_cls


if __name__ == "__main__":
    import argparse

    classes = (
        "__background__",  # always index 0
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    parser = argparse.ArgumentParser(description="evaluation with gt annotation and detection results")
    parser.add_argument("--result_dir", help="result_dir", default="")
    parser.add_argument("--mode", default="voc", help="pr mode: voc or area")
    args = parser.parse_args()

    cached_annopath = osp.join(cur_dir, "../../vocdata/VOCdevkit/onedet_cache/test.pkl")
    cachedir = osp.join(cur_dir, "../../vocdata/VOCdevkit/annotations_cache")

    result_dir = args.result_dir
    output_dir = osp.join(result_dir, "pr_curve")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # use_07_metric = False
    use_07_metric = True
    aps = []
    ap11 = np.zeros(11)

    annots = get_gt(cached_annopath, cachedir)
    for i, cls in enumerate(classes):
        if cls == "__background__":
            continue
        filename = osp.join(
            result_dir,
            f"comp4_det_test_{cls}.txt")
        rec, prec, ap, ap11_cls = voc_eval(
            filename,
            annots,
            cls,
            ovthresh=0.5,
            use_07_metric=use_07_metric,
        )
        ap11 += np.array(ap11_cls) / (len(classes) - 1)
        aps.append(ap)
        print(("AP for {} = {:.4f}".format(cls, ap)))
        print("----------------------------------")
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(("Mean AP = {:.4f}".format(np.mean(aps))))