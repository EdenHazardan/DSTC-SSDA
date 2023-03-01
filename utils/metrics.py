import numpy as np
from sklearn.metrics import confusion_matrix

synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_16_to_13 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, return_class=False, return_confusion=False):
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)

        if return_confusion:
            return mean_iu, iu, hist

        if return_class:
            return mean_iu, iu
        else:
            return mean_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScore_acc(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, return_class=False):
        hist = self.confusion_matrix
        PA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        print("PA = ",PA)
        MPA = np.nanmean(PA)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)

        if return_class:
            return mean_iu, iu, PA, MPA
        else:
            return mean_iu, PA, MPA

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class runningScore_syn(object):
    def __init__(self, n_classes):
        self.n_classes = 16
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, return_class=False):
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # iu = np.nan_to_num(iu) # notice that there is no bycible in VIPER val set
        iu_13 = iu[synthia_set_16_to_13]
        mean_iu = np.nanmean(iu)
        mean_iu_13 = np.nanmean(iu_13)

        if return_class:
            return mean_iu, mean_iu_13, iu, iu_13
        else:
            return mean_iu, mean_iu_13

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class runningScore_recall(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, return_class=False):
        hist = self.confusion_matrix
        iou = np.diag(hist) / hist.sum(axis=0)
        miou = np.nanmean(iou)
        if return_class:
            return miou, iou
        else:
            return miou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class runningScore_precision(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, return_class=False):
        hist = self.confusion_matrix
        iou = np.diag(hist) / hist.sum(axis=1)
        miou = np.nanmean(iou)
        if return_class:
            return miou, iou
        else:
            return miou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))