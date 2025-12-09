import copy
import logging, torch 
import numpy as np 
import os 
import torch.nn as nn
from scipy import linalg
from scipy.optimize import brentq
from scipy.stats import gaussian_kde
from sklearn.mixture._gaussian_mixture import GaussianMixture
from sklearn.metrics import recall_score, average_precision_score
import torchmetrics
import torch.nn.functional as F
import random

# CODE CITATIONS:
# the classes for the TTA come from https://github.com/yuyongcan/STAMP/tree/master
# the functions for GMM related tasks come from https://github.com/pascalschlachter/GMM/blob/main/adaptation.py

#### helpers ####

def entropy(probabilities):
    """
    Computes the entropy of each row in the input tensor
    """

    epsilon = 1e-5
    entropy = -probabilities * torch.log(probabilities + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy 

def gmm_threshold(known_scores, weights_init, fixed_weights, thr_curr = None, reset_weights = True, momentum = 0.5, seed = 2020):
    """
    Fits a 2-component GMM to the known scores and returns the threshold 
    between the two components
    """

    random.seed(seed)
    np.random.seed(seed)

    logger = logging.getLogger(__name__)
    logging.info("[DEBUG] ---start: gmm_threshold---")

    # prepare scores
    try:
        known_scores = known_scores.detach()
    except:
        pass
        
    _scores = known_scores.unsqueeze(1)
    _scores = torch.nan_to_num(_scores, nan=0.5, posinf=1, neginf=0)

    # fit GMM
    gm = CustomGaussianMixture(n_components=2,
                                random_state=0,
                                weights_init=weights_init,
                                means_init=np.array([[0.75], [0.25]]),
                                covariance_type="spherical",
                                fixed_weights=fixed_weights,
                                verbose=False,
                                verbose_interval=1,
                                n_init=3).fit(_scores)
    
    # get GMM means and weights
    mi, ma = gm.means_.min(), gm.means_.max()
    logging.info(f"gm.means_: {gm.means_.squeeze()}, gm.weights_: {gm.weights_}, "
          f"gm.covariance_: {gm.covariances_}")

    # reset the weights to balance the OS* and UNK (see TASC Appendix)
    if reset_weights:
        gm.weights_ = np.array([0.5, 0.5])
    
    # calculate the threshold
    rang = np.array([i for i in np.arange(mi, ma, 0.001)]).reshape(-1, 1)
    temp = gm.predict(rang)
    thr_new = np.array(temp).mean()*(ma - mi) + mi
    thr_curr = thr_new if thr_curr is None else thr_curr
    thr = (1 - momentum)*thr_new + momentum*thr_curr

    logging.info(f"thr: {thr:0.2f}, thr_curr: {thr_curr:0.2f}, thr_new: {thr_new:0.2f}")
    return thr

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """
    Entropy of softmax distribution from logits
    """
    
    temperature = 1
    x = x / temperature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """
    Estimate the diagonal covariance vectors 

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features). The covariance vector of the current components.
    """

    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means**2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """
    Estimate the spherical variance values 

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """

    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """
    Estimate the full covariance matrices 

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """

    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances

def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """
    Estimate the Gaussian distribution parameters 

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances

def h_score_and_accuracies_sfunida(class_list, gt_label_all, pred_cls_all, open_flag=True, open_thresh=0.55, pred_unc_all=None):
    """
    Harmonic mean score (h-score) and accuracies for known and unknown classes
    in SFUnida models (GLC and LEAD)

    - class_list: list of class labels, including known and unknown classes
    - gt_label_all: ground truth labels for all samples
    - pred_cls_all: predicted class probabilities for all samples
    - open_flag: indicates if open-set classification is enabled
    - open_thresh: threshold for open-set classification (defaultfrom LEAD model_config.py w_0)
    - pred_unc_all: represents the uncertainty for each prediction

    NOTE: This method predicts unknowns adjusting predicted known labels based on the open-set threshold
    """
    
    logger = logging.getLogger(__name__)
    logging.info("[DEBUG] ---start: h_score_and_accuracies_sfunida---")
    # logging.info(f"[DEBUG] gt_label_all: {gt_label_all}")

    # placeholders for per class predictions
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)

    # obtaining predicted labels from logits
    pred_label_all = torch.max(pred_cls_all, dim=1)[1] #[N]
    # logging.info(f"[DEBUG] predicted label (before update): f{pred_label_all}")
    
    # readjusting logits of predicted labels for open-set classification
    if open_flag: 
        cls_num = pred_cls_all.shape[1]
        
        if pred_unc_all is None: # if no uncertainty tensor provided, we normalize the shannon entropy
            pred_unc_all = entropy(pred_cls_all)/np.log(cls_num) # [N]

        # logging.info(f"[DEBUG] normalized entropies: {pred_unc_all}")

        unc_idx = torch.where(pred_unc_all > open_thresh)[0]
        pred_label_all[unc_idx] = cls_num # set these pred results to unknown

    # logging.info(f"[DEBUG] predicted after update: f{pred_label_all}")

    # computing number of correct predictions per class
    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_all == label)[0]
        correct_idx = torch.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))

    per_class_acc = per_class_correct / (per_class_num + 1e-5) # computing accuracy per class (plus offset)

    # computting known classes avg accuracy, unknown accuracy and h-score
    if open_flag:
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    else:
        known_acc = per_class_correct.sum() / (per_class_num.sum() + 1e-5)
        unknown_acc = 0.0
        h_score = 0.0

    logging.info("[DEBUG] ---end: h_score_and_accuracies_sfunida---")

    return h_score, known_acc, unknown_acc, per_class_acc, pred_label_all

def h_score_and_accuracies_tasc(tasc_model, class_list, gt_label_all, pred_cls_all, unknown_scores_dict, open_flag=True, inference=True):
    """
    Harmonic mean score (h-score) and accuracies for known and unknown classes
    in TASC models
    """

    logger = logging.getLogger(__name__)
    logging.info("[DEBUG] ---start: h_score_and_accuracies_tasc---")

    # placeholders for per class predictions
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)

    # predictions in source and target domains
    _, pred_label_all = torch.max(pred_cls_all, -1)
    pred_label_all_without_unknown = copy.deepcopy(pred_label_all)

    # unknown scores normalization
    known_scores = -unknown_scores_dict["UniMS"]
    known_scores = (known_scores - known_scores.min()) / (known_scores.max() - known_scores.min())

    # finding threshold to detect unknown samples
    weights_init = np.array([tasc_model.estimated_shared, \
                             tasc_model.num_clusters - \
                             tasc_model.estimated_shared])
    weights_init = weights_init / weights_init.sum()
    logging.info(f"[DEBUG] weights_init: {weights_init}")
    reset_weights = True if inference else False
    thr = gmm_threshold(known_scores, weights_init, fixed_weights=True, thr_curr=tasc_model.thr_curr, reset_weights=reset_weights)
    tasc_model.thr_curr = thr

    # adjusting predicted labels based on the open-set threshold
    if open_flag:
        known_mask = known_scores > thr
        pred_label_all[~known_mask] = tasc_model.num_classes

    # computing number of correct predictions per class
    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_all == label)[0]
        correct_idx = torch.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))

    per_class_acc = per_class_correct / (per_class_num + 1e-5) # computing accuracy per class (plus offset)

    # computting known classes avg accuracy, unknown accuracy and h-score
    if open_flag:
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    else:
        known_acc = per_class_correct.sum() / (per_class_num.sum() + 1e-5)
        unknown_acc = 0.0
        h_score = 0.0
    
    return h_score, known_acc, unknown_acc, per_class_acc, pred_label_all

def _compute_precision_cholesky(covariances, covariance_type):
    """
    Compute the Cholesky decomposition of the precisions

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """

    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    elif covariance_type == "tied":
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances)
    return precisions_chol

def np_threshold(h0_probs, h1_probs, length_prob_grid=2001, alpha=0.05, epsilon = 1e-5):
    """
    Neyman-Pearson threshold for likelihood ratio test

    h0_probs: function that returns the probabilities under the null hypothesis (H0)
    h1_probs: function that returns the probabilities under the alternative hypothesis (H1)
    length_prob_grid: number of points in the probability grid
    alpha: significance level (false positive rate)
    """

    arr0 = h0_probs.detach().cpu().numpy().astype(float)
    arr1 = h1_probs.detach().cpu().numpy().astype(float)

    density_h0 = gaussian_kde(arr0) # h0 (not passed filters)
    density_h1 = gaussian_kde(arr1) # h1/ha (passed filters)

    # density_h0 = gaussian_kde(h0_probs.to_numpy(float))
    # density_h1 = gaussian_kde(h1_probs.to_numpy(float))

    # probability grid
    max_probability_grid = np.linspace(0, 1, length_prob_grid)

    # density under null hypothesis (h0) and likelihood ratio
    p0 = density_h0(max_probability_grid)
    likelihood_ratio = density_h1(max_probability_grid) / (p0 + epsilon)

    # function to obtain an area under H0 equal to alpha 
    def size_under_null(kval):
        mask = likelihood_ratio >= kval
        return np.trapz(p0[mask], max_probability_grid[mask]) - alpha

    # find threshold using Brent's method
    optimal_threshold = brentq(size_under_null, likelihood_ratio.min(), likelihood_ratio.max())
    
    return optimal_threshold, density_h0, density_h1

def np_optimal_classifier(current_probability, np_threshold, density_h0, density_h1, epsilon=1e-5):
    """
    Classifier based on Neyman-Pearson threshold
    """

    likelihood_ratio = density_h1(current_probability) / (density_h0(current_probability) + epsilon)
    return likelihood_ratio >= np_threshold

def vectorize_density(fn):
    """
    Vectorizes a density function to apply it to a tensor
    """

    return lambda x: torch.tensor([fn(xi.item()) for xi in x], dtype=torch.float32, device=x.device)

def compute_os_variance(os, th):
    """
    Calculates the area of a rectangle.

    Parameters:
        os : OOD score queue.
        th : Given threshold to separate weak and strong OOD samples.

    Returns:
        float: Weighted variance at the given threshold th.
    """

    thresholded_os = np.zeros(os.shape)
    thresholded_os[os >= th] = 1

    # compute weights
    nb_pixels = os.size
    nb_pixels1 = np.count_nonzero(thresholded_os)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = os[thresholded_os == 1]
    val_pixels0 = os[thresholded_os == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1

#### classes ####

class CrossentropyLabelSmooth(nn.Module):
    """
    Cross entropy loss with label smoothing regularizer.
    
    NOTE: Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    """      

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossentropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets, applied_softmax=True):
        """
        Calculates the cross-entropy loss with label smoothing
        """

        if applied_softmax: # checks if inputs already have softmax applied
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)
        
        if inputs.shape != targets.shape:
            targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        
        # applying label smoothing
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
         
        if self.reduction:
            return loss.mean()
        else:
            return loss

class CustomGaussianMixture(GaussianMixture):
    """
    TASC Custom Gaussian Mixture Model with fixed weights option
    """

    def __init__(self, n_components=1,
                       covariance_type="full",
                       tol=1e-3,
                       reg_covar=1e-6,
                       max_iter=100,
                       n_init=1,
                       init_params="kmeans",
                       weights_init=None,
                       means_init=None,
                       precisions_init=None,
                       random_state=None,
                       warm_start=False,
                       verbose=0,
                       verbose_interval=10,
                       fixed_weights=False):
        
        self.fixed_weights = fixed_weights
        super().__init__(n_components=n_components,
                         covariance_type=covariance_type,
                         tol=tol,
                         reg_covar=reg_covar,
                         max_iter=max_iter,
                         n_init=n_init,
                         init_params=init_params,
                         weights_init=weights_init,
                         means_init=means_init,
                         precisions_init=precisions_init,
                         random_state=random_state,
                         warm_start=warm_start,
                         verbose=verbose,
                         verbose_interval=verbose_interval)
 
    def _m_step(self, X, log_resp):
        """
        Maximization step
        """

        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
                            X, np.exp(log_resp), self.reg_covar, self.covariance_type)
        
        if self.fixed_weights:
            self.weights_ = self.weights_init
        
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    
    def __init__(self, projector, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature

        self.projector = projector

    def forward(self, features, labels=None, mask=None, confident_unknown_features=torch.tensor([])):
        """
        Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            confident_unknown_features: features of samples labeled as unkown by pseudo-labeling
        Returns:
            A loss scalar.
        """
        
        device = (torch.device("cuda")
                  if features.is_cuda
                  else torch.device("cpu"))

        self.projector.to(device)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # number M of different views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # stack views to "batch" (size: M*N)
        contrast_feature = self.projector(contrast_feature)  # project into projection space
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits i.e. M*NxM*N-matrix of z_p*z_q/tau for all p,q in I
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if confident_unknown_features.numel() != 0:
            confident_unknown_features = torch.cat(torch.unbind(confident_unknown_features, dim=1), dim=0)
            confident_unknown_features = self.projector(confident_unknown_features)
            confident_unknown_features = F.normalize(confident_unknown_features, p=2, dim=1)
            # compute dot products of each known sample with all unknown samples, respectively
            confident_unknown_contrast = torch.div(torch.matmul(anchor_feature, confident_unknown_features.T),
                                                   self.temperature)
            confident_unknown_contrast = confident_unknown_contrast - logits_max.detach()[0]
            confident_unknown_contrast = torch.exp(confident_unknown_contrast)
        else:
            confident_unknown_contrast = torch.tensor([0])

        # tile mask, i.e. main diagonals of the M^2 NxN-submatrices
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # mask is 1 where z_i*z_j(i) in logits

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # exp(z_i*z_a/tau) for all a in A(i), 0 for a=i (on main diagonal)
        # compute log(exp(z_i*z_j(i)/tau)/sum_a exp(z_i*z_a/tau))=z_i*z_a/tau-log(sum_a exp(z_i*z_a/tau))
        exp_logits_sum = exp_logits.sum(1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum + torch.ones_like(exp_logits_sum) * confident_unknown_contrast.sum())

        # compute mean of log-likelihood over positive for each i in I
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # compute mean over all i in I
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class HScore(torchmetrics.Metric):
    def __init__(self, known_classes_num, shared_classes_num):
        super(HScore, self).__init__()

        self.total_classes_num = known_classes_num + 1
        self.shared_classes_num = shared_classes_num

        self.add_state("correct_per_class", default=torch.zeros(self.total_classes_num), dist_reduce_fx="sum")
        self.add_state("total_per_class", default=torch.zeros(self.total_classes_num), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        for c in range(self.total_classes_num):
            self.total_per_class[c] += (target == c).sum()
            self.correct_per_class[c] += ((preds == target) * (target == c)).sum()

    def compute(self):
        per_class_acc = self.correct_per_class / (self.total_per_class + 1e-5)
        known_acc = per_class_acc[:self.shared_classes_num].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
        return h_score, known_acc, unknown_acc             

if __name__ == "__main__":
    print("Module with metrics functions/classes")