import scipy.optimize as optimize
from metrics import *


def vcol(vector):
    return vector.reshape((vector.shape[0], 1))


def vrow(vector):
    return vector.reshape((1, vector.shape[0]))


def num_samples(dataset):
    return dataset.shape[1];


def logreg_obj_wrap(DTR, LTR, l, prior):
    def logreg_obj(v):
        # ...
        # Compute and return the objective function value using DTR,LTR, l
        N = num_samples(DTR)
        nt = num_samples(DTR[:, LTR == 1])
        nf = num_samples(DTR[:, LTR == 0])
        w, b = v[0:-1], v[-1]  # v=[w,b]
        J = (l / 2) * np.sum(w * w)
        z = 2 * LTR - 1
        for index_sample in range(N):
            if LTR[index_sample] == 1:
                J += prior * (np.logaddexp(0, -z[index_sample] * (np.dot(w.T, DTR[:, index_sample]) + b)) / nt)
            else:
                J += (1 - prior) * (np.logaddexp(0, -z[index_sample] * (np.dot(w.T, DTR[:, index_sample]) + b)) / nf)

        return J

    return logreg_obj

def bin_logistic_regression_train(DTR, LTR, l, prior):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, prior)
    x0 = np.zeros(DTR.shape[0] + 1)
    v, _, _ = optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, iprint=-1)  # v=[w*,b*]
    w_best, b_best = v[0:-1], v[-1]
    print("J(w*,b*) = " + str(logreg_obj(v)))
    return w_best, b_best


def logistic_regression_eval(DTE, LTE, w_best, b_best, prior):
    N = num_samples(DTE)
    preds = np.array([])
    scores = []
    for index_sample in range(N):
        score = np.dot(w_best.T, DTE[:, index_sample]) + b_best
        scores.append(score - math.log(prior / (1 - prior)))
        if score > 0:
            pred = 1
        else:
            pred = 0

        preds = np.append(preds, pred)

    num_correct_preds = np.sum(preds == LTE)
    accuracy = num_correct_preds / N

    return preds, accuracy, scores

