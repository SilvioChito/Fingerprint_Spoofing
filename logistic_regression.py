import math
import scipy.optimize as optimize
import numpy as np
from metrics import *
from PCA import *


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


def quad_logreg_obj_wrap(DTR, LTR, l, prior):
    def quad_logreg_obj(v):
        # ...
        # Compute and return the objective function value using DTR,LTR, l
        N = num_samples(DTR)
        nt = num_samples(DTR[:, LTR == 1])
        nf = num_samples(DTR[:, LTR == 0])
        w, b = v[0:-1], v[-1]  # v=[w,b]
        J = (l / 2) * np.sum(w * w)
        z = 2 * LTR - 1
        for index_sample in range(N):
            x = vcol(DTR[:, index_sample])
            mat_x = np.dot(x, x.T)
            vec_x = vcol(np.hstack(mat_x))
            fi_x = np.vstack((vec_x, x))
            if LTR[index_sample] == 1:
                J += prior * (np.logaddexp(0, -z[index_sample] * (np.dot(w.T, fi_x))) / nt)
            else:
                J += (1 - prior) * (np.logaddexp(0, -z[index_sample] * (np.dot(w.T, fi_x))) / nf)

        return J

    return quad_logreg_obj


def quad_logreg_obj_wrap_with_grad(DTR, LTR, l, prior):
    def quad_logreg_obj(v):
        # ...
        # Compute and return the objective function value using DTR,LTR, l
        N = num_samples(DTR)
        nt = num_samples(DTR[:, LTR == 1])
        nf = num_samples(DTR[:, LTR == 0])
        w, b = v[0:-1], v[-1]  # v=[w,b]
        J = (l / 2) * np.sum(w * w)
        z = 2 * LTR - 1
        for index_sample in range(N):
            x = vcol(DTR[:, index_sample])
            mat_x = np.dot(x, x.T)
            vec_x = vcol(np.hstack(mat_x))
            fi_x = np.vstack((vec_x, x))
            if LTR[index_sample] == 1:
                J += prior * (np.logaddexp(0, -z[index_sample] * (np.dot(w.T, fi_x))) / nt)
            else:
                J += (1 - prior) * (np.logaddexp(0, -z[index_sample] * (np.dot(w.T, fi_x))) / nf)

        return J

    def quad_logreg_grad(v):
        # Compute the gradient of the objective function with respect to v

        N = num_samples(DTR)
        nt = num_samples(DTR[:, LTR == 1])
        nf = num_samples(DTR[:, LTR == 0])
        w, b = v[0:-1], v[-1]  # v=[w, b]
        grad_w = vcol(l * w)  # Gradient of the regularization term
        grad_b = 0  # Gradient of the bias term
        z = 2 * LTR - 1

        for index_sample in range(N):
            x = vcol(DTR[:, index_sample])
            mat_x = np.dot(x, x.T)
            vec_x = vcol(np.hstack(mat_x))
            fi_x = np.vstack((vec_x, x))
            dot_product = np.dot(w.T, fi_x)

            if LTR[index_sample] == 1:
                exp_term = np.exp(-z[index_sample] * dot_product)
                term1 = -prior * z[index_sample] * fi_x * exp_term
                term2 = (nt * (1 + exp_term))
                grad_w += term1 / term2
                grad_b += -prior * z[index_sample] * exp_term / (nt * (1 + exp_term))
            else:
                exp_term = np.exp(-z[index_sample] * dot_product)
                grad_w += -(1 - prior) * z[index_sample] * fi_x * exp_term / (nf * (1 + exp_term))
                grad_b += -(1 - prior) * z[index_sample] * exp_term / (nf * (1 + exp_term))

        grad = np.concatenate((grad_w, [grad_b]))  # Concatenate gradients of w and b
        return grad

    return quad_logreg_obj, quad_logreg_grad


def bin_logistic_regression_train(DTR, LTR, l, prior):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, prior)
    x0 = np.zeros(DTR.shape[0] + 1)
    v, _, _ = optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, iprint=-1)  # v=[w*,b*]
    w_best, b_best = v[0:-1], v[-1]
    print("J(w*,b*) = " + str(logreg_obj(v)))
    return w_best, b_best


def bin_quadratic_logistic_regression_train(DTR, LTR, l, prior):
    logreg_obj = quad_logreg_obj_wrap(DTR, LTR, l, prior)
    x0 = np.zeros(DTR.shape[0] * DTR.shape[0] + DTR.shape[0] + 1)
    v, _, _ = optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, iprint=-1)  # v=[w*,b*]
    w_best, b_best = v[0:-1], v[-1]
    print("J(w*,b*) = " + str(logreg_obj(v)))
    return w_best, b_best


def bin_quadratic_logistic_regression_with_grad_train(DTR, LTR, l, prior):
    logreg_obj, logreg_grad = quad_logreg_obj_wrap_with_grad(DTR, LTR, l, prior)
    x0 = np.zeros(DTR.shape[0] * DTR.shape[0] + DTR.shape[0] + 1)
    v, _, _ = optimize.fmin_l_bfgs_b(logreg_obj, x0, fprime=logreg_grad, iprint=-1)  # v=[w*,b*]
    w_best, b_best = v[0:-1], v[-1]
    print("J(w*,b*) = " + str(logreg_obj(v)))
    return w_best, b_best


def quad_logistic_regression_eval(DTE, LTE, w_best, b_best, prior):
    N = num_samples(DTE)
    preds = np.array([])

    scores = []
    for index_sample in range(N):
        x = vcol(DTE[:, index_sample])

        mat_x = np.dot(x, x.T)
        vec_x = vcol(np.hstack(mat_x))
        fi_x = np.vstack((vec_x, x))

        score = np.dot(w_best.T, fi_x) + b_best
        scores.append(score - math.log(prior / (1 - prior)))
        if score > 0:
            pred = 1
        else:
            pred = 0

        preds = np.append(preds, pred)

    num_correct_preds = np.sum(preds == LTE)
    accuracy = num_correct_preds / N

    return preds, accuracy, scores


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


def loadFile(file):
    vec_list = []
    labelsList = []
    with open(file, 'r') as myfile:
        for line in myfile:
            features = line.split(',')[:10]
            label = line.split(',')[-1].strip()
            my_vec = np.array([float(f) for f in features])
            vec_list.append(vcol(my_vec))
            labelsList.append(label)

    return np.hstack(vec_list), np.array(labelsList, dtype=np.int32)

"""
 FOR THE VARIOUS EXPERIMENTS WE USED DIFFERENT K-FOLDS, WITH DIFFERENT PARAMETERS.
 THE FUNCTION IS ALWAYS THE SAME, BUT THE FUNCTION INSIDE (THE MODEL I USED EACH TIME)
 VARIES DEPENDING ON THE MODEL I USED (IN THE FOLLOWING I'VE USED RBF SVM)
"""
def K_fold(K, train_set, labels, l, prior, cfn, cfp):  # K, D, l, prior, cfn, cfp
    np.random.seed(0)
    indices = np.arange(train_set.shape[1])
    np.random.shuffle(indices)
    train_set = train_set[:, indices]
    labels = labels[indices]

    fold_size = train_set.shape[1] // K
    predicted_labels_tot = []
    scores_tot = []
    LTR_for_minDcf = []

    for i in range(K):
        start = i * fold_size
        end = (i + 1) * fold_size
        val_range = np.in1d(range(train_set.shape[1]), range(int(start), int(end)))

        DTR = train_set[:, np.invert(val_range)]
        LTR = labels[np.invert(val_range)]

        DTE = train_set[:, val_range]
        LTE = labels[val_range]

        # LR LINEAR
        # w_opt, b_opt = bin_logistic_regression_train(DTR, LTR, l, prior)
        # predicted_labels, _, scores = logistic_regression_eval(DTE, LTE, w_opt, b_opt, prior)

        # LR QUADRATIC
        w_best, b_best = bin_quadratic_logistic_regression_with_grad_train(DTR, LTR, l, prior)
        predicted_labels, _, scores = quad_logistic_regression_eval(DTE, LTE, w_best, b_best, prior)

        predicted_labels_tot.append(predicted_labels)
        scores_tot.append(scores)
        LTR_for_minDcf.append(LTE)


    actual_labels = np.concatenate(LTR_for_minDcf).astype(int)
    #pred_labels = np.concatenate(predicted_labels_tot).astype(int)
    scores_actual = np.concatenate(scores_tot).flatten()

    min_DCF = compute_DCFmin(scores_actual, actual_labels, prior, cfn, cfp)
    return min_DCF


if __name__ == '__main__':
    file = "Train.txt"
    file2 = "Test.txt"
    D, L = loadFile(file)
    DTE, LTE = loadFile(file2)

    D_pca8, P = pca(D, 8)
    D_pca8_z = z_score_normalization(D_pca8)
    DTE_pca8 = np.dot(P.T, DTE)
    DTE_pca8_z = z_score_normalization(DTE_pca8)

    pi = 0.5
    Cfn = 1
    Cfp = 10
    pi_tilde = (pi * Cfn) / (pi * Cfn + (1 - pi) * Cfp)  # ci permette di avere costi uguali
    # folds = 10
    l = 0.001
    """
    l_list=[0-0001, 0.001, 0.01, 0.1, 1]
    # THIS IS AN EXAMPLE OF USING K-FOLD CROSS VALIDATION FOR THE HYPER PARAMETER TUNING
    for l in l_list:
        min_dcf_curr_pca8 = K_fold(folds, DTR_pca8, LTR, l, pi_tilde, 1, 1)
        min_dcf_curr_pca8_z = K_fold(folds, DTR_pca8_z, LTR, l, pi_tilde, 1, 1)

        min_dcf_list_pca8.append(min_dcf_curr_pca8)
        min_dcf_list_pca8_z.append(min_dcf_curr_pca8_z)

    print('min dcf list (quad.) with pca-8:')
    print(min_dcf_list_pca8)

    print('min dcf list (quad.) with pca-8 + z-norm:')
    print(min_dcf_list_pca8_z)

    plt.semilogx(l_list, min_dcf_list_pca8, color='blue', label='Q Log-Reg (PCA-8)', linestyle='dashed')
    plt.semilogx(l_list, min_dcf_list_pca8_z, color='red', label='Q Log-Reg (PCA-8 + Z norm)')
    plt.xlabel('lambda')
    plt.ylabel('min DCF')
    plt.legend()
    plt.show()
    """
    ### TRAINING OVER DTR, LTR ###
    (DTR, LTR), (DTC, LTC) = split_db_2to1(D_pca8_z, L, seed=0) # DIVIDO IL TRAINING SET DAL CALIBRATION SET
    w_best, b_best = bin_quadratic_logistic_regression_with_grad_train(DTR, LTR, l, pi_tilde)

    ### TRAINING WITH THE CALIBRATOR OVER DTC, LTC ###
    _, _, scores_train_cal = quad_logistic_regression_eval(DTC, LTC, w_best, b_best, pi_tilde)
    # SI BASA SUGLI SCORE E NON SUI DATI NUDI E CRUDI !!!
    w_best_to_cal, b_best_to_cal = bin_logistic_regression_train(vrow(np.array(scores_train_cal)), LTC, l, pi_tilde) # PARAMETRI PER RICALIBRARE !!!

    ### EVALUATION USING TRAINING PARAMETERS ###
    _, _, scores_UNCAL = quad_logistic_regression_eval(DTE_pca8_z, LTE, w_best, b_best, pi_tilde)
    predicted, _, scores_CAL = logistic_regression_eval(vrow(np.array(scores_UNCAL)), LTE, w_best_to_cal, b_best_to_cal, pi_tilde)     # ricalibrazione
    plot_Bayes_error(scores_CAL, LTE)

    confusion_matrix = compute_confusion_matrix(LTE, predicted.astype(int), 2)
    actual_dcf = compute_DCF(pi_tilde, 1, 1, confusion_matrix)
    min_dcf = compute_DCFmin(scores_CAL, LTE, pi_tilde, 1, 1)
    print("actual dcf: {}".format(actual_dcf))
    print("min dcf: {}".format(min_dcf))
