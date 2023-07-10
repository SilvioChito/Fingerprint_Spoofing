from metrics import *
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.optimize as optimize
import numpy as np
from calibrator import *
import matplotlib.pyplot as plt
from PCA import *


def vcol(vector):
    return vector.reshape((vector.shape[0], 1))


def vrow(vector):
    return vector.reshape((1, vector.shape[0]))


def J_primal(w_best, C, DTR, LTR, K):
    ones_row = K * np.ones((1, DTR.shape[1]))
    DTR_b = np.append(DTR, ones_row, axis=0)
    scores = np.dot(vrow(w_best), DTR_b)
    ones = np.ones(scores.shape)
    z = np.where(LTR == 0, -1, 1)
    hinged_scores = np.maximum(ones - z * scores, 0)
    summed_scores = np.sum(hinged_scores)
    return 0.5 * np.dot(vrow(w_best), vcol(w_best)).squeeze() + C * summed_scores


def Ld(alpha, K, DTR):
    # ...
    # Compute and return the loss function value using DTR,LTR, l
    ones_row = K * np.ones((1, DTR.shape[1]))

    # Concatenate the row of ones with 'DTR' vertically
    DTR_b = np.append(DTR, ones_row, axis=0)

    G = np.dot(DTR_b.T, DTR_b)

    z = np.where(LTR == 0, -1, 1)

    z = np.dot(vcol(z), vrow(
        z))  # IMPORTANTE! il primo parametro deve essere verticale perciò assicurati che z sia orizzontale quando lo passi

    H = z * G

    Ld = 0.5 * np.dot(np.dot(vrow(alpha), H), vcol(alpha)) - np.dot(vrow(alpha), np.ones((alpha.shape[0], 1)))
    return Ld


def Ld_poly(alpha, c, DTR, d):
    z = np.where(LTR == 0, -1, 1)
    z = np.dot(vcol(z), vrow(
        z))  # IMPORTANTE! il primo parametro deve essere verticale perciò assicurati che z sia orizzontale quando lo passi
    k = np.power((np.dot(DTR.T, DTR) + c * np.ones(np.dot(DTR.T, DTR).shape)), d) + K * K * np.ones(
        np.dot(DTR.T, DTR).shape)
    H = z * k
    Ld = 0.5 * np.dot(np.dot(vrow(alpha), H), vcol(alpha)) - np.dot(vrow(alpha), np.ones((alpha.shape[0], 1)))

    return Ld


def Ld_wrap(DTR, LTR, K):
    # ...
    # Compute and return the loss function value using DTR,LTR, l
    ones_row = K * np.ones((1, DTR.shape[1]))

    # Concatenate the row of ones with 'DTR' vertically
    DTR_b = np.append(DTR, ones_row, axis=0)
    G = np.dot(DTR_b.T, DTR_b)
    z = np.where(LTR == 0, -1, 1)
    z = np.dot(vcol(z), vrow(
        z))  # IMPORTANTE! il primo parametro deve essere verticale perciò assicurati che z sia orizzontale quando lo passi
    H = z * G

    def Ld(alpha):
        Ld = 0.5 * np.dot(np.dot(vrow(alpha), H), vcol(alpha)) - np.dot(vrow(alpha), np.ones((alpha.shape[0], 1)))

        return Ld

    def Ld_grad(alpha):
        Ld_grad = np.dot(H, vcol(alpha)) - np.ones((alpha.shape[0], 1))
        Ld_grad = np.squeeze(Ld_grad)

        return Ld_grad

    return Ld, Ld_grad


def Ld_poly_wrap(DTR, LTR, K, c, d):
    z = np.where(LTR == 0, -1, 1)
    z = np.dot(vcol(z), vrow(
        z))  # IMPORTANTE! il primo parametro deve essere verticale perciò assicurati che z sia orizzontale quando lo passi
    k = np.power((np.dot(DTR.T, DTR) + c * np.ones(np.dot(DTR.T, DTR).shape)), d) + K * K * np.ones(
        np.dot(DTR.T, DTR).shape)
    H = z * k

    def Ld_poly(alpha):
        Ld = 0.5 * np.dot(np.dot(vrow(alpha), H), vcol(alpha)) - np.dot(vrow(alpha), np.ones((alpha.shape[0], 1)))

        return Ld

    def Ld_poly_grad(alpha):
        Ld_grad = np.dot(H, vcol(alpha)) - np.ones((alpha.shape[0], 1))
        Ld_grad = np.squeeze(Ld_grad)

        return Ld_grad

    return Ld_poly, Ld_poly_grad


def Ld_RBF_wrap(DTR, LTR, K, gamma):
    z = np.where(LTR == 0, -1, 1)
    z = np.dot(vcol(z), vrow(
        z))  # IMPORTANTE! il primo parametro deve essere verticale perciò assicurati che z sia orizzontale quando lo passi
    num_data = DTR.shape[1]  # Number of data points
    num_features = DTR.shape[0]  # Number of data features
    distances = pdist(DTR.T)  # Compute pairwise distances between the data points

    distance_matrix = squareform(distances)  # Convert the distances to a square matrix
    distance_matrix = np.square(distance_matrix)
    k = np.exp(-gamma * distance_matrix) + K * K  # *np.ones(distances.shape)
    H = z * k

    def Ld_RBF(alpha):
        Ld = 0.5 * np.dot(np.dot(vrow(alpha), H), vcol(alpha)) - np.dot(vrow(alpha), np.ones((alpha.shape[0], 1)))

        return Ld

    def Ld_RBF_grad(alpha):
        Ld_grad = np.dot(H, vcol(alpha)) - np.ones((alpha.shape[0], 1))
        Ld_grad = np.squeeze(Ld_grad)

        return Ld_grad

    return Ld_RBF, Ld_RBF_grad


def train_SVM(DTR, LTR, C, K):
    loss, loss_grad = Ld_wrap(DTR, LTR, K)
    x0 = np.zeros(DTR.shape[1])
    bounds = np.array([(0, C)] * DTR.shape[1])

    alpha_opt, _, _ = optimize.fmin_l_bfgs_b(loss, x0, fprime=loss_grad, bounds=bounds, factr=1.0)

    z = np.where(LTR == 0, -1, 1)

    # Compute and return the loss function value using DTR,LTR, l
    ones_row = K * np.ones((1, DTR.shape[1]))

    # Concatenate the row of ones with 'DTR' vertically
    DTR_b = np.append(DTR, ones_row, axis=0)

    w_best = np.sum(np.dot(vrow(alpha_opt * z), DTR_b.T), axis=0)

    return w_best, alpha_opt


def train_poly_SVM(DTR, LTR, C, K, c, d):
    loss, loss_grad = Ld_poly_wrap(DTR, LTR, K, c, d)
    x0 = np.zeros(DTR.shape[1])
    bounds = np.array([(0, C)] * DTR.shape[1])
    alpha_opt, _, _ = optimize.fmin_l_bfgs_b(loss, x0, fprime=loss_grad, bounds=bounds, factr=1.0)
    return alpha_opt


def train_RBF_SVM(DTR, LTR, C, K, gamma):
    loss, loss_grad = Ld_RBF_wrap(DTR, LTR, K, gamma)
    x0 = np.zeros(DTR.shape[1])
    bounds = np.array([(0, C)] * DTR.shape[1])
    alpha_opt, _, _ = optimize.fmin_l_bfgs_b(loss, x0, fprime=loss_grad, bounds=bounds, factr=1.0)
    print(loss(alpha_opt))
    return alpha_opt


def evaluate_svm(w_best, DTE, LTE, K):
    # Compute and return the loss function value using DTR,LTR, l
    ones_row = K * np.ones((1, DTE.shape[1]))

    # Concatenate the row of ones with 'DTR' vertically
    DTE_b = np.append(DTE, ones_row, axis=0)
    scores = np.dot(vrow(w_best), DTE_b)
    # Create predicted_labels array based on scores
    predicted_labels = np.where(scores > 0, 1, 0).squeeze()
    return scores, predicted_labels


def evaluate_poly_svm(alpha_opt, DTE, LTE, DTR, LTR, K, c, d):
    z = np.where(LTR == 0, -1, 1)
    k = np.power((np.dot(DTR.T, DTE) + c * np.ones(np.dot(DTR.T, DTE).shape)), d) + K * K * np.ones(
        np.dot(DTR.T, DTE).shape)
    scores = np.dot(vrow(alpha_opt * z), k)

    # Create predicted_labels array based on scores
    predicted_labels = np.where(scores > 0, 1, 0).squeeze()
    return scores, predicted_labels


def evaluate_RBF_svm(alpha_opt, DTE, LTE, DTR, LTR, K, gamma):
    z = np.where(LTR == 0, -1, 1)
    distance_matrix = cdist(DTR.T, DTE.T)
    distance_matrix = np.square(distance_matrix)
    k = np.exp(-gamma * distance_matrix) + K * K  # *np.ones(np.dot(DTR.T,DTE).shape)
    scores = np.dot(vrow(alpha_opt * z), k)
    scores_bis = scores - math.log((1/11) / (1 - (1/11)))

    # Create predicted_labels array based on scores
    predicted_labels = np.where(scores > 0, 1, 0).squeeze()
    return scores_bis, predicted_labels


def accuracy(predicted_labels, original_labels):
    if len(predicted_labels) != len(original_labels):
        return
    total_samples = len(predicted_labels)

    correct = (predicted_labels == original_labels).sum()
    return (correct / total_samples) * 100


def error_rate(predicted_labels, original_labels):
    # print(predicted_labels)
    # print(original_labels)
    return 100 - accuracy(predicted_labels, original_labels)


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
def K_fold(K, train_set, labels, C, k, prior, cfn, cfp, gamma):
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

        alpha_opt = train_RBF_SVM(DTR, LTR, C, k, gamma)
        scores, predicted_labels = evaluate_RBF_svm(alpha_opt, DTE, LTE, DTR, LTR, k, gamma)

        predicted_labels_tot.append(predicted_labels)
        scores_tot.append(scores)
        LTR_for_minDcf.append(LTE)


    actual_labels = np.concatenate(LTR_for_minDcf)
    #pred_labels =  np.concatenate(predicted_labels_tot)
    scores_actual = np.concatenate(scores_tot).flatten()

    min_DCF = compute_DCFmin(scores_actual, actual_labels, prior, cfn, cfp)

    return min_DCF


if __name__ == '__main__':
    file = "Train.txt"
    file2 = "Test.txt"
    D, L = loadFile(file)
    DTE, LTE = loadFile(file2)

    pi = 0.5
    Cfn = 1
    Cfp = 10
    pi_tilde = (pi * Cfn) / (pi * Cfn + (1 - pi) * Cfp)  # ci permette di avere costi uguali
    D_pca8, P = pca(D, 8)
    DTE_pca8 = np.dot(P.T, DTE)
    # hyperparameters
    C = 10
    K = 0
    gamma = 0.001
    l = 0.001
    #folds = 10


    '''
    # THIS IS AN EXAMPLE OF USING K-FOLD CROSS VALIDATION FOR THE HYPER PARAMETER TUNING
    
    min_dcf_list_g1_z = []
    min_dcf_list_g1_w = []
    min_dcf_list_g2_z = []
    min_dcf_list_g2_w = []

    for C_curr in C:
        min_dcf_curr_g1_z = K_fold(folds, DTR_pca8, LTR, C, K, pi_tilde, 1, 1, 0.001)
        min_dcf_curr_g1_w = K_fold(folds, DTR_w, LTR, C_curr, K, pi_tilde, 1, 1, 0.001)
        min_dcf_curr_g2_z = K_fold(folds, DTR_z, LTR, C_curr, K, pi_tilde, 1, 1, 0.01)
        min_dcf_curr_g2_w = K_fold(folds, DTR_w, LTR, C_curr, K, pi_tilde, 1, 1, 0.01)

        min_dcf_list_g1_z.append(min_dcf_curr_g1_z)
        min_dcf_list_g1_w.append(min_dcf_curr_g1_w)
        min_dcf_list_g2_z.append(min_dcf_curr_g2_z)
        min_dcf_list_g2_w.append(min_dcf_curr_g2_w)
                                 
    plt.semilogx(C, min_dcf_list_g1_z, color='black', label='RBF - NO PCA + Z norm, gamma=0.001:')
    plt.semilogx(C, min_dcf_list_g1_w, color='red', label='RBF - NO PCA + whitening, gamma=0.001:')
    plt.semilogx(C, min_dcf_list_g2_z, color='blue', label='RBF - NO PCA + Z norm, gamma=0.01:')
    plt.semilogx(C, min_dcf_list_g1_w, color='green', label='RBF - NO PCA + whitening, Z-norm, gamma=0.01:')
    plt.xlabel('C')
    plt.ylabel('min DCF')
    plt.legend()
    plt.show()
    '''

    # FINAL CODE => THE ONE USED ON THE EVALUATION TEST

    ### TRAINING SU DTR, LTR ###
    (DTR, LTR), (DTC, LTC) = split_db_2to1(D_pca8, L, seed=0) 
    # DTE_pca8 = np.dot(P.T, DTE)     # pca = 8 per il test set
    alpha_opt = train_RBF_SVM(DTR, LTR, C, K, gamma)

    ### TRAINING DEL CALIBRATORE SU DTC, LTC ###
    scores_train_cal, _ = evaluate_RBF_svm(alpha_opt, DTC, LTC, DTR, LTR, K, gamma)
    w_best, b_best = bin_logistic_regression_train(scores_train_cal, LTC, l, pi_tilde)

    ### EVALUATION, USANDO I PARAMETRI DEL TRAINING ###
    scores_UNCAL, _ = evaluate_RBF_svm(alpha_opt, DTE_pca8, LTE, DTR, LTR, K, gamma)
    predicted, _, scores_CAL = logistic_regression_eval(scores_UNCAL, LTE, w_best, b_best, pi_tilde)
    plot_Bayes_error(scores_CAL, LTE)

    confusion_matrix = compute_confusion_matrix(LTE, predicted.astype(int), 2)
    actual_dcf = compute_DCF(pi_tilde, 1, 1, confusion_matrix)
    min_dcf = compute_DCFmin(scores_CAL, LTE, pi_tilde, 1, 1)
    print("actual dcf: {}".format(actual_dcf))
    print("min dcf: {}".format(min_dcf))
