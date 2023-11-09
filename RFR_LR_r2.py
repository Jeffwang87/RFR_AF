import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import torch
import math
from tqdm import tqdm
import argparse


# Initialization 
parser = argparse.ArgumentParser(description='RFR')
parser.add_argument('--F_1', default=1, type=int,
                    metavar='N', help='Inital F1 setting (default: 1)')
parser.add_argument('--d', default=200, type=int,
                    metavar='N', help='Initial dimension setting (default: 200)')
parser.add_argument('--F_star', default=1, type=int,
                    metavar='N', help='Initial F star setting (default: 2)')
parser.add_argument('--psi_1', default=2, type=int,
                    metavar='N', help='Initial Psi 1 setting (default: 2)')
parser.add_argument('--psi_2', default=200, type=int,
                    metavar='N', help='Initial Psi 2 setting (default: 200)')
parser.add_argument('--tau', default=1, type=int,
                    metavar='N', help='Initial tau setting (default: 2)')
parser.add_argument('--lambda_i', default=0.1, type=float,
                    metavar='N', help='Initial lambda setting (default: 0.1)')

args = parser.parse_args()



F_1 = args.F_1
d = args.d
F_star = args.F_star
psi_2 = args.psi_2
tau = args.tau
alpha = 0
lambda_i = args.lambda_i
psi_1 = args.psi_1
n = math.ceil(d * psi_2)
N = math.ceil(psi_1 * d) 




def f(x, beta_1, F_star, G, tau):
    return x@beta_1 + (F_star/d) * ((x@G)@np.transpose(x)-np.sum(G.diagonal())) + tau*np.random.normal(size=(1,1)) 






def relu(x):
    return np.maximum(x, 0)

def der_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def AF(x, a, b, c):
    return a*(x**2) + b*x + c

def DerAF(x, a, b, c):
    return 2*a*x + b

def xavier(m, h):
    out = torch.Tensor(m, h).uniform_(-1, 1)*math.sqrt(1./h)
    return out.numpy()

def kaiming(m, h):
    out = torch.randn(m, h)*math.sqrt(2./m)
    return out.numpy()




def RFR_result(ratio):
    RandomFeatures = np.random.normal(size=(N, d))
    for i in range(N):
        norm = LA.norm(RandomFeatures[i, :])
        RandomFeatures[i, :] *= (math.sqrt(d)/norm)
    beta_1 = np.random.normal(size=(d))
    beta_1 = (F_1 * beta_1)/(LA.norm(beta_1))

    G  = np.random.normal(size=(d, d))
    x_train = np.random.normal(size=(n, d))
    y_train = np.zeros((n))
    for sample in range(n):
        norm = LA.norm(x_train[sample, :])
        x_train[sample, :] *= (math.sqrt(d)/norm)
        y_train_temp = f(x_train[sample, :], beta_1, F_star, G,tau)
        y_train[sample] = y_train_temp


    x_test = np.random.normal(size=(n, d))
    y_test = np.zeros((n))
    for sample in range(n):
        norm = LA.norm(x_test[sample, :])
        x_test[sample, :] *= (math.sqrt(d)/norm)
        y_test_temp = f(x_test[sample, :], beta_1, F_star, G, tau)
        y_test[sample] = y_test_temp

    lambda_t = lambda_i ** ratio
    lambda_effect = lambda_t*N*n/d 
    x_train_features = relu((x_train@np.transpose(RandomFeatures))/math.sqrt(d))
    clf_2 = Ridge(alpha=lambda_effect , solver='svd')
    clf_2.fit(x_train_features, y_train)
    x_test_features = relu((x_test@np.transpose(RandomFeatures))/math.sqrt(d))
    y_pred = clf_2.predict(x_test_features)


    sensitivity = (LA.norm(np.sum((der_relu((x_test@np.transpose(RandomFeatures))/math.sqrt(d))@np.diag(clf_2.coef_)@RandomFeatures)/math.sqrt(d),axis=0)/n))**2
    return (1 - alpha)*((LA.norm(y_pred-y_test)**2)/n) + alpha*(sensitivity)


def RFR_xavier_result(ratio):
    RandomFeatures = xavier(N, d)

    beta_1 = np.random.normal(size=(d))
    beta_1 = (F_1 * beta_1)/(LA.norm(beta_1))

    G  = np.random.normal(size=(d, d))
    x_train = np.random.normal(size=(n, d))
    y_train = np.zeros((n))
    for sample in range(n):
        norm = LA.norm(x_train[sample, :])
        x_train[sample, :] *= (math.sqrt(d)/norm)
        y_train_temp = f(x_train[sample, :], beta_1, F_star, G,tau)
        y_train[sample] = y_train_temp


    x_test = np.random.normal(size=(n, d))
    y_test = np.zeros((n))
    for sample in range(n):
        norm = LA.norm(x_test[sample, :])
        x_test[sample, :] *= (math.sqrt(d)/norm)
        y_test_temp = f(x_test[sample, :], beta_1, F_star, G, tau)
        y_test[sample] = y_test_temp

    lambda_t = lambda_i ** ratio
    lambda_effect = lambda_t*N*n/d 

    x_train_features = relu((x_train@np.transpose(RandomFeatures))/math.sqrt(d))
    clf_2 = Ridge(alpha=lambda_effect , solver='svd')
    clf_2.fit(x_train_features, y_train)

    x_test_features = relu((x_test@np.transpose(RandomFeatures))/math.sqrt(d))
    y_pred = clf_2.predict(x_test_features)

    sensitivity = (LA.norm(np.sum((der_relu((x_test@np.transpose(RandomFeatures))/math.sqrt(d))@np.diag(clf_2.coef_)@RandomFeatures)/math.sqrt(d),axis=0)/n))**2
    return (1 - alpha)*((LA.norm(y_pred-y_test)**2)/n) + alpha*(sensitivity)

def RFR_kaiming_result(ratio):
    RandomFeatures = kaiming(N, d)
    beta_1 = np.random.normal(size=(d))
    beta_1 = (F_1 * beta_1)/(LA.norm(beta_1))

    G  = np.random.normal(size=(d, d))
    x_train = np.random.normal(size=(n, d))
    y_train = np.zeros((n))
    for sample in range(n):
        norm = LA.norm(x_train[sample, :])
        x_train[sample, :] *= (math.sqrt(d)/norm)
        y_train_temp = f(x_train[sample, :], beta_1, F_star, G,tau)
        y_train[sample] = y_train_temp


    x_test = np.random.normal(size=(n, d))
    y_test = np.zeros((n))
    for sample in range(n):
        norm = LA.norm(x_test[sample, :])
        x_test[sample, :] *= (math.sqrt(d)/norm)
        y_test_temp = f(x_test[sample, :], beta_1, F_star, G, tau)
        y_test[sample] = y_test_temp

    lambda_t = lambda_i ** ratio
    lambda_effect = lambda_t*N*n/d 

    x_train_features = relu((x_train@np.transpose(RandomFeatures))/math.sqrt(d))
    clf_2 = Ridge(alpha=lambda_effect , solver='svd')
    clf_2.fit(x_train_features, y_train)

    x_test_features = relu((x_test@np.transpose(RandomFeatures))/math.sqrt(d))
    y_pred = clf_2.predict(x_test_features)

    sensitivity = (LA.norm(np.sum((der_relu((x_test@np.transpose(RandomFeatures))/math.sqrt(d))@np.diag(clf_2.coef_)@RandomFeatures)/math.sqrt(d),axis=0)/n))**2
    return (1 - alpha)*((LA.norm(y_pred-y_test)**2)/n) + alpha*(sensitivity)


# Main
if __name__ == "__main__":
    number = 60
    iteration = 1
    ratio = np.linspace(-4, 4, num=number)
    final_result = np.zeros(number)
    final_result_xavier = np.zeros(number)
    final_kaiming_result = np.zeros(number)
    
    for i in tqdm(range(iteration)):
        result_RFR = []
        result_xavier =[]
        result_kaiming = []
        for r in ratio:
            result_RFR.append(RFR_result(r))
            result_kaiming.append(RFR_xavier_result(r))
            result_xavier.append(RFR_kaiming_result(r))
        final_result += np.array(result_RFR)
        final_result_xavier += np.array(result_kaiming)
        final_kaiming_result += np.array(result_xavier)
    
    
    final_result = final_result/iteration
    final_result_xavier = final_result_xavier/iteration
    final_kaiming_result = final_kaiming_result/iteration
    
    
    
    
    plt.plot(ratio, final_result, marker='o', label= 'Original init')
    plt.plot(ratio, final_result_xavier, marker='o', label= 'Xavier init')
    plt.plot(ratio, final_kaiming_result,marker='o', label='Kaiming init')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 4)
    plt.legend()
    plt.savefig('./RFR_plot_case_regime_2.png')
