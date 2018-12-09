# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:26:09 2016

@author: sepanta
"""
import numpy as np
import pandas as pd
    
from random import random
from bisect import bisect
from scipy.stats import mvn
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from scipy import stats
 
def get_linear_UF(data, user_col, data_col, rating_col):
    X = []
    y = []
    prevUser = '0'
    result = []
    for row in data:

        if prevUser != '0' and prevUser != row[user_col]: 
            result.append(np.linalg.lstsq(X, y)[0])
            X = []
            y = []
              
        tempR = []      
        for i in data_col:
            tempR.append(row[i])
        tempY =row[rating_col] 
        X.append(tempR)
        y.append(tempY)
        prevUser = row[user_col]
        
    result.append(np.linalg.lstsq(X, y)[0])
    
    return result

def get_linear_UF_csv(data, user_col, data_col, rating_col):
    X = []
    y = []
    prevUser = '0'
    result = []
    for r in data.iterrows():
        row = r[1]
        if prevUser != '0' and prevUser != row[user_col]: 
            result.append(np.linalg.lstsq(X, y)[0])
            X = []
            y = []
              
        tempR = []      
        for i in data_col:
            tempR.append(row[i])
        tempY =row[rating_col] 
        X.append(tempR)
        y.append(tempY)
        prevUser = row[user_col]
        
    result.append(np.linalg.lstsq(X, y)[0])
    
    return result

def weighted_choice(choices):
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random() * total
    i = bisect(cum_weights, x)
    return values[i]    
    
def get_sample(k, d, N):
    index = np.random.permutation(k)
    weights = [-1]*k
    weights_sum = 0
    for i in range(0, k - 1):
        weights[index[i]] = np.random.uniform(0, 1 - weights_sum)
        weights_sum += weights[index[i]]
        
    for i in range(k ):
        if weights[i] == -1:
            weights[i] = 1 - weights_sum
            break
    

    weight_choices = []
    for i in range(0, k):
        weight_choices.append((i, weights[i]))
    
    #centers = np.random.uniform(0, 1, (k, d))
    bounds =  np.random.uniform(0, 1, (k, d, 2))

    sample = []
    for i in range(N):
        cluster = weighted_choice(weight_choices)    
        x = []
        for j in range(d):
            if bounds[cluster][j][0] > bounds[cluster][j][1]:
                temp = bounds[cluster][j][1]
                bounds[cluster][j][1] = bounds[cluster][j][0]
                bounds[cluster][j][0] = temp
            x.append(np.random.uniform(bounds[cluster][j][0], bounds[cluster][j][1]))
            
        sample.append(x)
        
    return sample, bounds, weights

def get_gmm_prob(k, gmm, low, upp):
    prob = 0;
    for i in range(k):
        mu = gmm.means_[i]
        S = gmm.covars_[i]
        p,i = mvn.mvnun(low,upp,mu,S)
        prob += gmm.weights_[i] * p
    
    return prob

    
def get_gmm_error(sample, k, bounds, weights ):
    gmm = mixture.GMM(n_components=k, covariance_type='full')
    gmm.fit(sample)
    
    squareError = 0;
    for i in range(k):
        low, upp = zip(*bounds[i])
        prob = get_gmm_prob(k, gmm, low, upp)
        squareError += (prob - weights[i])**2
        
    return squareError/float(k)
    
def get_gmm_error_withLowUpp(sample, k, lower_bounds, upper_bounds, weights):
    gmm = mixture.GMM(n_components=k, covariance_type='full')
    gmm.fit(sample)
    
    squareError = 0;
    for i in range(k):
        low = lower_bounds[i]
        upp = upper_bounds[i]
        prob = get_gmm_prob(k, gmm, low, upp)
        squareError += (prob - weights[i])**2

        
    return squareError/float(k)
    
def get_covariance(sample, N, d):
    
    cov = np.zeros(shape = (d, d))
    for i in range(d):
        base = np.std(np.transpose(sample)[i])*4/float(N*(d + 2))
        exp = 2/float(d + 4)
        value = np.power(base, exp)
        cov[i][i] = value
    
    return cov

def get_kde_prob(sample, covar, N, low, upp):
    prob = 0;
    for i in range(N):
        mu = sample[i]
        p,i = mvn.mvnun(low,upp,mu,covar)
        prob += p
    
    return prob/float(N)

def get_kde_error_withLowUpp(sample, N, d, k, lower_bounds, upper_bounds, weights):
    covar = get_covariance(sample, N, d)
    
    squareError = 0;
    for i in range(k):
        low = lower_bounds[i]
        upp = upper_bounds[i]
        for j in range(len(low)):
            if low[j] > upp[j]:
                temp = low[j]
                low[j] = upp[j]
                upp[j] = temp
        prob = get_kde_prob(sample, covar, N, low, upp)
        squareError += (prob - weights[i])**2
        
    return squareError/float(k)
    
def get_gmm_kde_error(d, n, k, N):
    print "extracting linear utility functions"
    database = np.random.uniform(0, 1, (n, d))

    ratings, bounds, weights = get_sample(k, n, N)
    
    sample = []
    userId = 0
    for user in ratings:
        point_no = 0
        
        for rating in user:
            row = []
            row.append(userId)
            row.append(rating)
            for dim in range(d):
                row.append(database[point_no][dim])
            sample.append(row)
            point_no += 1                
                
        userId += 1
        
    lower_bounds_toMakeLinear = []
    upper_bounds_toMakeLinear = []
    
    userId = 0
    for cluster in bounds:
        point_no = 0
        
        for bound in cluster:
            row_lower = []
            row_lower.append(userId)
            row_lower.append(bound[0])
            row_upper = []
            row_upper.append(userId)
            row_upper.append(bound[1])
            for dim in range(d):
                row_lower.append(database[point_no][dim])
                row_upper.append(database[point_no][dim])
                
            lower_bounds_toMakeLinear.append(row_lower)
            upper_bounds_toMakeLinear.append(row_upper)
            point_no += 1                
                
        userId += 1
        
    linear_UF = get_linear_UF(sample, user_col=0, data_col = range(2, 2 + d), rating_col=1)
    upperbound_Linear = get_linear_UF(lower_bounds_toMakeLinear, user_col=0, data_col = range(2, 2 + d), rating_col=1)
    lowerbound_Linear = get_linear_UF(upper_bounds_toMakeLinear, user_col=0, data_col = range(2, 2 + d), rating_col=1)
    
    print "gettin gmm error"
    gmm_err = get_gmm_error_withLowUpp(linear_UF, k, lowerbound_Linear, upperbound_Linear, weights)
    print "gettin kde error"
    kde_err = get_kde_error_withLowUpp(linear_UF, N, d, k, lowerbound_Linear, upperbound_Linear, weights)
    
    return gmm_err, kde_err
    
def get_gmm_kde_error_linear(d, n, k, N):
    linear_UF, bounds, weights = get_sample(k, d, N)
    upperbound_Linear = []
    lowerbound_Linear = []
    for i in range(k):
        lower_bound = []
        upper_bound = []
        for dim in range(d):
            lower_bound.append(bounds[i][dim][0])
            upper_bound.append(bounds[i][dim][1])
        
        upperbound_Linear.append(upper_bound)
        lowerbound_Linear.append(lower_bound)
            
    
    gmm_err = get_gmm_error_withLowUpp(linear_UF, k, lowerbound_Linear, upperbound_Linear, weights)
    kde_err = get_kde_error_withLowUpp(linear_UF, N, d, k, lowerbound_Linear, upperbound_Linear, weights)
    
    return gmm_err, kde_err

def get_mean_error(d, n, k, N, iter_no):
    total_gmm_err = 0
    total_kde_err = 0

    for i in range(iter_no):
        gmm_err, kde_err = get_gmm_kde_error(d, n, k, N)
        total_gmm_err += gmm_err
        total_kde_err += kde_err
    
    return total_gmm_err/float(iter_no), total_kde_err/float(iter_no)

def get_mean_error_linear(d, n, k, N, iter_no):
    total_gmm_err = 0
    total_kde_err = 0

    for i in range(iter_no):
        print i
        gmm_err, kde_err = get_gmm_kde_error_linear(d, n, k, N)
        total_gmm_err += gmm_err
        total_kde_err += kde_err
    
    return total_gmm_err/float(iter_no), total_kde_err/float(iter_no)

if __name__ == '__main__':
    n=100
    k=10
    d = 3
    N=1000
    
    f = open('output.txt', 'a')
    f.write("General********************************************\n")
    f.write("Experiment on d--------------------------\n")
    f.close()
    for d in [2, 4, 6, 8, 10]:
        gmm_err, kde_err = get_mean_error(d, n, k, N, 10)
        s = "d = " + str(d) + " n = " + str(n) + " k = " + str(k) + " N = " + str(N) + " gmm_err = " + str(gmm_err) + " kde_err = " + str(kde_err) + "\n"
        print s
        f = open('output.txt', 'a')
        f.write(s)
        f.close()
        
    d = 3
       
    f = open('output.txt', 'a')
    f.write("Experiment on k--------------------------\n")
    f.close()
    for k in [2, 5, 8, 12, 15, 18, 21, 24]:
        print k
        gmm_err, kde_err = get_mean_error(d, n, k, N, 10)
        s = "d = " + str(d) + " n = " + str(n) + " k = " + str(k) + " N = " +str(N) +   " gmm_err = " + str(gmm_err) + " kde_err = " + str(kde_err) + "\n"
        print s
        f = open('output.txt', 'a')
        f.write(s)
        f.close()
        
    k = 10
       
    f = open('output.txt', 'a')
    f.write("Experiment on n--------------------------\n")
    f.close()
    for n in [1000]:
        gmm_err, kde_err = get_mean_error(d, n, k, N, 10)
        s = "d = " + str(d) + " n = " + str(n) + " k = " + str(k) + " N = " +str(N) +   " gmm_err = "  +  str(gmm_err) + " kde_err = " + str(kde_err) + "\n"
        print s
        f = open('output.txt', 'a')
        f.write(s)
        f.close()
    
    n = 100
    
    f = open('output.txt', 'a')
    f.write("Experiment on N--------------------------\n")
    f.close()
    for N in [100000]:
        gmm_err,     = get_mean_error(d, n, k, N, 10)
        s = "d = " + str(d) + " n = " + str(n) + " k = " + str(k) + " N = " + str(N) +" gmm_err = " +   str(gmm_err) + " kde_err = " + str(kde_err) + "\n"
        print s
        f = open('output.txt', 'a')
        f.write(s)
        f.close()
        
        
        
    n=100
    k=10
    d = 3
    N=1000
    
    f = open('output.txt', 'a')
    f.write("LINAEAR********************************************\n")
    f.write("Experiment on d--------------------------\n")
    f.close()
    for d in [2, 4, 6, 8, 10]:
        gmm_err, kde_err = get_mean_error_linear(d, n, k, N, 10)
        s = "d = " + str(d) + " n = " + str(n) + " k = " + str(k) + " N = " + str(N) + " gmm_err = " + str(gmm_err) + " kde_err = " + str(kde_err) + "\n"
        print s
        f = open('output.txt', 'a')
        f.write(s)
        f.close()
        
    d = 3
       
    f = open('output.txt', 'a')
    f.write("Experiment on k--------------------------\n")
    f.close()
    for k in [2, 5, 8, 12, 15, 18, 21, 24]:
        print k
        gmm_err, kde_err = get_mean_error_linear(d, n, k, N, 10)
        s = "d = " + str(d) + " n = " + str(n) + " k = " + str(k) + " N = " +str(N) +   " gmm_err = " + str(gmm_err) + " kde_err = " + str(kde_err) + "\n"
        print s
        f = open('output.txt', 'a')
        f.write(s)
        f.close()
        
    k = 10
       
    f = open('output.txt', 'a')
    f.write("Experiment on n--------------------------\n")
    f.close()
    for n in [1000]:
        gmm_err, kde_err = get_mean_error_linear(d, n, k, N, 10)
        s = "d = " + str(d) + " n = " + str(n) + " k = " + str(k) + " N = " +str(N) +   " gmm_err = "  +  str(gmm_err) + " kde_err = " + str(kde_err) + "\n"
        print s
        f = open('output.txt', 'a')
        f.write(s)
        f.close()
    
    n = 100
    
    f = open('output.txt', 'a')
    f.write("Experiment on N--------------------------\n")
    f.close()
    for N in [100000]:
        gmm_err,     = get_mean_error_linear(d, n, k, N, 10)
        s = "d = " + str(d) + " n = " + str(n) + " k = " + str(k) + " N = " + str(N) +" gmm_err = " +   str(gmm_err) + " kde_err = " + str(kde_err) + "\n"
        print s
        f = open('output.txt', 'a')
        f.write(s)
        f.close()





    
    
    