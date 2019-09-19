'''
This module contains three functions (naive, partially-vectorized and fully-vectorized)
computing the loss and the gradients. 
'''

import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with two loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss variable and the gradient in dW. If you are not    #
    # careful here, it is easy to run into numeric instability. Don't forget    #
    # the regularization both in loss and in the gradient computations!         #
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    pass ## Write your code here
    # compute the loss and the gradient
    num_classes = W.shape[1] # 10
    num_train = X.shape[0]  # 500
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        exp_scores = np.exp(scores)
        correct_exp_score = exp_scores[y[i]]
        loss += -np.log(correct_exp_score / np.sum(exp_scores))
        
        ## Update dW here
        Prob_each_score = exp_scores / np.sum(exp_scores)
        for j in range(num_classes):
            if j == y[i]:
                dW[:,j] += (Prob_each_score[j] - 1) * X[i]
            else:
                dW[:,j] += Prob_each_score[j] * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    ## Average dW here
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    ## Add the contribution of regularization to dW
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

def softmax_loss_partially_vectorized(W, X, y, reg):
    """
    Softmax loss function, partially vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using one explicit loop.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # the regularization both in loss and in the gradient computations!         #
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    pass ## Write your code here
    # compute the loss and the gradient
    num_classes = W.shape[1] # 10
    num_train = X.shape[0]  # 500
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        exp_scores = np.exp(scores)
        correct_exp_score = exp_scores[y[i]]
        loss += -np.log(correct_exp_score / np.sum(exp_scores))
        
        ## Update dW here
        Prob_each_score = exp_scores / np.sum(exp_scores)
        Prob_each_score[y[i]] -= 1;
        
        temp = np.dot( np.diag(Prob_each_score), np.tile(X[i,:],(num_classes,1)) )
        dW += temp.T
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    
    loss /= num_train
    ## Average dW here
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    ## Add the contribution of regularization to dW
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # the regularization both in loss and in the gradient computations!         #
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    pass ## Write your code here
    # compute the loss and the gradient
    num_classes = W.shape[1] # 10
    num_train = X.shape[0]  # 500
    
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    Prob_each_score = exp_scores / np.sum(exp_scores, axis = 1).reshape(-1,1)
    loss = np.sum(-np.log( Prob_each_score[np.arange(num_train), y] ))
        
    ## Update dW here
    Prob_each_score[np.arange(num_train), y] -= 1
    dW = (X.T).dot(Prob_each_score)
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    
    # average
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################


    return loss, dW

