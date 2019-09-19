import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # 10
    num_train = X.shape[0]  # 500
    loss = 0.0
    dscores = np.zeros([num_train, num_classes])
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                ## Update dW here
                dW[:,j] += X[i,:] 
                dW[:,y[i]] -= X[i,:] 

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
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss. Make sure to add regularization.                          #
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################    
    pass ## Write your code here
    num_classes = W.shape[1] # 10
    num_train = X.shape[0]  # 500
    
    ## forward pass
    scores = X.dot(W)
    # create the correct score matrix of the same size as score matrix
    correct_class_scores = np.tile(scores[np.arange(num_train), y], (num_classes, 1)).T
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins)
    
    # average
    loss /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW. Make sure to add regularization.          #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################    
    pass ## Write your code here
    ## backward pass
    dscores = np.ones(scores.shape)
    dscores[margins <= 0] = 0
    dscores[np.arange(num_train), y] = -np.sum(dscores, axis=1).T
    dW = np.dot(X.T, dscores)
    
    dW /= num_train
    dW += reg * 2 * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
