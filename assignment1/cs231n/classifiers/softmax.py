import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    train_num = X.shape[0]
    class_num = W.shape[1]

    for i in range(train_num):
        label_index = y[i]
        now_x = X[i]

        scores = np.dot(now_x, W)
        exp_scores = np.exp(scores)
        exp_scores_sum = np.sum(exp_scores)

        assert (scores.shape == (class_num,))
        assert (exp_scores.shape == (class_num,))
        assert (np.isscalar(exp_scores_sum))

        probs = exp_scores / exp_scores_sum
        label_prob = probs[label_index]

        assert (probs.shape == (class_num,))
        assert (np.isscalar(label_prob))

        now_loss = -np.log(label_prob)
        loss += now_loss

        for j in range(class_num):
            if j == label_index:
                dW[:, j] = (1 - exp_scores[j]) * now_x
            else:
                dW[:, j] = exp_scores[j] * now_x

    loss = loss / float(train_num) + 0.5 * reg * np.sum(W * W)
    dW = dW / float(train_num) + reg * W

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
    # regularization!                                                           #
    #############################################################################
    train_num = X.shape[0]
    class_num = W.shape[1]

    scores = np.dot(X, W)
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores, axis=1)

    assert (scores.shape == (train_num, class_num))
    assert (exp_scores.shape == (train_num, class_num))
    assert (exp_scores_sum.shape == (train_num, ))

    probs = exp_scores / exp_scores_sum[:, np.newaxis]
    label_probs = np.choose(y, probs.T)

    assert (probs.shape == (train_num, class_num))
    assert (label_probs.shape == (train_num,))

    loss = np.sum(-np.log(label_probs))
    loss = loss / float(train_num) + 0.5 * reg * np.sum(W * W)

    probs_copy = probs.copy()
    probs_copy[np.arange(train_num), y] -= 1.0
    probs_copy /= float(train_num)

    dW = np.dot(X.T, probs_copy) / + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
