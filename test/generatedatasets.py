#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy

# generate datasets
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    numpy.random.seed(0)
    C = numpy.array([[0., -0.23], [0.83, .23]])
    X = numpy.r_[numpy.dot(numpy.random.randn(n, dim), C),
              numpy.dot(numpy.random.randn(n, dim), C) + numpy.array([1, 1])]
    y = numpy.hstack((numpy.zeros(n), numpy.ones(n)))
    return X, y
