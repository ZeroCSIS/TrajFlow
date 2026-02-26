# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from nose.tools import raises, timed, eq_
from jismesh import utils as ju

def _data_scalar():
    return [
        ({'meshcode': 5339, 'lat_multiplier': 0, 'lon_multiplier': 0},                          (35+1/3, 139)),
        ({'meshcode': 53391, 'lat_multiplier': 0, 'lon_multiplier': 0},                         (35+1/3, 139)),
        ({'meshcode': 5339115, 'lat_multiplier': 0, 'lon_multiplier': 0},                       (35+1/3, 139)),
        ({'meshcode': 5339007, 'lat_multiplier': 0, 'lon_multiplier': 0},                       (35+1/3, 139)),
        ({'meshcode': 533900, 'lat_multiplier': 0, 'lon_multiplier': 0},                        (35+1/3, 139)),
        ({'meshcode': 5339006, 'lat_multiplier': 0, 'lon_multiplier': 0},                       (35+1/3, 139)),
        ({'meshcode': 5339001, 'lat_multiplier': 0, 'lon_multiplier': 0},                       (35+1/3, 139)),
        ({'meshcode': 533900617, 'lat_multiplier': 0, 'lon_multiplier': 0},                     (35+1/3, 139)),
        ({'meshcode': 533900116, 'lat_multiplier': 0, 'lon_multiplier': 0},                     (35+1/3, 139)),
        ({'meshcode': 533900005, 'lat_multiplier': 0, 'lon_multiplier': 0},                     (35+1/3, 139)),
        ({'meshcode': 53390000, 'lat_multiplier': 0, 'lon_multiplier': 0},                      (35+1/3, 139)),
        ({'meshcode': 533900001, 'lat_multiplier': 0, 'lon_multiplier': 0},                     (35+1/3, 139)),
        ({'meshcode': 5339000011, 'lat_multiplier': 0, 'lon_multiplier': 0},                    (35+1/3, 139)),
        ({'meshcode': 53390000111, 'lat_multiplier': 0, 'lon_multiplier': 0},                   (35+1/3, 139)),
        ({'meshcode': 53393599212, 'lat_multiplier': 0.5, 'lon_multiplier': 0.5},               (35.6588542, 139.74609375)),
    ]

def _data_vector(num_elements=10):
    return [
        ({'meshcode': np.array([53390000111]*num_elements), 'lat_multiplier': 0, 'lon_multiplier': 0},                   (np.array([35+1/3]*num_elements), np.array([139]*num_elements))),
    ]

# テストヘルパ
def _eq_scalars(actual, expect, input):
    np.testing.assert_almost_equal(actual, expect, err_msg='{} as {} but expectes {} as {} when {} is given.'.format(actual, type(actual), expect, type(expect), input), decimal=7)

def _eq_vectors(actual, expect, input):
    np.testing.assert_almost_equal(actual, expect, err_msg='{} as {} but expectes {} as {} when {} is given.'.format(actual, type(actual), expect, type(expect), input), decimal=7)

def _eq_test_helper(expect, input, eq, target=ju.to_meshpoint):
    actual = target(**input)
    eq(actual, expect, input)

def test_normal_scalar_default():
    for input, expect in _data_scalar():
        _eq_test_helper(expect, input, eq=_eq_scalars)

def test_normal_vector_default():
    for input, expect in _data_vector():
        _eq_test_helper(expect, input, eq=_eq_vectors)

@timed(4)
def test_performance_vector():
    for input, expect in _data_vector(num_elements=1000000):
        _eq_test_helper(expect, input, eq=_eq_vectors)
