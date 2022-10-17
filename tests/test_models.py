"""Tests for statistics functions within the Model layer."""
import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import patch

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    # NB: the comment 'yapf: disable' disables automatic formatting using
    # a tool called 'yapf' which we have used when creating this project
    test_array = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([0, 0]), daily_mean(test_array))


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_array = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([3, 4]), daily_mean(test_array))

@patch('inflammation.models.get_data_dir', return_value='/data_dir')
def test_load_csv(mock_get_data_dir):
    from inflammation.models import load_csv
    with patch('numpy.loadtxt') as mock_loadtxt:
        load_csv('test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[0]
        assert kwargs['fname'] == '/data_dir/test.csv'
        load_csv('/test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[1]
        assert kwargs['fname'] == '/test.csv'

# TODO(lesson-automatic) Implement tests for the other statistical functions
@pytest.mark.parametrize("test, expected", [ ([[4, 2, 8],
                           [3, 4, 9],
                           [5, 6, 2]], [5, 6, 9]), ([[1,5,8], [4,2,6], [5,9,2]], [5,9,8]),]) 
def test_daily_max(test, expected):
    """Test that the max function works with an array of positive integers."""
    from inflammation.models import daily_max
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array(expected), daily_max(np.array(test)))



@pytest.mark.parametrize("test, expected", [ ([[-4, 2, 8],
                           [3, 4, -9],
                           [5, 6, -2]], [-4, 2, -9]), ([[1,-5,8], [4,2,-6], [-5,9,2]], [-5,-5,-6]),]) 
def test_daily_min(test, expected):
    """Test that the min function works with an array of positive and negative integers."""
    from inflammation.models import daily_min
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array(expected), daily_min(np.array(test)))


# TODO(lesson-mocking) Implement a unit test for the load_csv function
