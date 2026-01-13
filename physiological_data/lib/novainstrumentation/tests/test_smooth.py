from numpy import arange
from numpy.testing import assert_allclose, run_module_suite

from novainstrumentation import smooth


def test_symmetric_window():
    x = arange(1., 11.)
    sx = smooth(x, window_len=5, window='flat')
    assert_allclose(x, sx)

if __name__ == "__main__":
    run_module_suite()
