from hyperseti.dedoppler import plan_optimal, plan_stepped
import numpy as np

def test_plan_optimal():
    """ Test that plan_full() works as intended """ 
    assert len(plan_optimal(16, -500, 500)) == 1001
    assert len(plan_optimal(16, 0, 500)) == 501
    assert len(plan_optimal(16, -500, 0)) == 501
    assert np.allclose(plan_optimal(16, -500, 0), plan_optimal(16, 0, -500)[::-1])

def test_plan_stepped():
    """ Test that plan_stepped() works as intended """
    # Check generating lower -> 0 and upper -> 0 separately gives same
    # result as generating from lower -> upper
    ps_full  = plan_stepped(16, -500, 500)
    ps_lower = plan_stepped(16, -500, 0)
    ps_upper = plan_stepped(16, 0, 500)
    assert np.allclose(ps_lower, ps_full[:len(ps_lower)])
    assert np.allclose(ps_upper, ps_full[len(ps_lower)-1:])

    # Check that it works if non-zero starts are given
    ps_nonzero_start = plan_stepped(16, 100, 300)
    assert ps_nonzero_start[0] > 100
    assert ps_nonzero_start[-1] > 300

    assert ps_nonzero_start[0] == 112
    assert ps_nonzero_start[-1] == 480
    assert ps_nonzero_start.shape[0] == 32
    
    # Check that lower > upper gives reversed output
    assert np.allclose(plan_stepped(16, 0, -500), plan_stepped(16, -500, 0)[::-1])
    assert np.allclose(plan_stepped(16, 100, 500), plan_stepped(16, 500, 100)[::-1])
    
if __name__ == "__main__":
    test_plan_full()
    test_plan_stepped()
