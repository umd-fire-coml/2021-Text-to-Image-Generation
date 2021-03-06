from src.environment_check import EnvironmentCheck

def test_mat_plot_vers():
    assert(EnvironmentCheck.checkMatPlotVers() == '3.4.3')

def test_pandas_vers():
    assert(EnvironmentCheck.checkPandasVers() == '1.3.3')

def test_np_vers():
    assert(EnvironmentCheck.checkNPVers() == '1.21.2') 