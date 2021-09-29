from src.environment_check import EnvironmentCheck


def test_pandas_vers():
    assert(EnvironmentCheck.checkPandasVers() == '1.2.4')

def test_np_vers():
    assert(EnvironmentCheck.checkNPVers() == '1.20.1') 