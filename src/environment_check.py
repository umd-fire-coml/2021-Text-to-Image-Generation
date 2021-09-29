import matplotlib
import numpy as np
import pandas as pd
class EnvironmentCheck:
    
    def checkMatPlotVers():
        return matplotlib.__version__

    def checkNPVers():
        return np.__version__
    
    def checkPandasVers():
        return pd.__version__   