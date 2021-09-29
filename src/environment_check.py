class EnvironmentCheck:
    import matplotlib
    import numpy as np
    import pandas as pd
    
    def checkMatPlotVers():
        print(matplotlib.__version__)

    def checkNPVers():
        print(np.__version__)
    
    def checkPandasVers():
        print(pd.__version__)

    