from .DSVGD import DSVGD
from .FedAvg import FedAvg
from .DSGLD import DSGLD

"""
We borrow the code for the competing methods in Logistic Regression from Kassab and Simeone (2022).
Their files, besides the current, are all contained within this folder.
We have not modified their code besides simple bug fixes that occured due to incompatible versions of 
python libraries, or to set the seeds and initialisations to be equal to our method.

Their code can be found publicly available at: https://github.com/kclip/DSVGD.

"""

class Methods():

    def run_competing_methods(self):
        
        print("Running Distributed Stein Variational Gradient Descent (Kassab and Simeone, 2022):")
        ret_DSVGD = DSVGD().run()

        print("Running Federated Averaging (McMahan et al., 2017):")
        ret_FedAvg = FedAvg().run()

        print("Running Distributed Stochastic Gradient Langevin Dynamics (Ahn et al., 2014):")
        ret_DSGLD = DSGLD().run()

        return {
            "DSVGD": ret_DSVGD,
            "FedAvg": ret_FedAvg,
            "DSGLD": ret_DSGLD,
        }