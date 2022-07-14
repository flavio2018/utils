import matplotlib.pyplot as plt


class GenericPlot:
    def __init__(self, cfg, dataframe):
        self.path = cfg.run.project_path + '/reports/figures/'
        self.dataframe = dataframe
        
    def _plot(self):
        pass
    
    def plot(self, filename):
        self._plot()
        plt.savefig(self.path + filename + ".pdf", format="pdf")
