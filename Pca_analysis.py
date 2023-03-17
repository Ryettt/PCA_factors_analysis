from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import multiprocessing


class PCA_stock:
    def __init__(self, data, future_days):
        '''

        :param data: a dataframe of one trading date with index['Date', 'features', 'group'],and its values are return
        '''

        self.data = data
        self.date = data.columns[-1]
        self.future_days = future_days

    def get_pca_obj_zscore(self):
        normalize_data = self.data.apply(lambda x: (x - x.mean()) / x.std())
        self.pca_obj = PCA().fit(np.array(normalize_data.T))
        return self.pca_obj

    def get_pca_obj_demean(self):
        normalize_data = self.data.apply(lambda x: x - x.mean())
        self.pca_obj = PCA().fit(np.array(normalize_data.T))
        return self.pca_obj

    def get_pca_no_process(self):
        self.pca_obj = PCA().fit(np.array((self.data).T))
        return self.pca_obj

    def get_results_zscore(self, res_path):
        self.get_pca_obj_zscore()
        df1 = self.get_components()
        df2 = self.get_variance()
        res = pd.concat([df2, df1], axis=1)
        res.to_csv(res_path + r"zscore/" + str(self.date) + "." + str(future_days) + "_days_zscore.csv",
                   index_label=('Date', 'component'))

        return res

    def get_results_demean(self, res_path):
        self.get_pca_obj_demean()
        df1 = self.get_components()
        df2 = self.get_variance()
        res = pd.concat([df2, df1], axis=1)
        res.to_csv(res_path + r"demean/" + str(self.date) + "." + str(future_days) + "_days_demean.csv",
                   index_label=('Date', 'component'))

        return res

    def get_results_no_process(self, res_path):
        self.get_pca_no_process()
        df1 = self.get_components()
        df2 = self.get_variance()
        res = pd.concat([df2, df1], axis=1)
        res.to_csv(res_path + r"no_process/" + str(self.date) + "." + str(future_days) + "_days_no_process.csv",
                   index_label=('Date', 'component'))

        return res

    def get_components(self):
        components = pd.DataFrame(index=self.data.index, columns=range(len(self.data.columns), ),
                                  data=(self.pca_obj.components_).T).stack().unstack(level=[1, 2])
        components.columns = components.columns._get_level_values(0) + "_" + components.columns._get_level_values(
            1).astype(str)
        components.index.name = ('Date', 'component')
        # components = components.reset_index(drop=False).set_index(['Date'])
        # components.columns = ['component'].append(components.columns[1:])

        return components

    def get_variance(self):
        res = pd.DataFrame(index=['Date', 'proportion', 'EV'], columns=range(len(self.data.columns)))
        res.loc['Date'] = self.data.index.levels[0][0]
        res.loc['EV'] = self.pca_obj.explained_variance_
        res.loc['proportion'] = self.pca_obj.explained_variance_ratio_

        res = res.T
        res = res.reset_index(drop=False).set_index(['Date', 'index'])

        res.index.name = ('Date', 'component')
        return res


