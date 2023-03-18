import pandas as pd
import numpy as np
import multiprocessing
import os
import warnings

warnings.filterwarnings("ignore")


class select_structure:

    def sort_t_stat(self, select_n1, res_path, pca_name, return_path, return_name, date, t_stat_path):
        '''

        :param select_n1: firstly keep the first n1 components
        :param res_path: pca outcome path
        :param pca_name: in ["demean","zscore","no_proces"]
        :param return_path: the group weighted return path
        :param return_name: name of the file
        :param date:
        :param t_stat_path: store t_stat of the components return in this path
        '''

        cdata = pd.read_csv(res_path + pca_name, index_col=[0, 1])
        components = cdata.iloc[:, 2:]
        returns = pd.read_csv(return_path + return_name, index_col=[0, 1, 2])
        components_return = pd.DataFrame(np.dot(np.array(components), np.array(returns)), columns=returns.columns)
        components_return_select_n1 = components_return.iloc[0:select_n1, :]
        t_stat = components_return_select_n1.apply(lambda x: x.mean() / x.std(), axis=1).to_frame()
        t_stat.columns = ['t_stat']
        t_stat['t_stat_abs'] = abs(t_stat['t_stat'])
        t_stat_sort = t_stat.sort_values(by='t_stat_abs', ascending=False)
        t_stat_sort.drop('t_stat_abs', inplace=True, axis=1)
        t_stat_sort.to_csv(t_stat_path + date + ".t_stat_sort.csv",
                           index_label=('components'))

    def get_components(self, res_path, res_name, t_stat_path, t_stat_name, date, select_component_path):
        '''

        :param select_component_path: store the selected n1 components multiply by their t_stat's sign
                                     (t_stat > 0: long ,else short)
        :return: the invest_ratio in every group in the selected components
        '''

        cdata = pd.read_csv(res_path + res_name, index_col=[0, 1])
        components = cdata.iloc[:, 2:]
        t_data = pd.read_csv(t_stat_path + t_stat_name)

        # get the sign of t_stat
        t_data['sign'] = 0
        t_data['sign'][t_data['t_stat'] >= 0] = 1
        t_data['sign'][t_data['t_stat'] < 0] = -1
        select_c = components.loc[:, list(t_data.components), :]
        change_sign_c = select_c.apply(lambda x: x * np.array(t_data['sign']))
        change_sign_c.to_csv(
            select_component_path + date + ".component_sort.csv",
            index_label=('Date', 'components'))

    def merge_stock_component(self, select_component_path, component_name, group_path, group_name, date,
                              merge_stock_component_path):
        '''

        merge the components with groups and the stocks in the group

        :param merge_stock_component_path: outcome store path
        '''

        scdata = pd.read_csv(select_component_path + component_name)
        # change the index to date（the last day of the future days return,that is the date in the filename
        scdata['Date'] = int(date)
        scdata = scdata.reset_index(drop=True).set_index(['Date', 'components'])
        scdata_c = scdata.stack().unstack(level=1).reindex(scdata.index.get_level_values(1).unique(), axis=1)

        gdata = pd.read_csv(group_path + group_name)

        # generate new columns in order to merge data
        gdata['f_g'] = gdata['features'] + "_" + gdata['group'].astype("str")
        gdata = gdata.reset_index(drop=True).set_index(['Date', 'f_g'])
        merge_data = pd.concat([gdata, scdata_c], join='outer', axis=1)
        merge_data = merge_data.droplevel(1)
        merge_data.drop(labels=['values'], axis=1, inplace=True)
        merge_data.to_csv(merge_stock_component_path + date + ".stock_component.csv", index_label=('Date'))

    def get_invest_ratio(self, merge_stock_component_path, mdata_name, select_n2, date, invest_ratio_path):
        '''

        according to the invest_ratio of the groups in every components(select n2 by t_stats),
        get to invest ratio in the stocks.

        :param select_n2: construct portfolio by the first n2 components according to the sorted abs(t_stat)
        :param date:
        :param invest_ratio_path: store outcome path
        :return:
        '''
        mdata = pd.read_csv(merge_stock_component_path + mdata_name, index_col=[0, 1])
        mdata.drop(['features', 'group'], inplace=True, axis=1)
        #select components
        select_data = mdata.iloc[:, :select_n2]
        #firstly add all the components ratio
        select_data['sum1'] = select_data.apply(lambda x: x.sum(), axis=1)
        #get every stock's ratio
        stock_invest = select_data.loc[:, ['sum1']].groupby(by=['Date','Ticker']).sum()
        #keep if ratio>=0
        stock_invest_long = stock_invest[stock_invest['sum1'] >= 0]
        stock_invest_long.columns = ['ratio']
        #turn the sum to 1
        stock_invest_ratio = stock_invest_long.apply(lambda x: x / x.sum())
        stock_invest_ratio.to_csv(invest_ratio_path + date + ".invest_ratio.by_" + str(select_n2) + "_components.csv",
                                  index_label=('Date', 'Ticker'))


