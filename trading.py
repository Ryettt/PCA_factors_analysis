import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


class Trading:
    def get_portfolio_return(self, invest_ratio_path, inv_r_name, stockspath, stock_name_end, trading_date_df,
                             end_date):
        '''

        get portfolio return according to the invest ratio of the stocks

        :param trading_date_df: trading date dataframe
        :param end_date: end date of the backtest period

        '''

        inve_ratio = pd.read_csv(invest_ratio_path + inv_r_name)
        date = int(inv_r_name.split(".")[0])

        # get the T+2 date return(measured in bp)
        t2_date = str(list(trading_date_df.values)[list(trading_date_df.values).index(date) + 2][0])
        if t2_date <= end_date:
            stock_name = t2_date + "." + stock_name_end
            return_t2 = pd.read_csv(stockspath + stock_name, usecols=['Date', 'Ticker', 'Return'])
            return_t2.columns = ['T+2 Date', 'Ticker', 'Return']

            return_t2_p = pd.merge(inve_ratio, return_t2, how='left', left_on=['Ticker'], right_on=['Ticker'])
            # turn to no unit return
            portfolio_return = pd.DataFrame(index=[t2_date], columns=['return'],
                                            data=(return_t2_p['ratio'] * return_t2_p['Return']).sum() * 0.0001)
            return portfolio_return

    def get_nav(self, all_return, name):

        '''

        get the portfolio nav of the constructed portfolio return
        :return:
        '''

        self.all_return = all_return
        self.all_return['nav'] = (1 + self.all_return['return']).cumprod()
        plt.plot(all_return.loc[:, ['nav']])
        plt.title(name)
        plt.show()
        return self.all_return

    def calculate_turnover(self, invest_ratio_path, inv_r_name1, trading_date_df, end_date):
        '''
        get daily turnover
        '''

        inve_ratio1 = pd.read_csv(invest_ratio_path + inv_r_name1, usecols=[1, 2], index_col=[0])
        inve_ratio1.columns = ['ratio1']
        date = int(inv_r_name1.split(".")[0])
        inv_ratio_name_end = ".".join(inv_r_name1.split(".")[1:])
        next_date = str(list(trading_date_df.values)[list(trading_date_df.values).index(date) + 1][0])
        if next_date <= end_date:
            inv_r_name2 = next_date + "." + inv_ratio_name_end
            inve_ratio2 = pd.read_csv(invest_ratio_path + inv_r_name2, usecols=[1, 2], index_col=[0])
            inve_ratio2.columns = ['ratio2']
            all_ratio = pd.concat([inve_ratio1, inve_ratio2], join='outer', axis=1).fillna(0)
            uni_turnover = abs(all_ratio['ratio2'] - all_ratio['ratio1']).sum() / 2
            turnover_df = pd.DataFrame(index=[next_date], data=uni_turnover, columns=['uni_turnover'])
            return turnover_df

    def calculate_max_draw(self):
        self.all_return['cummax'] = self.all_return['nav'].expanding().max()
        draw = (self.all_return['cummax'] - self.all_return['nav']) / self.all_return['cummax']
        max_draw = draw.max()
        return max_draw

    def calculate_sharpe(self):
        sharpe = self.all_return['return'].mean() / (self.all_return['return'].std()) * np.sqrt(252)
        return sharpe

    def calculate_annual_return(self):
        anu_r = self.all_return['return'].mean() * 252
        return anu_r

    def get_alpha(self, portfoilo_return, index_path):
        '''

        alpha= portfolio return - return of CSI500

        :param portfoilo_return: dataframe of the return with index date

        :return:
        '''

        # generate a new_col
        portfoilo_return['CSI500_return'] = 0
        index_name_end = ".".join(os.listdir(index_path)[0].split(".")[1:])
        for dateint in list(portfoilo_return.index):
            date = str(dateint)
            index_name = date + "." + index_name_end
            data = pd.read_csv(index_path + index_name, index_col=[1])
            # get zz500
            portfoilo_return.loc[dateint, 'CSI500_return'] = data.loc[905, 'Return']

        portfoilo_return['CSI500_return'] = portfoilo_return['CSI500_return'] * 0.0001

        portfoilo_return['alpha'] = (portfoilo_return['return'] - portfoilo_return['CSI500_return'])

        portfoilo_return['alpha_nav'] = (1 + portfoilo_return['alpha']).cumprod()

        return portfoilo_return


