import pandas as pd
import os


path = "C:/Data"

# store_path
group_path = "./data/group/"
future_days_path = "./data/future days/"
return_path = "./data/return/"

'''
This file is used to group the stock by features.
'''

class Data_processing:

    def __init__(self, path, feature_dir = "/FC_BarraModel_V1_S/", trading_date_dir = "/Calendar/",
                 stock_dir = "/StockUniverse/"):
        self.path = path
        self.featurespath = self.path + feature_dir
        self.trading_date_path = self.path + trading_date_dir
        self.stockspath = self.path + stock_dir

    def process_features_data(self, group_path, group_num):
        '''

        :param group_path: output store path
        :param group_num: divided into group_num groups by feature values
        :return:
        '''
        filelist = os.listdir(self.featurespath)
        for filename in filelist:
            partlist = filename.split(".")
            if filename.endswith(".csv"):
                date = partlist[0]
                features_data = pd.read_csv(self.featurespath + filename, index_col = 0)
                # add Date column
                features_data['Date'] = date
                features_data.drop(labels = ['INDUSTRY', 'WEIGHT'], axis = 1, inplace = True)
                sort_data = features_data.sort_values(by = ['Date', 'Ticker']).reset_index().set_index(
                    ['Date', 'Ticker'])
                # get groups
                res = self.sort_to_groups(sort_data, group_num)
                res.to_csv(group_path + date + ".group.csv")

    def sort_to_groups(self, features_data, group_num):
        '''

        :param features_data: Dataframe with index ['Date', 'Ticker']
        :param group_num: divided into group_num groups
        :return: Dataframe with group labels every trading date
        '''

        features_data_stack = features_data.stack().reset_index().set_index('Date')
        features_data_stack.columns = ['Ticker', 'features', 'values']
        sort_data = features_data_stack.groupby(['Date', 'features']).apply(lambda x: x.sort_values(by = 'values'))
        # add group labels
        sort_data['group'] = (sort_data.groupby(['Date', 'features']).apply(
            lambda x: pd.qcut(x['values'], group_num, labels = range(group_num)))).values

        return sort_data

    def get_responding_future_date(self, group_path, future_days_path, future_days, filelist):
        '''
        get futures trading dates

        :param future_days_path: output store path
        :param future_days: of future days from T+2
        :param filelist:the file name list under group_path
        '''

        trading_date = (pd.read_csv(self.trading_date_path + "TradeDate.csv")).astype('str')
        # last day in all the data
        last_day = filelist[-1].split(".")[0]
        for filename1 in filelist:
            partlist = filename1.split(".")
            if filename1.endswith(".csv"):
                # get the date
                date = partlist[0]
                group_data = pd.read_csv(group_path + filename1, index_col = ['Date'])

                # generate a dataframe to store the future days return
                future_dates = pd.DataFrame(index = [int(date)], columns = range(future_days))
                f_day_index = list(trading_date.values).index(date) + 2 + future_days

                # get the end date of the future days
                f_day = list(trading_date.values)[f_day_index - 1][0]

                if f_day <= last_day:
                    future_dates.iloc[0, :] = \
                        (trading_date.iloc[list(trading_date.values).index(date) + 2:f_day_index]).T.values[0]

                    future_dates = future_dates.stack().to_frame().droplevel(1)
                    future_dates.columns = ['future_date']
                    data = pd.merge(group_data.drop(labels = 'values', axis = 1), future_dates, how = "left",
                                    left_index = True,
                                    right_index = True)
                    data.to_csv(future_days_path + f_day + ".future_" + str(future_days) + "days.csv")

    def get_group_returns(self, return_path, future_days_path, future_day_list, future_days):

        stocklist = os.listdir(self.stockspath)
        stockname_end = stocklist[0].split(".")[1:]
        for filename1 in future_day_list:
            partlist = filename1.split(".")
            if filename1.endswith(".csv"):
                date = partlist[0]
                future_data = pd.read_csv(future_days_path + filename1, index_col = 0)
                future_data.index.name = 'Date'
                daylist = list(future_data['future_date'].astype("str").unique())
                stockdata = pd.DataFrame()
                for day in daylist:
                    name = self.stockspath + day + "." + ".".join(stockname_end)
                    stockdata = pd.concat([stockdata, pd.read_csv(name, usecols = ['Date', 'Ticker', 'Return'],
                                                                  index_col = ['Date', 'Ticker'])])
                group_future_days_return = pd.merge(future_data, stockdata, how = 'left',
                                                    left_on = ['future_date', 'Ticker'], right_index = True)

                group_future_days_return.fillna(0, inplace = True)

                #simply give same weight of the stocks in each group.
                group_future_days_return = group_future_days_return.groupby(
                    ['Date', 'future_date', 'features', 'group']).apply(
                    lambda x: x['Return'].mean()).to_frame().unstack(level = 1)

                group_future_days_return.columns = group_future_days_return.columns.droplevel(0)
                group_future_days_return.to_csv(return_path + date + "." + str(future_days) + "_days.group_return.csv")


