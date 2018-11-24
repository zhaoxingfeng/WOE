#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
@Time: 2018/11/24 17:34
@Author: zhaoxingfeng
@Function：Weight of Evidence,根据iv值最大思想求最优分箱
@Version: V1.1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
pd.set_option('precision', 4)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


class Tree(object):
    def __init__(self):
        self.split_value = None
        self.sub_sample_cnt = None
        self.sub_sample_bad_cnt = None
        self.sub_sample_good_cnt = None
        self.iv = None
        self.woe = None
        self.tree_left = None
        self.tree_right = None

    # tree structure by JSON format
    def describe_tree(self):
        if not self.tree_left or not self.tree_right:
            tree_node = "{iv:" + str(self.iv) + \
                        ",woe:" + str(self.woe) + \
                        ",sub_sample_cnt:" + str(self.sub_sample_cnt) + \
                        ",sub_sample_bad_cnt:" + str(self.sub_sample_bad_cnt) + \
                        ",sub_sample_good_cnt:" + str(self.sub_sample_good_cnt) + "}"
            return tree_node
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_value:" + str(self.split_value) + \
                         ",sub_sample_cnt:" + str(self.sub_sample_cnt) + \
                         ",sub_sample_bad_cnt:" + str(self.sub_sample_bad_cnt) + \
                         ",sub_sample_good_cnt:" + str(self.sub_sample_good_cnt) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure

    def format_tree(self, tree, woe_list, split_value_list):
        if tree.split_value == None:
            tree_node = {"iv": tree.iv,
                         "woe": tree.woe,
                         "sub_sample_cnt": tree.sub_sample_cnt,
                         "sub_sample_bad_cnt": tree.sub_sample_bad_cnt,
                         "sub_sample_good_cnt": tree.sub_sample_good_cnt}
            woe_list.append(tree_node)
            return woe_list, split_value_list
        self.format_tree(tree.tree_left, woe_list, split_value_list)
        split_value = tree.split_value
        split_value_list.append(split_value)
        self.format_tree(tree.tree_right, woe_list, split_value_list)
        return woe_list, split_value_list


class WoeFeatureProcess(object):
    def __init__(self, path_conf, path_woe, min_sample_rate=0.1):
        """
        :param path_conf: The config describe the features which you want to transfrom and features's dtype.
            config structure:
                is_continuous: identify whether the feature is continuous(1) or discrete(0).
                is_identify: whether you want to transoform this feature.
                var_dtype: dtype, float, string.
                var_name: feature name.
        :param path_woe: The path to preserve woe rule.
        :param min_sample_rate: The percentile of minimum samples required to be at a bin.
        """
        self.dataset = None
        self.conf = pd.read_csv(path_conf)
        self.continuous_var_list = self.conf[(self.conf['is_continuous'] == 1) & (self.conf['is_identify'] != 1)]['var_name']
        self.woe_rule_dict = dict()
        self.woe_rule_df = pd.DataFrame()
        self.woe_rule_path = path_woe
        self.min_sample_rate = min_sample_rate
        self.total_bad_cnt = 0
        self.total_good_cnt = 0
        self.min_sample = 0

    def fit(self, dataset):
        self.dataset = dataset
        self.total_bad_cnt = dataset[dataset['label'] == 1].__len__()
        self.total_good_cnt = dataset[dataset['label'] == 0].__len__()
        self.min_sample = int(len(self.dataset) * self.min_sample_rate)

        for var in self.continuous_var_list:
            if var in self.dataset.columns:
                var_tree, var_df = self.fit_continuous(self.dataset[[var, 'label']], var)
                self.woe_rule_df = var_df if self.woe_rule_df.empty else \
                    pd.concat([self.woe_rule_df, var_df], ignore_index=1, sort=False)
        self.woe_rule_df = self.woe_rule_df.sort_values(by=['var_name', 'split_left']).reset_index(drop=True)
        self.woe_rule_df.to_csv(self.woe_rule_path, index=None, float_format="%.4f")

        for var, grp in self.woe_rule_df.groupby(['var_name']):
            self.woe_rule_dict[var] = list(zip(grp.split_right, grp.woe))

    #
    def fit_continuous(self, dataset, split_var):
        print(split_var.center(80, '='))
        var_tree = self._fit_continuous(dataset, split_var)
        print(var_tree.describe_tree())
        woe_list, split_value_list = var_tree.format_tree(var_tree, [], [])

        var_df = pd.DataFrame({"var_name": split_var,
                               "split_left": [float("-inf")] + split_value_list,
                               "split_right": split_value_list + [float("+inf")],
                               "sub_sample_cnt": [x['sub_sample_cnt'] for x in woe_list],
                               "sub_sample_bad_cnt": [x['sub_sample_bad_cnt'] for x in woe_list],
                               "sub_sample_good_cnt": [x['sub_sample_good_cnt'] for x in woe_list],
                               "woe": [x['woe'] for x in woe_list],
                               "iv": [x['iv'] for x in woe_list]
                               })
        columns = ['var_name', 'split_left', 'split_right', 'sub_sample_cnt', 'sub_sample_bad_cnt',
                   'sub_sample_good_cnt', 'woe', 'iv', 'iv_sum']
        var_df['iv_sum'] = var_df['iv'].sum()
        var_df = var_df[columns]
        return var_tree, var_df

    def _fit_continuous(self, dataset, split_var):
        var_woe, var_iv = self.calculate_iv_woe(dataset)
        if dataset['label'].unique().__len__() <= 1 or dataset[split_var].unique().__len__() <= 1:
            tree = Tree()
            tree.iv = var_iv
            tree.woe = var_woe
            tree.sub_sample_cnt = dataset.__len__()
            tree.sub_sample_bad_cnt = dataset[dataset['label'] == 1].__len__()
            tree.sub_sample_good_cnt = dataset[dataset['label'] == 0].__len__()
            return tree

        best_split_value, best_split_iv, best_dataset_left, best_dataset_right = \
            self.choose_best_split(dataset, split_var)

        if best_split_iv <= var_iv:
            tree = Tree()
            tree.iv = var_iv
            tree.woe = var_woe
            tree.sub_sample_cnt = dataset.__len__()
            tree.sub_sample_bad_cnt = dataset[dataset['label'] == 1].__len__()
            tree.sub_sample_good_cnt = dataset[dataset['label'] == 0].__len__()
            return tree
        else:
            tree = Tree()
            tree.iv = var_iv
            tree.woe = var_woe
            tree.split_value = best_split_value
            tree.sub_sample_cnt = dataset.__len__()
            tree.sub_sample_bad_cnt = dataset[dataset['label'] == 1].__len__()
            tree.sub_sample_good_cnt = dataset[dataset['label'] == 0].__len__()
            tree.tree_left = self._fit_continuous(best_dataset_left, split_var)
            tree.tree_right = self._fit_continuous(best_dataset_right, split_var)
            return tree

    # calculate the iv and woe value with given dataset
    def calculate_iv_woe(self, dataset):
        sub_bad_cnt = dataset[dataset['label'] == 1].__len__()
        sub_bad_rate = (sub_bad_cnt + 0.0001) * 1.0 / (self.total_bad_cnt + 0.0001)
        sub_good_cnt = dataset[dataset['label'] == 0].__len__()
        sub_good_rate = (sub_good_cnt + 0.0001) * 1.0 / (self.total_good_cnt + 0.0001)

        res_woe = np.log(sub_bad_rate / sub_good_rate)
        res_iv = (sub_bad_rate - sub_good_rate) * np.log(sub_bad_rate / sub_good_rate)
        return round(res_woe, 4), round(res_iv, 4)

    def choose_best_split(self, dataset, split_var):
        if dataset[split_var].unique().__len__() <= 50:
            split_value_list = dataset[split_var].unique()
        else:
            split_value_list = np.unique(np.percentile(dataset[split_var], range(100)))
        split_value_list = sorted([round(x, 4) for x in split_value_list])

        best_split_value = None
        best_split_iv = float("-inf")
        best_dataset_left = None
        best_dataset_right = None
        for split_value in split_value_list:
            dataset_left = dataset[dataset[split_var] <= split_value]
            dataset_right = dataset[dataset[split_var] > split_value]
            if dataset_right.__len__() < self.min_sample:
                break
            elif dataset_left.__len__() < self.min_sample:
                continue
            else:
                woe_left, iv_left = self.calculate_iv_woe(dataset_left)
                woe_right, iv_right = self.calculate_iv_woe(dataset_right)

                if iv_left + iv_right > best_split_iv:
                    best_split_value = split_value
                    best_split_iv = iv_left + iv_right
                    best_dataset_left = dataset_left
                    best_dataset_right = dataset_right
        return best_split_value, best_split_iv, best_dataset_left, best_dataset_right

    # axis-x: index of bins; axis-y: woe value
    def plot_woe_structure(self):
        var_list = self.woe_rule_df['var_name'].unique().tolist()
        for i in range(len(var_list)):
            try:
                woe_list = self.woe_rule_df[self.woe_rule_df['var_name'] == var_list[i]]['woe'].tolist()
                if len(woe_list) >= 2:
                    plt.plot(range(len(woe_list)), woe_list, label=str(len(woe_list)) + '_' + var_list[i])
            except:
                pass
        plt.legend()
        plt.show()

    # replace the feature value with given woe rule
    def transform(self, dataset):
        dataset_copy = copy.deepcopy(dataset)
        for var in dataset_copy.columns:
            if var in self.woe_rule_dict.keys():
                dataset_copy[var] = dataset_copy[var].apply(lambda x: self._transform(self.woe_rule_dict[var], x))
        return dataset_copy

    @staticmethod
    def _transform(sub_woe_rule, value):
        for rule in sub_woe_rule:
            if rule[0] > value:
                return rule[1]
        return -99


if __name__ == '__main__':
    df = pd.read_csv("source/credit_card.csv")
    woe = WoeFeatureProcess(path_conf="f_conf/credit_card.conf", path_woe="result/woe_rule.csv", min_sample_rate=0.1)
    woe.fit(df)
    woe.plot_woe_structure()
    print(woe.woe_rule_df)
    df_woed = woe.transform(df)
    print(df_woed.head())
