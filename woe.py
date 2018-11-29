# -*- coding: utf-8 -*-
"""
@Time: 2018/8/21 11:34
@Author: zhaoxingfeng
@Function：Weight of Evidence,根据iv值最大思想求最优分箱
@Version: V1.2
参考文献：
[1] kingsam_. 数据挖掘模型中的IV和WOE详解[DB/OL].https://blog.csdn.net/kevin7658/article/details/50780391/.
[2] boredbird. woe[DB/OL].https://github.com/boredbird/woe.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


# 存储分裂过程中的切分点、woe、iv等信息
class Tree(object):
    def __init__(self):
        self.bin_value_list = []
        self.split_value = None
        self.sub_sample_cnt = None
        self.sub_sample_bad_cnt = None
        self.sub_sample_good_cnt = None
        self.iv = None
        self.woe = None
        self.tree_left = None
        self.tree_right = None

    # 以JSON形式打印树结构，用于调试代码
    def describe_tree(self):
        if not self.tree_left or not self.tree_right:
            tree_node = "{iv:" + str(self.iv) + \
                        ",woe:" + str(self.woe) + \
                        ",bin_value_list:" + str(self.bin_value_list) + \
                        ",sub_sample_cnt:" + str(self.sub_sample_cnt) + \
                        ",sub_sample_bad_cnt:" + str(self.sub_sample_bad_cnt) + \
                        ",sub_sample_good_cnt:" + str(self.sub_sample_good_cnt) + "}"
            return tree_node
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{bin_value_list:" + str(self.bin_value_list) + \
                         ",split_value:" + str(self.split_value) + \
                         ",sub_sample_cnt:" + str(self.sub_sample_cnt) + \
                         ",sub_sample_bad_cnt:" + str(self.sub_sample_bad_cnt) + \
                         ",sub_sample_good_cnt:" + str(self.sub_sample_good_cnt) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure

    # 从分箱树结构中获取每一箱的woe值、切分点、离散特征取值集合
    def format_tree(self, tree, woe_iv_list, split_value_list):
        if tree.split_value == None:
            tree_node = {"bin_value_list": tree.bin_value_list,
                         "iv": tree.iv,
                         "woe": tree.woe,
                         "sub_sample_cnt": tree.sub_sample_cnt,
                         "sub_sample_bad_cnt": tree.sub_sample_bad_cnt,
                         "sub_sample_good_cnt": tree.sub_sample_good_cnt}
            woe_iv_list.append(tree_node)
            return woe_iv_list, split_value_list

        self.format_tree(tree.tree_left, woe_iv_list, split_value_list)
        split_value = tree.split_value
        split_value_list.append(split_value)
        self.format_tree(tree.tree_right, woe_iv_list, split_value_list)
        return woe_iv_list, split_value_list


class WoeFeatureProcess(object):
    def __init__(self, path_conf, path_woe_rule, min_sample_rate=0.1, min_iv=0.0005):
        """
        :param path_conf: 描述每个特征的情况
            is_continous: 1为连续型变量，0为离散型变量
            is_identify: 置为1表示该特征不参与woe转化
            var_dtype: 特征数据类型
            var_name: 特征名
        :param path_woe_rule: 存储csv格式特征分箱
        :param min_sample_rate: 每个分箱最小样本比例(*总体样本)
        :param min_iv: 每个分箱最小iv，如果小于给定值则该箱被合并
        """
        self.dataset = None
        self.conf = pd.read_csv(path_conf)
        self.continous_var_list = self.conf[(self.conf['is_continous'] == 1) & (self.conf['is_identify'] == 0)]['var_name']
        self.discrete_var_list = self.conf[(self.conf['is_continous'] == 0) & (self.conf['is_identify'] == 0)]['var_name']
        self.woe_rule_dict = dict()
        self.woe_rule_df = pd.DataFrame()
        self.path_woe_rule = path_woe_rule
        self.min_sample_rate = min_sample_rate
        self.total_bad_cnt = 1
        self.total_good_cnt = 1
        self.min_sample = 1
        self.min_iv = min_iv

    def fit(self, dataset):
        self.dataset = dataset
        self.total_bad_cnt = dataset[dataset['label'] == 1].__len__()
        self.total_good_cnt = dataset[dataset['label'] == 0].__len__()
        self.min_sample = int(len(self.dataset) * self.min_sample_rate)

        print("PROCESS CONTINOUS VARIABLES".center(80, '='))
        for var in self.continous_var_list:
            if var in self.dataset.columns:
                print(var.center(80, '='))
                var_df = self.fit_continous(self.dataset[[var, 'label']], var)
                self.woe_rule_df = var_df if self.woe_rule_df.empty else pd.concat([self.woe_rule_df, var_df], ignore_index=1)

        print("PROCESS DISCRETE VARIABLES".center(80, '='))
        for var in self.discrete_var_list:
            if var in self.dataset.columns:
                print(var.center(80, '='))
                var_df = self.fit_discrete(self.dataset[[var, 'label']], var)
                self.woe_rule_df = var_df if self.woe_rule_df.empty else pd.concat([self.woe_rule_df, var_df], ignore_index=1)

        cols = ['var_name', 'bin_value_list', 'split_left', 'split_right', 'sub_sample_cnt', 'sub_sample_bad_cnt',
                'sub_sample_good_cnt', 'woe', 'iv', 'iv_sum']
        self.woe_rule_df = self.woe_rule_df.sort_values(by=['var_name', 'split_left']).reset_index(drop=True)
        self.woe_rule_df = self.woe_rule_df[cols]
        self.woe_rule_df = self.woe_rule_df.sort_values(by=['iv_sum', 'var_name'], ascending=False).reset_index(drop=True)
        self.woe_rule_df.to_csv(self.path_woe_rule, index=None, float_format="%.4f")

        for var, grp in self.woe_rule_df.groupby(['var_name']):
            if isinstance(grp.bin_value_list.tolist()[0], list):
                self.woe_rule_dict[var] = list(zip(grp.bin_value_list, grp.woe))
            else:
                self.woe_rule_dict[var] = list(zip(grp.split_right, grp.woe))

    # 处理连续型变量
    def fit_continous(self, dataset, var):
        var_tree = self._fit_continous(dataset, var)
        print(var_tree.describe_tree())
        woe_iv_list, split_value_list = var_tree.format_tree(var_tree, [], [])

        var_df = pd.DataFrame({"var_name": var,
                               "bin_value_list": None,
                               "split_left": [float("-inf")] + split_value_list,
                               "split_right": split_value_list + [float("+inf")],
                               "sub_sample_cnt": [x['sub_sample_cnt'] for x in woe_iv_list],
                               "sub_sample_bad_cnt": [x['sub_sample_bad_cnt'] for x in woe_iv_list],
                               "sub_sample_good_cnt": [x['sub_sample_good_cnt'] for x in woe_iv_list],
                               "woe": [x['woe'] for x in woe_iv_list],
                               "iv": [x['iv'] for x in woe_iv_list]
                               })
        var_df['iv_sum'] = var_df['iv'].sum()
        return var_df

    # 处理连续型变量
    def _fit_continous(self, dataset, var):
        var_woe, var_iv = self.calculate_woe_iv(dataset)
        if dataset['label'].unique().__len__() <= 1 or dataset[var].unique().__len__() <= 1:
            tree = Tree()
            tree.iv = var_iv
            tree.woe = var_woe
            tree.sub_sample_cnt = dataset.__len__()
            tree.sub_sample_bad_cnt = dataset[dataset['label'] == 1].__len__()
            tree.sub_sample_good_cnt = dataset[dataset['label'] == 0].__len__()
            return tree

        best_split_value, best_split_iv, best_dataset_left, best_dataset_right = \
            self.choose_best_split(dataset, var)

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
            tree.tree_left = self._fit_continous(best_dataset_left, var)
            tree.tree_right = self._fit_continous(best_dataset_right, var)
            return tree

    # 处理离散型变量
    def fit_discrete(self, dataset, var):
        value_woe_dict = {}
        for value in dataset[var].unique():
            woe, iv = self.calculate_woe_iv(dataset[dataset[var] == value])
            value_woe_dict[value] = woe
        dataset[var] = dataset[var].map(value_woe_dict)

        temp = sorted(value_woe_dict.iteritems(), key=lambda x: x[1])
        bin_woe_list, bin_value_list = [x[1] for x in temp], [x[0] for x in temp]
        var_tree = self._fit_discrete(dataset, var, bin_value_list, bin_woe_list)
        # print(var_tree.describe_tree())
        woe_iv_list, split_value_list = var_tree.format_tree(var_tree, [], [])

        var_df = pd.DataFrame({"var_name": var,
                               "bin_value_list": [x['bin_value_list'] for x in woe_iv_list],
                               "split_left": None,
                               "split_right": None,
                               "sub_sample_cnt": [x['sub_sample_cnt'] for x in woe_iv_list],
                               "sub_sample_bad_cnt": [x['sub_sample_bad_cnt'] for x in woe_iv_list],
                               "sub_sample_good_cnt": [x['sub_sample_good_cnt'] for x in woe_iv_list],
                               "woe": [x['woe'] for x in woe_iv_list],
                               "iv": [x['iv'] for x in woe_iv_list]
                               })
        var_df['iv_sum'] = var_df['iv'].sum()
        return var_df

    # 处理离散型变量
    def _fit_discrete(self, dataset, var, bin_value_list, bin_woe_list):
        var_woe, var_iv = self.calculate_woe_iv(dataset)
        if dataset['label'].unique().__len__() <= 1 or dataset[var].unique().__len__() <= 1:
            tree = Tree()
            tree.bin_value_list = bin_value_list
            tree.iv = var_iv
            tree.woe = var_woe
            tree.sub_sample_cnt = dataset.__len__()
            tree.sub_sample_bad_cnt = dataset[dataset['label'] == 1].__len__()
            tree.sub_sample_good_cnt = dataset[dataset['label'] == 0].__len__()
            return tree

        best_split_value, best_split_iv, best_dataset_left, best_dataset_right = \
            self.choose_best_split(dataset, var)

        if best_split_iv <= var_iv:
            tree = Tree()
            tree.bin_value_list = bin_value_list
            tree.iv = var_iv
            tree.woe = var_woe
            tree.sub_sample_cnt = dataset.__len__()
            tree.sub_sample_bad_cnt = dataset[dataset['label'] == 1].__len__()
            tree.sub_sample_good_cnt = dataset[dataset['label'] == 0].__len__()
            return tree
        else:
            ix = bin_woe_list.index(best_split_value)
            tree = Tree()
            tree.bin_value_list = bin_value_list
            tree.iv = var_iv
            tree.woe = var_woe
            tree.split_value = best_split_value
            tree.sub_sample_cnt = dataset.__len__()
            tree.sub_sample_bad_cnt = dataset[dataset['label'] == 1].__len__()
            tree.sub_sample_good_cnt = dataset[dataset['label'] == 0].__len__()
            tree.tree_left = self._fit_discrete(best_dataset_left, var, bin_value_list[:ix+1], bin_woe_list[:ix+1])
            tree.tree_right = self._fit_discrete(best_dataset_right, var, bin_value_list[ix+1:], bin_woe_list[ix+1:])
            return tree

    # 计算给定样本的woe、iv
    def calculate_woe_iv(self, dataset):
        sub_bad_cnt = dataset[dataset['label'] == 1].__len__()
        sub_bad_rate = (sub_bad_cnt + 0.0001) * 1.0 / (self.total_bad_cnt + 0.0001)
        sub_good_cnt = dataset[dataset['label'] == 0].__len__()
        sub_good_rate = (sub_good_cnt + 0.0001) * 1.0 / (self.total_good_cnt + 0.0001)

        res_woe = np.log(sub_bad_rate / sub_good_rate)
        res_iv = (sub_bad_rate - sub_good_rate) * np.log(sub_bad_rate / sub_good_rate)
        return round(res_woe, 4), round(res_iv, 4)

    # 基于决策树分裂思想寻找最优切分点
    def choose_best_split(self, dataset, var):
        if dataset[var].unique().__len__() <= 50:
            split_value_list = dataset[var].unique()
        else:
            split_value_list = np.unique(np.percentile(dataset[var], range(100)))
        split_value_list = sorted([round(x, 4) for x in split_value_list])

        best_split_value = None
        best_split_iv = float("-inf")
        best_dataset_left = None
        best_dataset_right = None
        for split_value in split_value_list:
            dataset_left = dataset[dataset[var] <= split_value]
            dataset_right = dataset[dataset[var] > split_value]
            if dataset_right.__len__() < self.min_sample:
                break
            elif dataset_left.__len__() < self.min_sample:
                continue
            else:
                woe_left, iv_left = self.calculate_woe_iv(dataset_left)
                woe_right, iv_right = self.calculate_woe_iv(dataset_right)

                if iv_left + iv_right > best_split_iv and iv_left >= self.min_iv and iv_right >= self.min_iv:
                    best_split_value = split_value
                    best_split_iv = iv_left + iv_right
                    best_dataset_left = dataset_left
                    best_dataset_right = dataset_right
        return best_split_value, best_split_iv, best_dataset_left, best_dataset_right

    # 绘制分箱后的woe趋势图：X-分箱号，Y-箱内woe值
    def plot_woe_structure(self):
        var_list = self.woe_rule_df['var_name'].unique().tolist()
        for i in range(len(var_list)):
            try:
                woe_iv_list = self.woe_rule_df[self.woe_rule_df['var_name'] == var_list[i]]['woe'].tolist()
                if len(woe_iv_list) >= 2:
                    plt.plot(range(len(woe_iv_list)), woe_iv_list, label=str(len(woe_iv_list)) + '_' + var_list[i])
            except:
                pass
        plt.legend()
        plt.show()

    # 对原始样本进行woe转化
    def transform(self, dataset):
        dataset_copy = copy.deepcopy(dataset)
        for var in dataset_copy.columns:
            if var in self.woe_rule_dict.keys():
                if isinstance(self.woe_rule_dict[var][0][0], list):
                    dataset_copy[var] = dataset_copy[var].apply(lambda x: self._transform_discrete(self.woe_rule_dict[var], x))
                else:
                    dataset_copy[var] = dataset_copy[var].apply(lambda x: self._transform_continous(self.woe_rule_dict[var], x))
        return dataset_copy

    @staticmethod
    def _transform_continous(sub_woe_rule, value):
        for rule in sub_woe_rule:
            if rule[0] > value:
                return rule[1]
        return -99

    @staticmethod
    def _transform_discrete(sub_woe_rule, value):
        for rule in sub_woe_rule:
            if value in rule[0]:
                return rule[1]
        return -99


if __name__ == '__main__':
    df = pd.read_csv("source/credit_card.csv")
    woe = WoeFeatureProcess(path_conf="f_conf/credit_card.conf",
                            path_woe_rule="result/woe_rule.csv",
                            min_sample_rate=0.1,
                            min_iv=0.0005)
    woe.fit(df)
    print(woe.woe_rule_df)
    woe.plot_woe_structure()
    df_woed = woe.transform(df)
    print(df_woed.head())
