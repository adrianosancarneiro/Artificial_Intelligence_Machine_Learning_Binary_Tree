import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.externals.six import StringIO
import pydot
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split


class Node:

    def __init__(self, attribute_name, is_leaf, leaf_value, child_node):
        self.attribute_name = attribute_name
        self.is_leaf = is_leaf
        self.leaf_value = leaf_value
        self.children_node = child_node


class Util:
    def __init__(self):
        self.data = None

    # def get_best_column(self, columns_weighted_entropy_list):

    def most_frequent(self, List):
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]

    '''
    The first number in the dictionary value tuple is the value entropy.
    The second number is the times this value appear in the column
    '''

    def calculate_column_weighted_entropy(self, values_entropy_list):
        # data_count = len(values_entropy_list)
        # Getting the array of tuples converting to np.array
        # and sum the second column (index 1)
        # which contain the number of times this value appear in the column
        data_count = np.array([x[1] for x in values_entropy_list.values()]).sum()

        column_weighted_entropy = 0.
        for val in values_entropy_list.keys():
            column_weighted_entropy += (values_entropy_list[val][1] / float(data_count)) * values_entropy_list[val][0]
        return column_weighted_entropy

    def calculate_antropy(self, data):
        data_uniques_values = set(data)
        data_count = len(data)
        data_entropy = 0.

        if data_count <= 1:
            return 0
        else:
            for val in data_uniques_values:
                val_count = len(data.loc[data == val])
                data_entropy = -((val_count / float(data_count)) *
                                 math.log((val_count / float(data_count)), 2))
        return data_entropy

        # def calculate_antropy2(self, data):
    #     value,counts = np.unique(data, return_counts=True)
    #     norm_counts = counts / float(counts.sum())
    #     # base = e if base is None else base
    #     return -(norm_counts * np.log(norm_counts)/np.log(2)).sum()


class Decision_Tree_ID3:
    def __init__(self):
        self.data = None

    def predict(self, node, testing_data):
        prediction_ret = {}
        for index, row in testing_data.iterrows():
            row_value = row[node.attribute_name]
            child_node = node.children_node[row_value]
            if child_node.is_leaf == True:
                prediction_ret[index] = child_node.leaf_value
            else:
                pr = self.predict(child_node, testing_data.loc[[index]])
                prediction_ret[index] = list(pr.values())[0]

        return prediction_ret

    def print_tree(self, node, indent=0):
        for key, value in node.children_node.items():
            print('\t' * indent + str(node.attribute_name))
            if value.is_leaf == False:
                print('\t' * (indent + 1) + str(key))
                self.print_tree(value, indent + 2)
            else:
                print('\t' * (indent + 1) + str(value.attribute_name))
                print('\t' * (indent + 1) + str(value.leaf_value))

    # def print_tree1(self, node, data, target):
    #     dot_data = StringIO()
    #     tree.export_graphviz(node, 
    #                             out_file = dot_data,
    #                             feature_names = data.columns,
    #                             class_names = set(data[target]),
    #                             filled = True,
    #                             rounded = True,
    #                             impurity = False)
    #     graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #     graph.write_pdf("file123.pdf")

    def fit(self, current_data, available_columns, current_target, child_best_column, child_best_column_value):
        current_node = Node(None, None, None, {})
        u = Util()
        values_entropy_list = {}
        column_entropy_list = {}
        # Available columns minus the target - we are not going to calculate the 
        # target entropy neither the information gain
        available_columns = np.delete(available_columns, np.where(available_columns == current_target))

        # Base cases to control when the recursive loop get the the bases of the tree | leafs
        # verifing if all the values in the target column are the same
        # verifing if there is more available columns
        # verifing if there is more data
        if len(set(current_data[current_target])) <= 1:
            current_node.attribute_name = child_best_column_value
            # current_node.attribute_name = child_best_column + "_" + child_best_column_value
            current_node.is_leaf = True
            current_node.leaf_value = current_data[current_target].iloc[0]
            current_node.children_node = None
            return current_node
        elif len(current_data) == 0:
            return None
        elif len(available_columns) == 0:
            current_node.attribute_name = child_best_column_value
            # current_node.attribute_name = child_best_column + "_" + child_best_column_value
            current_node.is_leaf = True
            current_node.leaf_value = u.most_frequent(current_data[current_target])
            current_node.children_node = None
            return current_node

        for col in available_columns:
            # calculate antropy
            values_entropy_list = {}
            # Loop through all the unique values in the column
            for val in set(current_data[col]):
                # sending to calculate entropy method only target column where the looped column (col) has the looped value (val)
                # e1 = u.calculate_antropy(current_data.loc[current_data[col] == val][current_target])
                values_entropy_list[val] = \
                    (
                        u.calculate_antropy(current_data.loc[current_data[col] == val][current_target]),
                        len(current_data.loc[current_data[col] == val])
                    )

            column_entropy_list[col] = u.calculate_column_weighted_entropy(values_entropy_list)

        best_column = min(column_entropy_list, key=column_entropy_list.get)
        for best_column_value in set(current_data[best_column]):
            child_current_data = current_data.loc[current_data[best_column] == best_column_value].drop(best_column, 1)
            # child_current_data = current_data.loc[current_data[best_column] == best_column_value].drop(best_column, 1)
            child_available_columns = np.delete(available_columns, np.where(available_columns == best_column))

            child_node = self.fit(
                # sending all the data where in the best column the possible value exist
                # dropping the best column
                # df = df.drop('column_name', 1)
                # where 1 is the axis number (0 for rows and 1 for columns.)
                child_current_data,
                child_available_columns,
                current_target,
                best_column,
                best_column_value
            )

            current_node.attribute_name = best_column
            current_node.is_leaf = False
            current_node.children_node[best_column_value] = child_node

        return current_node


##########################################################################
# Fruit Data Test
# Fruit
# 1 apple | 2 Grape | 3 Lemon
# Color
# 1 Green | 2 Red | 3 Yellow
##########################################################################

fruit_training_data = np.array([
    ['id', 'color', 'hight', 'weight', 'fruit'],
    [1, 1, 5, 3, 1],
    [2, 3, 2, 3, 1],
    [3, 2, 2, 1, 2],
    [4, 2, 2, 1, 2],
    [5, 3, 2, 3, 3],
    [6, 3, 5, 1, 1],
    [7, 3, 2, 1, 3],
    [8, 3, 2, 1, 3],
    [9, 3, 2, 1, 1],
])

fruit_training_data = pd.DataFrame(data=fruit_training_data[1:, 1:],  # values
                                   index=fruit_training_data[1:, 0],  # 1st column as index
                                   columns=fruit_training_data[0, 1:])  # 1st row as the column names

fruit_testing_data = np.array([
    ['id', 'color', 'hight', 'weight', 'fruit'],
    [1, 3, 2, 1, 3],
    [2, 3, 5, 1, 1],
    [3, 2, 2, 1, 2],
    [4, 1, 5, 3, 1]
])

fruit_testing_data = pd.DataFrame(data=fruit_testing_data[1:, 1:],  # values
                                  index=fruit_testing_data[1:, 0],  # 1st column as index
                                  columns=fruit_testing_data[0, 1:])  # 1st row as the column names

fruit_my_tree = Decision_Tree_ID3()
fruit_Node = fruit_my_tree.fit(fruit_training_data, fruit_training_data.columns, "fruit", None, None)
print "My Fruit Tree"
fruit_my_tree.print_tree(fruit_Node)
fruit_tree_predict = fruit_my_tree.predict(fruit_Node, fruit_testing_data)
print "My Testing Data"
print fruit_testing_data
print "My Prediction"
print fruit_tree_predict

# fruit_training_data.color = fruit_training_data.color.astype('category')
# fruit_training_data["color"] = fruit_training_data.color.cat.codes

# fruit_training_data.fruit = fruit_training_data.fruit.astype('category')
# fruit_training_data["fruit"] = fruit_training_data.fruit.cat.codes


fruit_training_data_X = fruit_training_data.drop(columns=["fruit"]).as_matrix()
fruit_training_data_y = fruit_training_data["fruit"].as_matrix().flatten()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(fruit_training_data_X, fruit_training_data_y)

# fruit_testing_data.color = fruit_testing_data.color.astype('category')
# fruit_testing_data["color"] = fruit_testing_data.color.cat.codes

# fruit_testing_data.fruit = fruit_testing_data.fruit.astype('category')
# fruit_testing_data["fruit"] = fruit_testing_data.fruit.cat.codes

fruit_testing_data_X = fruit_testing_data.drop(columns=["fruit"]).as_matrix()



pred = clf.predict(fruit_testing_data_X)

print "Sklearn Prediction"
print pred

##########################################################################
# Lens Data Test
##########################################################################

'''
Columns and descriptions

AGE
1 - young
2 - pre-presbyopic
3 - presbyopic
SPECTACLE
1 - myope
2- hypermetrope
ASTIGMATIC
1 - no
2 - yes
TEARS
1 - reduced
2 - normal
TARGET
1 - the patient should be fitted with hard contact lenses,
2 - the patient should be fitted with soft contact lenses,
3 - the patient should not be fitted with contact lenses.
'''

lens_names = ['AGE',
              'SPECTACLE',
              'ASTIGMATIC',
              'TEARS',
              'TARGET'
              ]

lens = pd.read_csv(
    '/Users',
    names=lens_names,
    header=None,
    skipinitialspace=True,
    doublequote=True,
    delim_whitespace=True
)

lens_training_data, lens_testing_data = train_test_split(lens, test_size=0.2)

print "Lens Training Data"
print lens_training_data

print "Lens Testing Data"
print lens_testing_data

lens_my_tree = Decision_Tree_ID3()
lens_Node = lens_my_tree.fit(lens_training_data, lens_training_data.columns, "TARGET", None, None)
print "My Lens Tree"
lens_my_tree.print_tree(lens_Node)
lens_tree_predict = lens_my_tree.predict(lens_Node, lens_testing_data)
print "My Testing Data"
print lens_testing_data
print "My Prediction"
print lens_tree_predict

lens_training_data_X = lens_training_data.drop(columns=["TARGET"]).as_matrix()
lens_training_data_y = lens_training_data["TARGET"].as_matrix().flatten()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(lens_training_data_X, lens_training_data_y)

# fruit_testing_data.color = fruit_testing_data.color.astype('category')
# fruit_testing_data["color"] = fruit_testing_data.color.cat.codes

# fruit_testing_data.fruit = fruit_testing_data.fruit.astype('category')
# fruit_testing_data["fruit"] = fruit_testing_data.fruit.cat.codes

lens_testing_data_X = lens_testing_data.drop(columns=["TARGET"]).as_matrix()



pred = clf.predict(lens_testing_data_X)

print "Sklearn Prediction"
print pred

##########################################################################
# Iris Data Test
##########################################################################

iris = load_iris()

iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])

print(iris)

# iris_data = iris.drop(columns=["target"]).as_matrix()
# iris_target = iris["target"].as_matrix().flatten()
# iris_training_data, iris_testing_data, iris_training_target, iris_testing_taget = train_test_split(iris_data, iris_target, test_size=0.2)

iris_training_data, iris_testing_data = train_test_split(iris, test_size=0.95)

print "Iris Training Data"
print iris_training_data

print "Iris Testing Data"
print iris_testing_data

iris_my_tree = Decision_Tree_ID3()
iris_Node = iris_my_tree.fit(iris_training_data, iris_training_data.columns, "target", None, None)
print "My Iris Tree"
iris_my_tree.print_tree(iris_Node)
iris_tree_predict = iris_my_tree.predict(iris_Node, iris_testing_data)
print "My Testing Data"
print iris_testing_data
print "My Prediction"
print iris_tree_predict

# iris = load_iris()
# test_idx = [0, 50, 100]
# # pd.DataFrame(data= np.c_[iris['data'], iris['target']],
# #                      columns= iris['feature_names'] + ['target'])

# # iris_X = data.drop(columns=["class"]).as_matrix()
# # iris_y = data["class"].as_matrix().flatten()

# #training data
# train_target = np.delete(iris.target, test_idx)
# train_data = np.delete(iris.data, test_idx, axis=0)

# #testing data
# test_target = iris.target[test_idx]
# test_data = iris.data[test_idx]

# # iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X, iris_y, test_size=0.2)

# clf = tree.DecisionTreeClassifier()
# # clf.fit(iris_X_train, iris_y_train)
# clf.fit(train_data, train_target)

# print(test_target)
# print clf.predict(test_data)


# x = 1

# y = 1+111
# xx = ""


