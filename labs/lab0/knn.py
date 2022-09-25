from numpy import tile
import operator


class Knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        # Picture
        self.y = y
        # Label

    def predict(self, X):
        # TODO Predict the label of X by
        # the k nearest neighbors.

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE

        # 行数，也即训练样本的的个数，shape[1]为列数
        train_dataset_amount = len(self.y)

        # 将输入test_data变成了和train_dataset行列数一样的矩阵
        test_rep_mat = tile(X, (train_dataset_amount, 1))
        # FIXME !

        # tile(mat,(x,y)) Array类 mat 沿着行重复x次，列重复y次
        diff_mat = test_rep_mat - self.X

        # 求平方，为后面求距离准备
        sq_diff_mat = diff_mat ** 2

        # 将平方后的数据相加，sum(axis=1)是将一个矩阵的每一行向量内的数据相加，得到一个list，list的元素个数和行数一样;sum(axis=0)表示按照列向量相加
        sq_dist = sq_diff_mat.sum(axis=1)

        # 开平方，得到欧式距离
        distance = sq_dist ** 0.5

        # argsort 将元素从小到大排列，得到这个数组元素在distance中的index(索引)，dist_index元素内容是distance的索引
        dist_index = distance.argsort()

        class_count = {}
        for i in range(self.k):
            label = self.y[dist_index[i]]
            # 如果属于某个类，在该类的基础上加1，相当于增加其权重，如果不是某个类则新建字典的一个key并且等于1
            class_count[label] = class_count.get(label, 0) + 1
        # 降序排列
        class_count_list = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        print('Sorted result:', class_count_list)
        return class_count_list[0][0]
        # raise NotImplementedError
        ...

        # End of todo
