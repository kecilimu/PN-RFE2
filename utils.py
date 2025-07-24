
"""
@Time ： 2024/10/4 17:40
@Auth ： KECILIMU
@File ：utils.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""


# data_path = 'data/RefineDataset_numberic_v _pure.xlsx'
train_size_ratio = 0.8
result_path = 'File_log/result'

input_dimension = 16
hidden_dimension = [609, 609, 309, 309, 159, 159, 1]
output_dimension = 1
drop_ratio = 0

batch_size = 64
batch_size_ = 25
epoch = 1000
lr = 0.00001
step_size = 3000
gamma = 0.1
result_suffix = [
    'total_result.pkl',
    'total_losss.pkl',
    'hparams.pkl',
    'combined_result.pkl',
    'figure.png',
    "GBRTree.pkl",
    "hidden_dimension.pkl",
    "data.pkl"
]

# machine learning

loss = "squared_error"
learning_rate = 0.5
n_estimators = 90
max_depth = 6
min_samples_split = 5