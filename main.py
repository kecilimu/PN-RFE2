
"""
@Time ： 2024/10/4 17:24
@Auth ： KECILIMU
@File ：main.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
import numpy as np
from matplotlib import pyplot as plt

import utils
from model.REF2 import REF2
from model.dataloader import data_gen, data_load_s
from File_log import log as lg
from File_log import file_create as fc
from train_model import train_model, val, train_ML, transform_data

# for specify structure test with same seed.
# def set_seed(seed=105):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    lg_i = lg.log()
    set_seed()
    result_path = lg_i.result_path + "/"
    fc.check_dictory_path(result_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lg_i.log_write('device: %s' % device)


    plt.close("all")
    train_loader, val_loader = data_gen(device)
    lg_i.log_write('data loaded')
    model = REF2(utils.input_dimension, utils.hidden_dimension, utils.output_dimension, utils.drop_ratio)
    lg_i.log_write('model created')
    lg_i.log_write('start training')
    model, result, loss, fig1, fig2 = train_model(model, [train_loader, val_loader], device)
    nresult = np.array(result)

    # deep learning
    lg_i.log_write('training finished')
    key_result = [nresult[np.argmin(nresult[:, 1], 0), :]
        , nresult[np.argmin(nresult[:, 2], 0), :]
        , nresult[np.argmax(nresult[:, 3], 0), :]
        , nresult[np.argmin(nresult[:, 4], 0), :]
        , nresult[np.argmin(nresult[:, 5], 0), :]
        , nresult[np.argmax(nresult[:, 6], 0), :]
                  ]
    lg_i.model_write(model)

    lg_i.save(result, result_path + utils.result_suffix[0])
    lg_i.save(loss, result_path + utils.result_suffix[1])
    rmse, mae, r2 = val(model, val_loader)
    print(lg_i.time + ": RMSE: %.4f, MAE: %.4f, R2: %.4f" % (rmse, mae, r2))
    combine_results = [key_result, rmse, mae, r2]
    Hparams = [utils.train_size_ratio,
               utils.input_dimension,
               utils.hidden_dimension,
               utils.output_dimension,
               utils.drop_ratio,
               utils.batch_size,
               utils.batch_size_,
               utils.epoch,
               utils.lr,
               utils.step_size,
               utils.gamma
               ]
    lg_i.save(Hparams, result_path + utils.result_suffix[2])
    lg_i.save(combine_results, result_path + utils.result_suffix[3])
    lg_i.img_write(fig1, result_path + utils.result_suffix[4])
    lg_i.img_write(fig2, result_path + "2" + utils.result_suffix[4])
    lg_i.log_write('combine_results saved')

