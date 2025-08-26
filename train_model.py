
"""
@Time ： 2024/10/4 17:27
@Auth ： KECILIMU
@File ：train_model.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import trange
import utils
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
from File_log.file_create import reflash_time as rt
from model.dataloader import get_all_data


def train_model(model,
                data,
                device,
                epoch = utils.epoch,
                lr = utils.lr):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=utils.step_size, gamma=utils.gamma, last_epoch=-1)
    criterion = torch.nn.MSELoss()
    train_loader, val_loader = data[0], data[1]

    des_result = []
    des_loss = []
    bar = trange(epoch, desc='Epoch')
    for i in bar:

        epoch_label = None
        epoch_output = None
        epoch_loss = 0
        for data, label in train_loader:
            label = torch.unsqueeze(label, 1)
            model.train()
            optimizer.zero_grad()
            out, _ = model(data)
            loss = criterion(out, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if epoch_label is None:
                epoch_label = label
                epoch_output = out
            else:
                epoch_label = torch.cat((epoch_label, label), 0)
                epoch_output = torch.cat((epoch_output, out), 0)

        # scheduler.step()
        epoch_loss /= len(train_loader)
        val_loss, rmse, mae, r2 = test_model(model, val_loader, criterion)
        t_rmse, t_mae, t_r2 = describe(epoch_output, epoch_label)
        # print("%s: %d epoch lr: %f" % (it(), i + 1, optimizer.param_groups[0]['lr']))
        # print('Train Loss: {}, Val Loss: {}'.format(epoch_loss, val_loss))
        # print('Train RMSE: {}, MAE: {}, R2: {}'.format(t_rmse, t_mae, t_r2))
        # print('Val RMSE: {}, MAE: {}, R2: {}'.format(rmse, mae, r2))
        # print('-----------------------------------------')
        des_result.append([i, t_rmse, t_mae, t_r2, rmse, mae, r2])
        des_loss.append([epoch_loss, val_loss])
        bar.set_description("Epoch %d %.3f %.3f" % (i + 1, t_r2, r2))


    fig1 = plot_loss(des_loss)
    fig2 = plot_describe(des_result)
    return model, des_result, des_loss, fig1, fig2


def test_model(model, dataloader, criterion):
    model.eval()
    test_loss = 0
    preds_list = []
    labs_list = []
    with torch.no_grad():
        for data, label in dataloader:
            label = torch.unsqueeze(label, 1)
            out, _ = model(data)
            loss = criterion(out, label)
            test_loss += loss.item()
            preds_list.append(out)
            labs_list.append(label)
        preds = torch.cat(preds_list, dim=0)
        labs = torch.cat(labs_list, dim=0)
        rmse, mae, r2 = describe(preds, labs)
    test_loss /= len(dataloader)
    return test_loss, rmse, mae, r2


def describe(P, T):
    P = P.cpu().detach().numpy()
    T = T.cpu().detach().numpy()
    rmse = root_mean_squared_error(T, P)
    mae = mean_absolute_error(T, P)
    r2 = r2_score(T, P)
    return rmse, mae, r2

def plot_loss(loss):
    loss = np.array(loss)
    fig, ax = plt.subplots(figsize=(24, 18))

    ax.plot(loss[:, 0], label='Train set', linewidth=2)
    ax.plot(loss[:, 1], label='Test set', linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=48)
    for spine in ax.spines.values():
        spine.set_linewidth(5)
    legend = ax.legend(loc='upper right', handlelength=24)
    plt.setp(legend.get_texts(), fontsize=48)
    for line in legend.get_lines():
        line.set_linewidth(6)
    ax.set_xlabel("Epoch", fontsize=48)
    ax.set_ylabel("Loss", fontsize=48)
    ax.set_ylim(0, 1)
    return fig

def plot_describe(des):
    des = np.array(des)
    fig, ax = plt.subplots(figsize=(24, 18))
    ax.plot(des[:, 1], label='train_rmse')
    ax.plot(des[:, 2], label='train_mae')
    ax.plot(des[:, 3], label='train_r2')
    ax.plot(des[:, 4], label='val_rmse')
    ax.plot(des[:, 5], label='val_mae')
    ax.plot(des[:, 6], label='val_r2')
    ax.tick_params(axis='both', which='major', labelsize=48)
    for spine in ax.spines.values():
        spine.set_linewidth(5)
    legend = ax.legend(loc='upper right', handlelength=24)
    plt.setp(legend.get_texts(), fontsize=48)
    for line in legend.get_lines():
        line.set_linewidth(6)
    ax.set_xlabel("Epoch", fontsize=48)
    ax.set_ylabel("Loss", fontsize=48)
    ax.set_ylim(-0.5, 1)
    return fig

def val(model, dataloader):
    model.eval()
    preds_list = []
    labs_list = []
    with torch.no_grad():
        for data, lab in dataloader:
            lab = torch.unsqueeze(lab, 1)
            output, _ = model(data)
            preds_list.append(output)
            labs_list.append(lab)
        preds = torch.cat(preds_list, dim=0)
        labs = torch.cat(labs_list, dim=0)
        rmse, mae, r2 = describe(preds, labs)
    return rmse, mae, r2

def get_feature(model, data):
    model.eval()
    with torch.no_grad():
        _, feature = model(data)
    return feature

def transform_data(model, dataloader):
    train_x, train_y = get_all_data(dataloader[0])
    val_x, val_y = get_all_data(dataloader[1])
    ex_train_x = get_feature(model, train_x).detach().to('cpu').numpy()
    ex_val_x = get_feature(model, val_x).detach().to('cpu').numpy()
    train_y = train_y.detach().to('cpu').numpy()
    val_y = val_y.detach().to('cpu').numpy()
    return [ex_train_x, train_y, ex_val_x, val_y]

def train_ML(data, model):
    time_random_seed = rt().second
    [ex_train_x, train_y, ex_val_x, val_y] = data
    GBRTree = GradientBoostingRegressor(loss=utils.loss,
                                        n_estimators=utils.n_estimators,
                                        learning_rate=utils.learning_rate,
                                        max_depth=utils.max_depth,
                                        random_state=time_random_seed,
                                        )
    GBRTree.fit(ex_train_x, train_y)
    rmset, maet, r2t = evluation_ML(GBRTree, ex_train_x, train_y)
    rmsev, maev, r2v = evluation_ML(GBRTree, ex_val_x, val_y)
    print('Train RMSE: {}, MAE: {}, R2: {}'.format(rmset, maet, r2t))
    print('Val RMSE: {}, MAE: {}, R2: {}'.format(rmsev, maev, r2v))
    return GBRTree, rmset, maet, r2t, rmsev, maev, r2v

def evluation_ML(model, x, y):
    y_p = model.predict(x)
    rmse = root_mean_squared_error(y, y_p)
    mae = mean_absolute_error(y, y_p)
    r2 = r2_score(y, y_p)

    return rmse, mae, r2
