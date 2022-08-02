import copy
import re

import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

modes = ["3D", "2D"]


class LossLandscape:
    """
    References:
        This integration class is used to generate the loss landscape of the mod
        el, where the model ensures that the pth file has been imported at the e
        nd of training. The loss landscape usually demonstrated in papers is a 3
        D model, but in fact the 3D model is more time consuming than the 2D mod
        el. It is worth noting that we do not provide reliability guarantees for
        the generation of loss landscapes in 2D mode.
    """
    def __init__(self, model, trainset, criticion, weight_decay, data_decay_rate=0.1, mode="3D"):
        """
        Args:
            model: torch.nn.Module, pytorch implementation of the training model
            trainset: torch.utils.data.Dataset, The Dataset implemented by pytor
                ch needs to ensure that trainset[i][1] is the sample's correspon
                ding int-type label.
            criticion: callable, the loss function, input the output of the mode
                l and the true label can calculate the loss.
            weight_decay: float, the L2 parametric weights used in your training
                of the model.
            data_decay_rate: float, what percentage of the data is used to gener
                ate the loss landscape
            mode: str, '2D' or '3D'
        """
        assert mode in modes
        self.mode = mode
        self.model = model
        self.trainset = trainset
        self.criticion = criticion
        self.weight_decay = weight_decay
        self.data_decay_rate = data_decay_rate
        # TODO: Only data_decay_rate of the data is used for plotting
        labels = [self.trainset[i][1] for i in range(len(self.trainset))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=1 - data_decay_rate, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
        self.trainset = torch.utils.data.Subset(self.trainset, train_indices)
        self.traindataloader = DataLoader(self.trainset, shuffle=True, num_workers=4, batch_size=48)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def normalize_filter(self, bs, ws):
        # TODO: normalize
        bs = {k: v.float() for k, v in bs.items()}
        ws = {k: v.float() for k, v in ws.items()}

        norm_bs = {}
        for k in bs:
            ws_norm = torch.norm(ws[k], dim=0, keepdim=True)
            bs_norm = torch.norm(bs[k], dim=0, keepdim=True)
            norm_bs[k] = ws_norm / (bs_norm + 1e-7) * bs[k]  # random * true_norm / rand_norm

        return norm_bs

    def ignore_bn(self, ws):
        ignored_ws = {}
        for k in ws:
            if len(ws[k].size()) < 2:
                ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
            else:
                ignored_ws[k] = ws[k]
        return ignored_ws

    def ignore_running_stats(self, ws):
        return self.ignore_kw(ws, ["num_batches_tracked"])

    def ignore_kw(self, ws, kws=None):
        kws = [] if kws is None else kws

        ignored_ws = {}
        for k in ws:
            if any([re.search(kw, k) for kw in kws]):
                ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
            else:
                ignored_ws[k] = ws[k]
        return ignored_ws

    @torch.no_grad()
    def l2(self, model, device):
        l2_norm = torch.tensor(0.0)
        l2_norm = l2_norm.to(device)

        for param in model.parameters():
            l2_norm += torch.norm(param)

        return l2_norm

    @torch.no_grad()
    def synthesize_coordinates(
        self, x_min=-0.1, x_max=0.1, x_interval=11, y_min=-0.1, y_max=0.1, y_interval=11
    ):
        x = np.linspace(x_min, x_max, x_interval)
        y = np.linspace(y_min, y_max, y_interval)
        self.x, self.y = np.meshgrid(x, y)

    @torch.no_grad()
    def draw(self, x_min=-0.5, x_max=0.5, x_interval=31, y_min=-0.5, y_max=0.5, y_interval=31):
        self.synthesize_coordinates(x_min, x_max, x_interval, y_min, y_max, y_interval)
        print("Complete coordinate system establishment")
        self.create_bases(self.model)
        print("Complete random parameter sampling")
        z = self._compute_for_draw()
        print("Complete the information collection of the loss landscape")
        self.draw_figure(self.x, self.y, z)

    def __call__(
        self,
        x_min=-0.5,
        x_max=0.5,
        x_interval=11,
        y_min=-0.5,
        y_max=0.5,
        y_interval=11,
        *args,
        **kwargs,
    ):
        self.draw(x_min, x_max, x_interval, y_min, y_max, y_interval)

    def create_bases(self, model, kws=None):
        kws = [] if kws is None else kws
        x0, y0 = self._find_direction(model)
        bases = [x0, y0]
        ws0 = copy.deepcopy(model.state_dict())
        bases = [self.normalize_filter(bs, ws0) for bs in bases]
        bases = [self.ignore_bn(bs) for bs in bases]
        bases = [self.ignore_kw(bs, kws) for bs in bases]
        self.x0, self.y0 = bases

    @torch.no_grad()
    def _find_direction(self, model):
        x0 = {}
        y0 = {}
        for name, param in model.named_parameters():
            x0[name] = torch.randn_like(param.data)
            y0[name] = torch.randn_like(param.data)
        return x0, y0

    @torch.no_grad()
    def _compute_for_draw(self):
        result = []
        if self.mode == "2D":
            self.x = self.x[0]
            for i in tqdm(range(self.x.shape[0])):
                now_x = self.x[i]
                loss = self._compute_loss_for_one_coordinate(now_x, 0)
                result.append(loss)
        else:
            for i in range(self.x.shape[0]):
                for j in range(self.x.shape[1]):
                    now_x = self.x[i, j]
                    now_y = self.y[i, j]
                    loss = self._compute_loss_for_one_coordinate(now_x, now_y)
                    result.append(loss)
                    print(f"--finish coordinates ({i},{j}) loss:{round(loss,3)}")

        result = np.array(result)
        result = result.reshape(self.x.shape)
        return result

    @torch.no_grad()
    def _compute_loss_for_one_coordinate(self, now_x: float, now_y: float):
        temp_model = copy.deepcopy(self.model)
        for name, param in temp_model.named_parameters():
            param.data = param.data + now_x * self.x0[name] + now_y * self.y0[name]
        l2 = self.l2(temp_model, device=self.device) * self.weight_decay
        total_loss = self.test(temp_model) + l2
        return total_loss.clone().detach().cpu().item()

    @torch.no_grad()
    def test(self, temp_model: nn.Module):
        temp_model.eval()
        result = 0.0
        for step, (x, y) in enumerate(self.traindataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pre = temp_model(x)
            result += self.criticion(pre, y).item() * y.shape[0]
        result /= len(self.traindataloader.dataset)
        return result

    def draw_figure(self, mesh_x, mesh_y, mesh_z):
        if self.mode == "3D":
            mesh_z = mesh_z - mesh_z[np.isfinite(mesh_z)].min()
            norm = plt.Normalize(
                mesh_z[np.isfinite(mesh_z)].min(), mesh_z[np.isfinite(mesh_z)].max()
            )  # normalize to [0,1]
            colors = cm.plasma(norm(mesh_z))
            rcount, ccount, _ = colors.shape
            fig = plt.figure(figsize=(10, 8), dpi=120)
            ax = fig.gca(projection="3d")
            ax.view_init(elev=15, azim=5)  # angle
            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            surf = ax.plot_surface(
                mesh_x, mesh_y, mesh_z, cmap=plt.get_cmap("rainbow"), shade=False
            )
            surf.set_facecolor((0, 0, 0, 0))
            # adjust_lim = 0.7
            # ax.set_xlim(-1 * adjust_lim, 1 * adjust_lim)
            # ax.set_ylim(-1 * adjust_lim, 1 * adjust_lim)
            ax.set_zlim(0,max(6,mesh_z.max()))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # frame = plt.gca()
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_zaxis().set_visible(False)
            plt.xticks([])
            plt.yticks([])
            ax.axis("off")
            font1 = {
                "family": "Times New Roman",
                "weight": "bold",
                "style": "normal",
                "size": 12,
            }
            plt.rc("font", **font1)
            ax.set_xlabel("X", fontdict=font1)
            ax.set_ylabel("Y", fontdict=font1)
            ax.set_zlabel("Training Loss", fontdict=font1)
            plt.savefig(LossLandscape.get_datetime_str() + ".png")
            plt.show()

        elif self.mode == "2D":
            plt.plot(mesh_x, mesh_z)
            plt.savefig(LossLandscape.get_datetime_str() + ".png")
            plt.show()

    @staticmethod
    def get_datetime_str(style="dt"):
        import datetime

        cur_time = datetime.datetime.now()
        date_str = cur_time.strftime("%y_%m_%d_")
        time_str = cur_time.strftime("%H_%M_%S")
        if style == "data":
            return date_str
        elif style == "time":
            return time_str
        else:
            return date_str + time_str
