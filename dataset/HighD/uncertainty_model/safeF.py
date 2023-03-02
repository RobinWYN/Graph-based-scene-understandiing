import numpy as np
import time

# import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt


class SafeField:
    def __init__(self, lanes, x_lim=[0, 1020], y_lim=[0, 90], m2grid=2) -> None:
        # 单位，米
        self.x_real = x_lim
        self.y_real = y_lim
        self.lanes = lanes
        # 单位，格
        self.m2grid = m2grid
        self.x_size = (self.x_real[1] - self.x_real[0]) * self.m2grid
        self.y_size = (self.y_real[1] - self.y_real[0]) * self.m2grid
        self.x_axis = np.linspace(self.x_real[0], self.x_real[1], self.x_size)
        self.y_axis = np.linspace(self.y_real[0], self.y_real[1], self.y_size)
        self.X_en, self.Y_en = np.meshgrid(self.x_axis, self.y_axis)
        # self.show_x_size = 1200
        # self.show_amplify = int(self.show_x_size / self.x_size)

        self.vnorm = mpl.colors.Normalize(vmin=0, vmax=1)

        self.E_base = self.set_E_base()
        self.E_all = np.zeros(shape=(self.y_size, self.x_size)) + self.E_base

    def set_E_base(self):
        E_base = np.zeros(shape=(self.y_size, self.x_size))
        for id in range(len(self.lanes) - 1):
            E_tmp_1 = np.zeros(shape=(self.y_size, self.x_size))
            E_tmp_2 = np.zeros(shape=(self.y_size, self.x_size))
            lane_center = (self.lanes[id] + self.lanes[id + 1]) / 2
            point_id_1 = (self.Y_en >= self.lanes[id]) & (self.Y_en <= lane_center)
            point_id_2 = (self.Y_en > lane_center) & (self.Y_en <= self.lanes[id + 1])

            E_tmp_1[point_id_1] = (self.Y_en[point_id_1] - lane_center) ** 6
            E_tmp_2[point_id_2] = (self.Y_en[point_id_2] - lane_center) ** 6
            if id == 0:
                E_tmp_1 /= E_tmp_1.max()
                E_tmp_2 /= E_tmp_2.max() * 3
            elif id == len(self.lanes) - 2:
                E_tmp_1 /= E_tmp_1.max() * 3
                E_tmp_2 /= E_tmp_2.max()
            else:
                E_tmp_1 /= E_tmp_1.max() * 3
                E_tmp_2 /= E_tmp_2.max() * 3
            # print(E_tmp.max())
            # plt.imshow(E_tmp_1 + E_tmp_2, cmap="jet", norm=self.vnorm)
            # plt.show()
            E_base += E_tmp_1 + E_tmp_2
        return E_base

    def add_obj(
        self,
        obj_x,
        obj_y,
        obj_vx,
        obj_vy,
        obj_type="Car",
        obj_l=5,
        obj_w=2,
        lane_width=3.5,
        varphi_d=1,
    ) -> None:
        # print(obj_x, obj_y, obj_vx)
        obj_v = (obj_vx ** 2 + obj_vy ** 2) ** 0.5 # get velocity
        obj_theta = np.arctan(obj_vy / obj_vx)  # get direction
        if abs(obj_v) < 10:
            obj_v = 10 * np.sign(obj_v) # minimum velocity = 10 or -10   极小值抑制

        #### wjh method=
        if obj_type == "Car" or obj_type == "C":
            Kj = 1500  # virtual mass
        elif obj_type == "Truck" or obj_type == "T":
            Kj = 3000  # virtual mass
        # Kj = 1000
        T = 5.3
        alpha = 1.566e-14
        beta = 6.687
        gamma = 0.3345
        l1 = obj_l
        l2 = obj_w
        lw = lane_width

        X_ego = (self.X_en - obj_x) * np.cos(obj_theta) + (self.Y_en - obj_y) * np.sin(
            obj_theta
        )
        Y_ego = (self.Y_en - obj_y) * np.cos(obj_theta) - (self.X_en - obj_x) * np.sin(
            obj_theta
        )

        # plt.imshow(X_ego, cmap="jet")
        # plt.colorbar()
        # plt.show()
        # plt.imshow(Y_ego, cmap="jet")
        # plt.colorbar()
        # plt.show()
        param_Y = (T * abs(obj_vx) + 0.5 * l1) / (
            lw + 0.05 * l2 + 3 * obj_vy
        )  # TODO why obj vy

        part1 = Kj * (alpha * obj_v ** beta + gamma)
        part2 = 1 / ((X_ego - 0.2 * T * obj_vx) ** 2 + (param_Y * Y_ego) ** 2)
        part3 = 1 / ((T * obj_v ** 2) ** 2)
        E = varphi_d * part1 * (part2 - part3)
        E[E < 0] = 0
        E[E > 1] = 1
        # print(f"max:\t{E.max()}, \tmin:\t{E.min()}")

        # #### old method
        # if obj_type == "Car" or obj_type == "C":
        #     M = 1500
        # elif obj_type == "Truck" or obj_type == "T":
        #     M = 5000
        # r2 = np.sqrt((self.X_en - obj_x) ** 2 + (obj_y - self.Y_en) ** 2)
        # # V_list.append(V/20)

        # E = (30 * M * obj_v ** 2) / (1 * r2)

        # costheta = (obj_x - self.X_en) / r2
        # # fig, axes = plt.subplots(1, 1)
        # E = E * costheta
        # np.clip(E, 0, E.max(), out=E)
        # E = E.reshape(self.y_size, self.x_size)

        self.E_all += E

    def get_field(self) -> np.ndarray:
        E_all = self.E_all

        self.E_all = np.zeros(shape=(self.y_size, self.x_size)) + self.E_base
        return E_all


if __name__ == "__main__":
    # 用于验证HighD
    lanes = [8.51, 12.59, 16.43, 21.00, 24.96, 28.80]
    safeF = SafeField(lanes)
    safeF.add_obj(obj_type="Car", obj_x=366.83, obj_y=21.68, obj_vx=40.89, obj_vy=0.01)
    safeF.add_obj(
        obj_type="Truck", obj_x=162.75, obj_y=9.39, obj_vx=-32.78, obj_vy=0.05
    )

    # # 用于绘图示意
    # x_lim = [-50, 50]
    # y_lim = [-10, 10]
    # m2grid = 10
    # # lanes = np.array(range(-2, 4, 1)) * 3.5 - 3.5 / 2
    # lanes = []
    # safeF = SafeField(lanes, x_lim=x_lim, y_lim=y_lim, m2grid=m2grid)

    # safeF.add_obj(obj_type="Car", obj_x=0, obj_y=0, obj_vx=30 / 3.6, obj_vy=0)
    # # safeF.add_obj(obj_type="Car", obj_x=0, obj_y=0, obj_vx=-30 / 3.6, obj_vy=0)
    # # safeF.add_obj(obj_type="Truck", obj_x=0, obj_y=0, obj_vx=30 / 3.6, obj_vy=0)
    # # safeF.add_obj(
    # #     obj_type="Car", obj_x=0, obj_y=0, obj_vx=15 / 3.6 * 3 ** 2, obj_vy=15 / 3.6
    # # )
    # # safeF.add_obj(
    # #     obj_type="Car", obj_x=0, obj_y=0, obj_vx=30 / 3.6, obj_vy=0, varphi_d=1.6
    # # )

    # x_tick_num = 5
    # x_tick_label = np.linspace(x_lim[0], x_lim[1], x_tick_num)
    # x_tick_locate = (x_tick_label - x_lim[0]) * m2grid

    # y_tick_num = 3
    # y_tick_label = np.linspace(y_lim[0], y_lim[1], y_tick_num)
    # y_tick_locate = (y_tick_label - y_lim[0]) * m2grid

    fig, axes = plt.subplots(1, 1)
    axes.imshow(safeF.get_field(), cmap="jet", norm=safeF.vnorm)
    # plt.xticks(x_tick_locate, x_tick_label)
    # plt.yticks(y_tick_locate, y_tick_label)

    # figure_path = "D:/Workspace/UncertaintySafeField/paper/figures/chap03/"
    # # plt.savefig(figure_path + "safeF.png", bbox_inches="tight")
    # # plt.savefig(figure_path + "safeF_back.png", bbox_inches="tight")
    # # plt.savefig(figure_path + "safeF_truck.png", bbox_inches="tight")
    # # plt.savefig(figure_path + "safeF_turn30.png", bbox_inches="tight")
    # # plt.savefig(figure_path + "safeF_lane.png", bbox_inches="tight")
    # # plt.savefig(figure_path + "safeF_uncertain1_6.png", bbox_inches="tight")

    plt.show()
    # # cv2.imshow(winname="safe field", mat=safeF.get_field())
    # # cv2.waitKey(0)

    # t_start = time.time()
    # for i in range(100):
    #     safeF.add_obj(
    #         obj_type="Car", obj_x=0, obj_y=0, obj_vx=30 / 3.6, obj_vy=0, varphi_d=1.6
    #     )
    # t_end = time.time()
    # print(t_end - t_start)

