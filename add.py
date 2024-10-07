'''
To do list

1. 그래프 차원 선택(1변수, 2변수)
2. 그래프 변수 선택(checkbox로 구현)
3. skopt 말고 다른 라이브러리 선택지 추가
4. 오차범위 설정 추가
5.

https://www.nature.com/articles/s41598-024-60478-9

https://www.nature.com/articles/s41578-021-00337-5

'''

import tkinter
import tkinter as tk
import tkinter.ttk as ttk
import math
import time

from tkinter import filedialog

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import norm

update_time = 0.01


def rbf_kernel(x1, x2, length_scale=0.3, sigma_f=1.0):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f ** 2 * np.exp(-0.5 / length_scale ** 2 * sqdist)


class Custom_GP:
    def __init__(self, kernel=rbf_kernel, noise=1e-8):
        self.kernel = kernel
        self.noise = noise

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        self.k = self.kernel(x_train, x_train) + self.noise ** 2 * np.eye(len(x_train))
        self.k_inv = np.linalg.inv(self.k)

    def predict(self, x_s):
        print("1")
        k_s = self.kernel(self.x_train, x_s)
        print("2")
        k_ss = self.kernel(x_s, x_s) + self.noise ** 2 * np.eye(len(x_s))
        print("3")
        mu_s = k_s.T.dot(self.k_inv).dot(self.y_train)
        cov_s = k_ss - k_s.T.dot(self.k_inv).dot(k_s)
        std_s = np.sqrt(np.maximum(np.diag(cov_s), -np.diag(cov_s)))
        return mu_s, std_s


class Parm_Set_Page:
    def __init__(self, root):
        self.window = tk.Toplevel(root.window)
        self.window.geometry('300x600')
        self.window.title('Paramter Setting')
        self.window.resizable(False, False)

        self.var_num = root.var_num + 1

        self.max_val = []
        self.min_val = []

        for i in range(self.var_num):
            row_frame = tk.Frame(self.window)
            row_frame.pack(side=tk.TOP, fill=tk.X)
            tk.Label(row_frame, text=f"var {i + 1}").pack(side=tk.TOP)
            inner_frame = tk.Frame(row_frame)
            inner_frame.pack(side=tk.TOP, anchor="center")
            tk.Label(inner_frame, text="min").pack(side=tk.LEFT)
            self.min_val.append(tk.Entry(inner_frame, width=10))
            self.min_val[i].insert(0, str(root.min_list[i]))
            self.min_val[i].pack(side=tk.LEFT)

            tk.Label(inner_frame, text="max").pack(side=tk.LEFT)
            self.max_val.append(tk.Entry(inner_frame, width=10))
            self.max_val[i].insert(0, str(root.max_list[i]))
            self.max_val[i].pack(side=tk.LEFT)

        tk.Button(self.window, text='apply', command=lambda: self.apply(root)).pack()
        tk.Label(self.window, text='')

    def apply(self, root):
        for i in range(self.var_num):
            root.max_list[i] = float(self.max_val[i].get())
            root.min_list[i] = float(self.min_val[i].get())
        root.param_change()
        self.window.destroy()


class New_Graph_Page:
    def __init__(self, root):
        self.window = tk.Toplevel(root.app)
        self.window.geometry('1280x720')
        self.window.title('Gaussian Process')
        self.window.resizable(False, False)

        self.data_file = None
        self.data_filename = None
        self.data_file_s = None
        self.var_num = 1

        self.graph_type = ttk.Combobox(self.window, height=5, values=["2D", "3D"])
        self.model_type = ttk.Combobox(self.window, height=5, values=["sklearn", "custom"])
        self.graph_type.current(0)
        self.model_type.current(0)

        self.select_var_num = 1
        self.select_var = [0]

        self.x_list = []
        self.nx_list = []
        self.max_list = []
        self.min_list = []

        for i in range(self.var_num):
            self.x_list.append(np.transpose(np.array([[0.0]])))
            self.max_list.append(6.4)
            self.min_list.append(0.0)
            self.nx_list.append(self.x_list[i] / self.max_list[i])

        self.rx = np.array([[0.0]])
        self.rx = np.transpose(self.rx)
        self.y = np.array([0.0])

        self.x_max = 6.4
        self.x = self.rx / self.x_max
        self.max_list.append(80.0)
        self.min_list.append(0.0)
        self.y_max = 80.0
        self.y_min = 0.0
        self.ny = self.y / self.y_max

        self.data_init()

        print(self.x)

        self.x = []
        for i in range(len(self.y)):
            self.x.append([])
            for f in range(self.var_num):
                self.x[i].append(self.x_list[i][f] / self.max_list[f])
        print(self.x)
        self.x = np.array(self.x)

        self.kernel_noise = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
        self.model = GaussianProcessRegressor(kernel=self.kernel_noise)
        self.model.fit(self.x, self.ny)
        self.x_test = np.linspace(0.0, 1.0, 101).reshape(-1, 1)
        self.y_pred, self.std_dev = self.model.predict(self.x_test, return_std=True)

        self.fig_A = plt.figure(1)
        self.ax_A = plt.axes()
        self.figure_A = FigureCanvasTkAgg(self.fig_A, master=self.window)
        self.figure_A_widget = self.figure_A.get_tk_widget()
        self.figure_A_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        plt.subplot(211)
        plt.scatter(self.x * self.x_max, self.ny * self.y_max, c='red', label='Training Data')
        plt.plot(self.x_test * self.x_max, self.y_pred * self.y_max, c='blue', label='Predictive Mean')
        plt.fill_between(self.x_test.flatten() * self.x_max, (self.y_pred - 1.96 * self.std_dev) * self.y_max,
                         (self.y_pred + 1.96 * self.std_dev) * self.y_max,
                         color='orange', alpha=0.3, label='Uncertainty')
        plt.legend()

        self.pi = []
        self.ei = []
        self.best_pi_x = 0
        self.best_ei_x = 0

        self.acquisition()

        '''
        self.fig_B = plt.figure ( 2 )
        self.ax_B = plt.axes ( )
        self.figure_B = FigureCanvasTkAgg ( self.fig_B, master=self.window )
        self.figure_B_widget = self.figure_B.get_tk_widget ( )
        self.figure_B_widget.pack ( side=tk.BOTTOM, fill=tk.BOTH, expand=1 )
        '''
        plt.subplot(212)
        plt.plot(self.x_test * self.x_max, self.pi, label='PI', color='purple')
        plt.plot(self.x_test * self.x_max, self.ei, label='EI', color='green')

        plt.axvline(self.best_pi_x * self.x_max, color='purple', linestyle='--', label='next (PI)')
        plt.axvline(self.best_ei_x * self.x_max, color='green', linestyle='--', label='next (EI)')
        plt.legend()

        self.label = tk.Label(self.window, text='Curve1', width=50)
        self.label.pack(side=tk.TOP)
        self.label2 = tk.Label(self.window, text='Curve2', width=50)
        self.label2.pack(side=tk.TOP)

        self.graph_type.pack(side=tk.TOP)
        self.model_type.pack(side = tk.TOP)
        tk.Button(self.window, text='apply', command=self.draw_graph).pack(side=tk.TOP)

        tk.Button(self.window, text='select data.txt', command=self.file_select).pack(side=tk.TOP)

        tk.Button(self.window, text='setting', command=self.set_page).pack(side=tk.BOTTOM)
        tk.Label(self.window, text='Curve2', width=50).pack(side=tk.BOTTOM)
        tk.Label(self.window, text='Curve2', width=50).pack(side=tk.BOTTOM)
        tk.Label(self.window, text='Curve2', width=50).pack(side=tk.BOTTOM)
        tk.Label(self.window, text='Curve2', width=50).pack(side=tk.BOTTOM)
        tk.Label(self.window, text='Curve2', width=50).pack(side=tk.BOTTOM)
        tk.Label(self.window, text='Curve2', width=50).pack(side=tk.BOTTOM)
        tk.Label(self.window, text='Curve2', width=50).pack(side=tk.BOTTOM)
        tk.Label(self.window, text='Curve2', width=50).pack(side=tk.BOTTOM)

        self.canvas = tk.Canvas(self.window)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar = tk.Scrollbar(self.window, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.table_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.table_frame, anchor="nw")
        self.cells = []
        self.create_table()

    def draw_graph(self):
        self.x_max = self.max_list[0]
        self.x = []
        for i in range(len(self.x_list)):
            print("wow")
            self.x.append([float(self.x_list[i][0]) / self.x_max])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.model = GaussianProcessRegressor(kernel=self.kernel_noise)
        self.ny = self.y / self.y_max
        self.model.fit(self.x, self.ny)
        self.x_test = np.linspace(0.0, 1.0, 101).reshape(-1, 1)
        self.y_pred, self.std_dev = self.model.predict(self.x_test, return_std=True)

        self.fig_A.clear()
        plt.subplot(211)
        plt.scatter(self.x * self.x_max, self.y, c='red', label='Training Data')
        plt.plot(self.x_test * self.x_max, self.y_pred * self.y_max, c='blue', label='Predictive Mean')
        plt.fill_between(self.x_test.flatten() * self.x_max, (self.y_pred - 1.96 * self.std_dev) * self.y_max,
                         (self.y_pred + 1.96 * self.std_dev) * self.y_max,
                         color='orange', alpha=0.3, label='Uncertainty')
        plt.legend()

        plt.subplot(212)
        plt.plot(self.x_test * self.x_max, self.pi, label='PI', color='purple')
        plt.plot(self.x_test * self.x_max, self.ei, label='EI', color='green')

        plt.axvline(self.best_pi_x * self.x_max, color='purple', linestyle='--', label='next (PI)')
        plt.axvline(self.best_ei_x * self.x_max, color='green', linestyle='--', label='next (EI)')
        plt.legend()

        self.figure_A.draw()
        print("finish")

    def create_table(self):
        for cell in self.cells:
            cell.destroy()
        self.cells = []
        print(self.y)
        for i in range(len(self.y)):
            cell = tk.Label(self.table_frame, text=f"{i + 1}", padx=10, pady=5, borderwidth=1,
                            relief="solid")
            cell.grid(row=i, column=0, sticky="nsew")
            self.cells.append(cell)
            for f in range(self.var_num):
                cell = tk.Label(self.table_frame, text=f"{self.x_list[i][f]}", padx=10, pady=5, borderwidth=1,
                                relief="solid")
                cell.grid(row=i, column=f + 1, sticky="nsew")
                self.cells.append(cell)
            cell = tk.Label(self.table_frame, text=f"{self.y[i]}", padx=10, pady=5, borderwidth=1,
                            relief="solid")
            cell.grid(row=i, column=self.var_num + 1, sticky="nsew")
            self.cells.append(cell)

        self.table_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        for i in range(self.var_num + 2):
            self.table_frame.grid_columnconfigure(i, weight=1)

    def data_init(self):
        self.rx = np.array([[1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 0.0]])
        self.x_list = []
        for i in range(len(self.rx[0])):
            self.x_list.append([self.rx[0][i]])
        self.rx = np.transpose(self.rx)
        print(self.rx)
        self.y = np.array([66.79, 67.33, 65.34, 55.97, 53.26, 48.81, 45, 0.0])

        self.x_max = 6.4
        self.x = self.rx / self.x_max
        self.y_max = 100.0
        self.y_min = 20.0
        self.ny = self.y / self.y_max

    def file_select(self):
        self.data_file = filedialog.askopenfile(title='파일 선택',
                                                filetypes=(('txt files', '*.txt'), ('all files', '*.*')))
        print(type(self.data_file))
        self.file_load()

    def file_load(self):
        f = open(self.data_file.name, "r")
        lines = f.readlines()
        self.x_list = []
        self.y = []
        max = []
        min = []

        cnt = 0
        print()

        for line in lines:
            if cnt == 0:
                self.var_num = int(line)
                cnt += 1
                continue
            max = np.zeros(self.var_num + 1)
            min = np.zeros(self.var_num + 1)
            row = line.split(' ')
            self.x_list.append([])
            for i in range(self.var_num):
                self.x_list[cnt - 1].append(row[i])
                if cnt == 1:
                    max[i] = float(row[i])
                    min[i] = float(row[i])
                else:
                    if max[i] < float(row[i]):
                        max[i] = float(row[i])
                    if min[i] > float(row[i]):
                        min[i] = float(row[i])
            yv = row[self.var_num].split('\n')[0]
            self.y.append(float(yv))

            if cnt == 1:
                max[self.var_num] = float(yv)
                min[self.var_num] = float(yv)
            else:
                if max[self.var_num] < float(yv):
                    max[self.var_num] = float(yv)
                if min[self.var_num] > float(yv):
                    min[self.var_num] = float(yv)
            cnt += 1
        self.min_list = min
        self.max_list = max
        self.create_table()
        self.draw_graph()

    def file_save(self):
        if self.data_file != None:
            self.file_save2();
        else:
            self.data_file_s = asksaveasfile(mode="w", defaultextension=".txt")
            if self.data_file_s is None:
                return
            else:
                self.data_filename = self.data_file_s.name
                self.file_save2()

    def file_save2(self):
        f = open(self.data_file.name, "w")

    def set_page(self):
        self.new_set_page = Parm_Set_Page(self)
        self.new_set_page.window.grab_set()

    def param_change(self):
        print(self.x_max)

    def set_widget(self):
        a = 0

    def opt_acq_pi(self, x, mu, std, model):
        best = max(mu);

        scores = norm.cdf((mu - best) / (std + 1e-9))
        index = np.argmax(scores)
        return x[index, 0], scores

    def opt_acq_ei(self, x, mu, std, model):
        best = max(mu);
        z = (mu - best) / (std + 1e-9)
        scores = (mu - best) * norm.cdf(z) + std * norm.pdf(z)
        index = np.argmax(scores)
        return x[index, 0], scores

    def acquisition(self):
        for i in range(1):
            self.best_pi_x, self.pi = self.opt_acq_pi(self.x_test, self.y_pred, self.std_dev, self.model)
            self.best_ei_x, self.ei = self.opt_acq_ei(self.x_test, self.y_pred, self.std_dev, self.model)
            print("next ratio (PI)")
            print(self.best_pi_x * self.x_max)
            print("next ratio (EI)")
            print(self.best_ei_x * self.x_max)


class New_Model_Page:
    def __init__(self, root):
        self.window = tk.Toplevel(root.app)
        self.window.geometry('200x600')
        self.window.title('Create New Model')
        self.window.resizable(False, False)
        self.window.attributes('-topmost', 'true')
        self.window.grab_set()
        self.select_algorithm = []
        self.layer_button = tk.Button(self.window, text='Add new Layer', command=self.add_layer)
        self.act_button = tk.Button(self.window, text='Add new Activaion', command=self.add_act)
        self.layer_button.pack()
        self.act_button.pack()
        self.layer_list = (
            "Linear", "Bilinear", "Conv1d", "Conv2d", "Conv3d", "MaxPool1d", "MaxPool2d", "MaxPool3d", "ZeroPad2d",
            "ReflectPad1d", "ReflectPad2d")
        self.act_list = ("Relu", "Sigmoid", "Tanh", "Softplus", "Threshold", "Softmin", "Softmax")
        self.layer_combobox = []
        self.act_combobox = []
        self.combobox_list = []
        self.layer_dim = []

        self.apply_button = tk.Button(self.window, text='Apply', command=lambda: self.apply(root)).pack()
        self.delete_button = tk.Button(self.window, text='Delete', command=self.delete).pack()

    def add_layer(self):
        combobox = ttk.Combobox(self.window, values=self.layer_list)
        combobox.pack()
        combobox.current(0)
        layerdim = tk.Entry(self.window, width=10)
        layerdim.pack()
        self.layer_combobox.append(combobox)
        self.combobox_list.append(['l'])
        self.layer_dim.append(layerdim)

    def add_act(self):
        combobox = ttk.Combobox(self.window, values=self.act_list)
        combobox.pack()
        combobox.current(0)
        self.act_combobox.append(combobox)
        self.combobox_list.append(['a'])

    def delete(self):
        if not self.combobox_list:
            return
        elem = self.combobox_list.pop()
        if elem[0] == 'l':
            self.layer_combobox.pop().destroy()
            self.layer_dim.pop().destroy()
        elif elem[0] == 'a':
            self.act_combobox.pop().destroy()

    def apply(self, root):
        slist = []
        layer_cnt = 0
        act_cnt = 0
        for i in self.combobox_list:
            if i[0] == 'l':
                slist.append(
                    [self.layer_combobox[layer_cnt].get()] + self.layer_dim[layer_cnt].get().split(' '))
                layer_cnt += 1
            elif i[0] == 'a':
                slist.append([self.act_combobox[act_cnt].get()])
                act_cnt += 1
        print(slist)
        root.text.set("".join(sum(slist, [])))
        self.window.destroy()
        self.window.quit()
        exit()


class Main_Page:
    def __init__(self):
        # region Window Setting
        self.app = tk.Tk()
        self.app.title("Model Generator")
        self.width = 400
        self.height = 400
        # self.fig = plt.plot([0,1,2,3,4],[0,1,2,3,4])
        # self.canvas = FigureCanvasTkAgg ( self.fig, master=self.app )
        self.canvas = tk.Canvas(self.app, width=self.width, height=self.height)
        self.app.resizable(False, False)
        self.canvas.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)

        self.running = True
        # endregion Window Setting

        # region Menu Bar
        self.menu_visible = tk.IntVar()
        self.menu_visible.set(1)
        self.menubar = tk.Menu(self.app)
        self.app.config(menu=self.menubar)
        self.menu_file = tk.Menu(self.menubar, tearoff=False)
        self.menu_edit = tk.Menu(self.menubar, tearoff=False)
        self.menu_view = tk.Menu(self.menubar, tearoff=False)
        self.view_appearance = tk.Menu(self.menu_view, tearoff=False)
        self.menu_help = tk.Menu(self.menubar, tearoff=False)

        self.set_menu()
        # endregion Menu Bar

        self.sequential_list = []
        self.text = tk.StringVar()
        self.text.set("wow")
        tk.Label(self.app, textvariable=self.text).pack()

        self.new_graph()

    def menufunc(self):
        pass

    def set_menu(self):
        self.menubar.add_cascade(label="file", menu=self.menu_file)
        self.menubar.add_cascade(label="edit", menu=self.menu_edit)
        self.menubar.add_cascade(label="view", menu=self.menu_view)
        self.menubar.add_cascade(label="help", menu=self.menu_help)

        self.menu_file.add_command(label="New Graph", command=self.new_graph)
        self.menu_file.add_command(label="New Model", command=self.new_model)
        self.menu_file.add_command(label="Open", command=self.menufunc)
        self.menu_file.add_command(label="Save As", command=self.menufunc)
        self.menu_file.add_command(label="Save", command=self.menufunc)
        self.menu_file.add_separator()
        self.menu_file.add_command(label="Exit", command=self.quit)

        self.menu_edit.add_command(label="Undo", command=self.menufunc)
        self.menu_edit.add_command(label="Redo", command=self.menufunc)

        self.menu_view.add_cascade(label="Appearance", menu=self.view_appearance)
        self.menu_view.add_checkbutton(label="Menu bar", variable=self.menu_visible)

        self.view_appearance.add_command(label="theme", command=self.menufunc)

        self.menu_help.add_command(label="Watch Help", command=self.menufunc)
        self.menu_help.add_command(label="Information", command=self.menufunc)

    def new_graph(self):
        self.new_graph_page = New_Graph_Page(self)
        self.new_graph_page.window.grab_set()

    def new_model(self):
        self.new_model_page = New_Model_Page(self)
        self.new_model_page.window.grab_set()

    def update(self):
        time.sleep(update_time)
        self.app.update()

    def quit(self):
        if self.running:
            self.running = False
        else:
            self.app.destroy()
            self.app.quit()
            exit()


if __name__ == '__main__':
    mainwin = Main_Page()

    mainwin.app.protocol('WM_DELETE_WINDOW', mainwin.quit)

    while 1:
        if mainwin.running:
            mainwin.update()
        else:
            mainwin.quit()
            break
