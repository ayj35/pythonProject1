import tkinter
import tkinter as tk
import tkinter.ttk as ttk
import math
import time

update_time = 0.01

class New_Model_Page:
    def __init__(self,root):
        self.window = tk.Toplevel(root.app)
        self.window.geometry('200x600')
        self.window.title('Create New Model')
        self.window.resizable(False,False)
        self.window.attributes('-topmost','true')
        self.window.grab_set()
        self.layer_button = tk.Button(self.window,text='Add new Layer',command = self.add_layer)
        self.act_button = tk.Button(self.window,text='Add new Activaion',command = self.add_act)
        self.layer_button.pack()
        self.act_button.pack()
        self.layer_list = ("Linear","Bilinear","Conv1d","Conv2d","Conv3d","MaxPool1d","MaxPool2d","MaxPool3d","ZeroPad2d","ReflectPad1d","ReflectPad2d")
        self.act_list = ("Relu","Sigmoid","Tanh","Softplus","Threshold","Softmin","Softmax")
        self.layer_combobox = []
        self.act_combobox = []
        self.combobox_list = []
        self.layer_dim = []

        self.apply_button = tk.Button(self.window,text='Apply',command=lambda : self.apply(root)).pack()
        self.delete_button = tk.Button(self.window,text='Delete',command=self.delete).pack()

    def add_layer(self):
        combobox = ttk.Combobox(self.window,values=self.layer_list)
        combobox.pack()
        combobox.current(0)
        layerdim = tk.Entry(self.window,width = 10)
        layerdim.pack()
        self.layer_combobox.append(combobox)
        self.combobox_list.append(['l'])
        self.layer_dim.append(layerdim)
    def add_act(self):
        combobox = ttk.Combobox(self.window,values=self.act_list)
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
    def apply(self,root):
        slist = []
        layer_cnt=0
        act_cnt=0
        for i in self.combobox_list:
            if i[0] == 'l':
                slist.append([self.layer_combobox[layer_cnt].get()]+self.layer_dim[layer_cnt].get().split(' '))
                layer_cnt+=1
            elif i[0] == 'a':
                slist.append([self.act_combobox[act_cnt].get()])
                act_cnt+=1
        print(slist)
        root.text.set("".join(sum(slist,[])))
        self.window.destroy()
        self.window.quit()
        exit()



class Main_Page:
    def __init__(self):
        #region Window Setting
        self.app = tk.Tk()
        self.app.title("Model Generator")
        self.width = 400
        self.height = 400
        self.canvas = tk.Canvas ( self.app, width = self.width, height = self.height )
        self.app.resizable(False,False)
        self.canvas.pack (side = tk.RIGHT, expand = tk.YES, fill = tk.BOTH)

        self.running = True
        #endregion Window Setting

        #region Menu Bar
        self.menu_visible = tk.IntVar()
        self.menu_visible.set(1)
        self.menubar = tk.Menu(self.app)
        self.app.config(menu=self.menubar)
        self.menu_file = tk.Menu(self.menubar,tearoff = False)
        self.menu_edit = tk.Menu(self.menubar,tearoff = False)
        self.menu_view = tk.Menu(self.menubar,tearoff = False)
        self.view_appearance = tk.Menu(self.menu_view, tearoff=False)
        self.menu_help = tk.Menu(self.menubar,tearoff = False)

        self.set_menu()
        #endregion Menu Bar

        self.sequential_list=[]
        self.text = tk.StringVar()
        self.text.set("wow")
        tk.Label(self.app,textvariable=self.text).pack()


    def menufunc(self):
        pass
    def set_menu(self):
        self.menubar.add_cascade(label="file", menu=self.menu_file)
        self.menubar.add_cascade(label="edit", menu=self.menu_edit)
        self.menubar.add_cascade(label="view", menu=self.menu_view)
        self.menubar.add_cascade(label="help", menu=self.menu_help)

        self.menu_file.add_command(label="New Model", command=self.new_model)
        self.menu_file.add_command(label="Open", command=self.menufunc)
        self.menu_file.add_command(label="Save As", command=self.menufunc)
        self.menu_file.add_command(label="Save", command=self.menufunc)
        self.menu_file.add_separator()
        self.menu_file.add_command(label="Exit", command=self.quit)

        self.menu_edit.add_command(label="Undo", command=self.menufunc)
        self.menu_edit.add_command(label="Redo", command=self.menufunc)

        self.menu_view.add_cascade(label="Appearance",menu=self.view_appearance)
        self.menu_view.add_checkbutton(label="Menu bar", variable=self.menu_visible)

        self.view_appearance.add_command(label="theme",command=self.menufunc)

        self.menu_help.add_command(label="Watch Help",command=self.menufunc)
        self.menu_help.add_command(label="Information",command=self.menufunc)

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

if __name__=='__main__':
    mainwin = Main_Page()

    mainwin.app.protocol('WM_DELETE_WINDOW', mainwin.quit)

    while 1:
        if mainwin.running:
            mainwin.update()
        else:
            mainwin.quit()
            break