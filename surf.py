# https://en.wikipedia.org/wiki/Surface_%28topology%29#/media/File:Saddle_Point.png
# z = 1/2 * cos(x/2) + sin(y/4)


import tkinter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import tkinter as tk
import tkinter.ttk as ttk
import copy

class surface1:
    def __init__(self,delta):
        self.delta = delta

    def x(self,u,v):
        return (u+v)/30

    def y(self,u,v):
        return (u-v)/30

    def z(self,u,v):
        #return ((u+v)/60)**2+((u-v)/60)**2
        return (0.5*np.cos((u+v)/2)+np.sin((u-v)/4))/3

    def x1(self,u):
        return (self.x(u+self.delta/2)-self.x(u-self.delta/2))/self.delta

    def y1(self,u):
        return (self.y(u+self.delta/2)-self.y(u-self.delta/2))/self.delta

    def z1(self,u):
        return (self.z(u+self.delta/2)-self.z(u-self.delta/2))/self.delta

    def x2(self,u):
        return (self.x1(u+self.delta/2)-self.x1(u-self.delta/2))/self.delta

    def y2(self,u):
        return (self.y1(u+self.delta/2)-self.y1(u-self.delta/2))/self.delta

    def z2(self,u):
        return (self.z1(u+self.delta/2)-self.z1(u-self.delta/2))/self.delta

    def silverman(self,x):
        n = len(x)
        std_x = np.std(x)
        iqr_x = np.percentile(x,75)-np.percentile(x,25)
        h = 0.9*min(std_x,iqr_x/1.34)*n**(-0.2)
        return h

    def kernel_size(self):
        return 0.1

    def gkernel(self,sigma,value):
        return np.exp(-(value)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

    def correntropy(self,sigma,values):
        n = len(values)
        sum = 0.0
        for i in range(n):
            sum += self.gkernel(sigma,values[i])
        return sum/n

    def regression(self,X_poly,z_value,epochs):
        coeff = np.array([[0.1,0.1,0.1,0.1,0.1,0.1]])
        delta = 0.000001
        learning_rate = 0.01
        print(coeff)
        for i in range(epochs):
            z_func = coeff.dot(np.transpose(X_poly))[0]
            diff = z_func - z_value
            k_size = self.silverman(diff)
            cost = self.correntropy(k_size,diff)
            grad_matrix = np.array([[0.0,0.0,0.0,0.0,0.0,0.0]])
            for f in range(6):
                pos_delta_coeff = copy.deepcopy(coeff)
                pos_delta_coeff[0][f]+=delta
                neg_delta_coeff = copy.deepcopy(coeff)
                neg_delta_coeff[0][f] -= delta
                pos_diff = pos_delta_coeff.dot(np.transpose(X_poly))[0]-z_value
                neg_diff = neg_delta_coeff.dot(np.transpose(X_poly))[0]-z_value
                pos_size = self.silverman(pos_diff)
                neg_size = self.silverman(neg_diff)
                print(self.correntropy(pos_size,pos_diff))
                grad_matrix[0][f] = (self.correntropy(pos_size,pos_diff)-self.correntropy(neg_size,neg_diff))/(2*delta)
            coeff[0] = coeff[0]+grad_matrix[0]*0.001
            if i%100==0:
                print("cost : "+str(cost))
                print(grad_matrix[0])
        return coeff[0]





    def curl(self,u,v,pnum):
        half = int(pnum/2)
        pointu = np.array([u+(i-half)*self.delta for i in range(pnum) for f in range(pnum)])
        pointv = np.array([v+(f-half)*self.delta for i in range(pnum) for f in range(pnum)])
        ptx = self.x(pointu,pointv)
        pty = self.y(pointu,pointv)
        ptz = self.z(pointu,pointv)
        points = np.array([[ptx[i],pty[i],ptz[i]] for i in range(pnum*pnum)])
        print(pointu)
        print(pointv)
        print(points[:,:2])
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(points[:,:2])
        print(X_poly)

        coefficients = self.regression(X_poly,points[:,2],1000)

        print("Coeff : ", coefficients)

        a = coefficients[3]
        b = coefficients[5]
        c = coefficients[4]
        d = coefficients[1]
        e = coefficients[2]
        f = coefficients[0]

        x = self.x(u,v)
        y = self.y(u,v)


        xu = np.array([1,0,2*a*x+c*y+d])
        xv = np.array([0,1,2*b*y+c*x+e])

        N = np.cross(xu,xv)
        print(N)
        N = N/np.linalg.norm(N)
        print(N)

        ee = N[2]*2*a
        ff = N[2]*c
        gg = N[2]*2*b

        eE = 1+xu[2]**2
        fF = xu[2]*xv[2]
        gG = 1+xv[2]**2

        aa = fF*gg-gG*ff
        bb = eE*gg-gG*ee
        cc = eE*ff-fF*ee

        lamda_1 = (-bb+np.sqrt(bb*bb-4*aa*cc))/(2*aa)
        lamda_2 = (-bb - np.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)

        curl_1 = (ee+2*ff+lamda_1+gg*(lamda_1**2))/(eE+2*fF*lamda_1+gG*(lamda_1**2))
        curl_2 = (ee + 2 * ff + lamda_2 + gg * (lamda_2 ** 2)) / (eE + 2 * fF * lamda_2 + gG * (lamda_2 ** 2))

        return N/10,curl_1,curl_2





window = tk.Tk()
window.title("Graph Setting")
window.geometry("1280x900+100+100")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

pu = np.linspace(0, 15, 101)
pv = np.linspace(0, 15, 101)
pcu,pcv = np.meshgrid(pu,pv)
print(pcu)
print(pcv)


canvas = FigureCanvasTkAgg(fig, master=window)
canvas_widget = canvas.get_tk_widget()
print("wow")
canvas_widget.pack(side=tk.LEFT,fill=tk.BOTH, expand=1)

label = tkinter.Label(window,text='Curve1',width=50)
label.pack()

func_list = ['Curve 1']
combobox = ttk.Combobox(window,values=func_list)
combobox.set(func_list[0])
combobox.pack()

func = surface1(0.3)

xvalue = func.x(pcu,pcv)
yvalue = func.y(pcu,pcv)
zvalue = func.z(pcu,pcv)

ax.plot_surface(xvalue, yvalue, zvalue, cmap='viridis',alpha = 0.7)

def calculate():
    global px, py, pz, nv, curl
    global x_label, y_label, z_label, curl_label
    u_val = u_slide.get()
    v_val = v_slide.get()
    value1 = "u : " + str(u_val)
    u_label.config(text=value1)
    value2 = "v : " + str(v_val)
    v_label.config(text=value2)
    px = func.x(u_val, v_val)
    py = func.y(u_val, v_val)
    pz = func.z(u_val, v_val)
    nv, curl1, curl2 = func.curl(u_val, v_val, 5)
    x_label.config(text="x : " + str(px))
    y_label.config(text="y : " + str(py))
    z_label.config(text="z : " + str(pz))
    curl_label.config(text="curl : " + str(curl1) + " ; " + str(curl2))

    ax.clear()
    ax.set_box_aspect([1, 1, 1])
    ax.plot_surface(xvalue, yvalue, zvalue, cmap='viridis', alpha=0.7)
    ax.plot3D(px, py, pz, 'o', c='g')
    curl_mean = (curl1 + curl2) / 2
    ax.plot3D([px, px + nv[0] * curl_mean], [py, py + nv[1] * curl_mean], [pz, pz + nv[2] * curl_mean], c='r')
    canvas.draw()

def u_select(self):
    global px, py, pz, nv, curl
    global x_label, y_label, z_label, curl_label
    u_val = u_slide.get()
    v_val = v_slide.get()
    value1 = "u : "+str(u_val)
    u_label.config(text=value1)
    value2 = "v : " + str ( v_val )
    v_label.config ( text=value2 )
    px = func.x(u_val,v_val)
    py = func.y(u_val,v_val)
    pz = func.z(u_val,v_val)
    x_label.config(text="x : "+str(px))
    y_label.config(text="y : "+str(py))
    z_label.config(text="z : "+str(pz))

    ax.clear()
    ax.set_box_aspect([1, 1, 1])
    ax.plot_surface ( xvalue, yvalue, zvalue, cmap='viridis', alpha=0.7 )
    ax.plot3D(px, py, pz, 'o', c='g')
    canvas.draw()

u_var = tkinter.StringVar
v_var = tkinter.StringVar

u_slide = tkinter.Scale(window,variable = u_var,command=u_select,orient="horizontal",showvalue=False,tickinterval=50,to=15.0,length=200,resolution=0.01)
u_slide.pack()

u_label = tkinter.Label(window,text="u : 0")
u_label.pack()

v_slide = tkinter.Scale(window,variable = v_var,command=u_select,orient="horizontal",showvalue=False,tickinterval=50,to=15.0,length=200,resolution=0.01)
v_slide.pack()

v_label = tkinter.Label(window,text="v : 0")
v_label.pack()


u_val = u_slide.get()
v_val = v_slide.get()
px = func.x(u_val,v_val)
py = func.y(u_val,v_val)
pz = func.z(u_val,v_val)

ax.plot3D(px,py,pz,'o',c='g')

nv,curl1,curl2 = func.curl(u_val,v_val,5)
print('normal vector : ',nv)
curl_mean = (curl1+curl2)/2
ax.plot3D([px,px+nv[0]*curl_mean],[py,py+nv[1]*curl_mean],[pz,pz+nv[2]*curl_mean],c='r')

x_label = tkinter.Label(window,text="x : "+str(px))
y_label = tkinter.Label(window,text="y : "+str(py))
z_label = tkinter.Label(window,text="z : "+str(pz))

curl_button = tkinter.Button(window,text="Calculate",command = calculate)

curl_label = tkinter.Label(window,text = "curl : "+str(curl1)+' ; '+str(curl2))
x_label.pack()
y_label.pack()
z_label.pack()
curl_button.pack()
curl_label.pack()

window.mainloop()
