from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import scipy
import time


#------------ variables ----------------
L = 1                
g = 9.81  
L1 = 1
L2 = 1
m1 = 1
m2 = 1

#------------ time step -----------------
time_max, dt = 40, 0.01
t = np.arange(0, time_max + dt, dt)

#------------ initial conditions --------
y01 = np.array([-3.14, 0, -3.14, 0 ,0,0,0,0,0,0,0,0]) #theta1, thetadot1, theta2, thetadot2
y02 = np.array([-3.141, 0, -3.141, 0,0,0,0,0,0,0,0,0]) #theta1, thetadot1, theta2, thetadot2
y03 = np.array([-3.142, 0, -3.142, 0,0,0,0,0,0,0,0,0]) #theta1, thetadot1, theta2, thetadot2

#------------ Animation values ----------
fps = 10
time_max, dt = 100, 0.01
t = np.arange(0, time_max + dt, dt)
frame_step = int(1/fps/dt)

#------------ shape values --------------
r = 0.05



class multiple_double_pendulum:
    ''' '''

    def __init__(self, L1, L2, m1, m2, g):
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g

        '''
        Parameters
        ----------
        L : scalar, float
            Pendulum rod length (m)
        m : scalar, float
            Mass of the pendulum bob (Kg)
        g : scalar, float
            Gravitational constant (ms^-2)
        '''


    def derivative(self, y, t, L1, L2, m1, m2, g):
        '''Returns the first derivatives of y'''
        theta11, thetadot11, theta21, thetadot21,a,a,a,a,a,a,a,a = y
        theta12, thetadot12, theta22, thetadot22,a,a,a,a,a,a,a,a = y
        theta13, thetadot13, theta23, thetadot23,a,a,a,a,a,a,a,a = y


        dtheta_dt11 = thetadot11
        dtheta_dt21 = thetadot21
        dtheta_dt12 = thetadot12
        dtheta_dt22 = thetadot22
        dtheta_dt13 = thetadot13
        dtheta_dt23 = thetadot23


        dthetadot_dt11 = ((m2*g*np.sin(theta21)*np.cos(theta11-theta21) 
        - m2*np.sin(theta11-theta21)*(L1*thetadot11**2*np.cos(theta11-theta21) 
        + L2*thetadot21**2) - (m1+m2)*g*np.sin(theta11)) / L1 / (m1 + m2*np.sin(theta11-theta21)**2))

        dthetadot_dt21 = (((m1+m2)*(L1*thetadot11**2*np.sin(theta11-theta21) - g*np.sin(theta21) 
        + g*np.sin(theta11)*np.cos(theta11-theta21)) 
        + m2*L2*thetadot21**2*np.sin(theta11-theta21)*np.cos(theta11-theta21)) / L2 / 
        (m1 + m2*np.sin(theta11-theta21)**2))


        dthetadot_dt12 = ((m2*g*np.sin(theta22)*np.cos(theta12-theta22) 
        - m2*np.sin(theta12-theta22)*(L1*thetadot12**2*np.cos(theta12-theta22) 
        + L2*thetadot22**2) - (m1+m2)*g*np.sin(theta12)) / L1 / (m1 + m2*np.sin(theta12-theta22)**2))

        dthetadot_dt22 = (((m1+m2)*(L1*thetadot12**2*np.sin(theta12-theta22) - g*np.sin(theta22) 
        + g*np.sin(theta12)*np.cos(theta12-theta22)) 
        + m2*L2*thetadot22**2*np.sin(theta12-theta22)*np.cos(theta12-theta22)) / L2 / 
        (m1 + m2*np.sin(theta12-theta22)**2))


        dthetadot_dt13 = ((m2*g*np.sin(theta23)*np.cos(theta13-theta23) 
        - m2*np.sin(theta13-theta23)*(L1*thetadot13**2*np.cos(theta13-theta23) 
        + L2*thetadot23**2) - (m1+m2)*g*np.sin(theta13)) / L1 / (m1 + m2*np.sin(theta13-theta23)**2))

        dthetadot_dt23 = (((m1+m2)*(L1*thetadot13**2*np.sin(theta13-theta23) - g*np.sin(theta23) 
        + g*np.sin(theta13)*np.cos(theta13-theta23)) 
        + m2*L2*thetadot23**2*np.sin(theta13-theta23)*np.cos(theta13-theta23)) / L2 / 
        (m1 + m2*np.sin(theta13-theta23)**2))

        return (dtheta_dt11, dthetadot_dt11, dtheta_dt21, dthetadot_dt21,
               dtheta_dt12, dthetadot_dt12, dtheta_dt22, dthetadot_dt22,
               dtheta_dt13, dthetadot_dt13, dtheta_dt23, dthetadot_dt23)


    def differential_equation(self):
        '''Returns a numerical solution to the differential equation'''
        y1 = odeint(self.derivative, y01, t, args=(self.L1, self.L2, self.m1, 
        self.m2, self.g))

        y2 = odeint(self.derivative, y02, t, args=(self.L1, self.L2, self.m1, 
        self.m2, self.g))

        y3 = odeint(self.derivative, y03, t, args=(self.L1, self.L2, self.m1, 
        self.m2, self.g))


        #Coordinate transformation
        self.theta11, self.theta21 = y1[:,0], y1[:,2]
        self.x11 = L1 * np.sin(self.theta11)
        self.y11 = -L1 * np.cos(self.theta11)
        self.x21 = self.x11 + L2 * np.sin(self.theta21)
        self.y21 = self.y11 - L2 * np.cos(self.theta21)


        self.theta12, self.theta22 = y2[:,0], y2[:,2]
        self.x12 = L1 * np.sin(self.theta12)
        self.y12 = -L1 * np.cos(self.theta12)
        self.x22 = self.x12 + L2 * np.sin(self.theta22)
        self.y22 = self.y12 - L2 * np.cos(self.theta22)


        self.theta13, self.theta23 = y3[:,0], y3[:,2]
        self.x13 = L1 * np.sin(self.theta13)
        self.y13 = -L1 * np.cos(self.theta13)
        self.x23 = self.x13 + L2 * np.sin(self.theta23)
        self.y23 = self.y13 - L2 * np.cos(self.theta23)

        return (self.x11, self.y11, self.theta11, self.x21, self.y21, self.theta21,
               self.x12, self.y12, self.theta12, self.x22, self.y22, self.theta22,
               self.x13, self.y13, self.theta13, self.x23, self.y23, self.theta23)


    def generate_plot(self):
        '''Animation of the plots and shows'''
        #----------- Set figure -----------
        #fig = plt.figure(figsize=(12, 4.5), dpi=80)
        #ax1 = fig.add_subplot(1,2,1)
        #ax2 = fig.add_subplot(1,2,2)

        gs = gridspec.GridSpec(2, 2)

        fig = plt.figure(figsize=(10, 10), dpi=80)
        ax1 = plt.subplot(gs[0, 0]) # row 0, col 0
        plt.plot([0,1])

        ax2 = plt.subplot(gs[0, 1]) # row 0, col 1
        plt.plot([0,1])

        ax3 = plt.subplot(gs[1, :]) # row 1, span all columns
        plt.plot([0,1])


        ax2.clear()
        ax3.clear()
        line11, = ax2.plot(self.x11, self.y11, '-', color='maroon')
        line21, = ax2.plot(self.x21, self.y21, '-', color='red')
        line31, = ax3.plot(t, self.theta11, '-', color='maroon')
        line41, = ax3.plot(t, self.theta21, '-', color='red')

        line12, = ax2.plot(self.x12, self.y12, '-', color='navy')
        line22, = ax2.plot(self.x22, self.y22, '-', color='deepskyblue')
        line32, = ax3.plot(t, self.theta12, '-', color='navy')
        line42, = ax3.plot(t, self.theta22, '-', color='deepskyblue')

        line13, = ax2.plot(self.x13, self.y13, '-', color='indigo')
        line23, = ax2.plot(self.x23, self.y23, '-', color='orchid')
        line33, = ax3.plot(t, self.theta13, '-', color='indigo')
        line43, = ax3.plot(t, self.theta23, '-', color='orchid')


        for i in range(0, len(t), frame_step):

            ax1.plot([0, self.x11[i], self.x21[i]], [0, self.y11[i], self.y21[i]])
            self.circle0 = Circle((0, 0), r/2, fc='b', zorder=10)
            self.circle1 = Circle((self.x11[i], self.y11[i]), r, fc='maroon', zorder=10)
            self.circle2 = Circle((self.x21[i], self.y21[i]), r, fc='red', zorder=10)
            ax1.add_patch(self.circle0)
            ax1.add_patch(self.circle1)
            ax1.add_patch(self.circle2)
            ax1.set_xlabel(r'$x$')
            ax1.set_ylabel(r'$y$')
            ax1.set_xlim(-2.1, 2.1)
            ax1.set_ylim(-2.1, 2.1)
            ax1.set_aspect('equal', adjustable='box')

            line11.set_xdata(self.x11[:i])
            line11.set_ydata(self.y11[:i])
            line21.set_xdata(self.x21[:i])
            line21.set_ydata(self.y21[:i])
            ax2.set_xlabel(r'$x$')
            ax2.set_ylabel(r'$y$')
            ax2.set_xlim(-2.1, 2.1)
            ax2.set_ylim(-2.1, 2.1)
            
            line31.set_xdata(t[:i])
            line41.set_xdata(t[:i])
            line31.set_ydata(self.theta11[:i])
            line41.set_ydata(self.theta21[:i])
            ax3.set_xlabel(r'$t$')
            ax3.set_ylabel(r'$\theta$')
            ax3.set_xlim(0, max(t))
            ax3.set_ylim(min(self.theta21) - 0.1, max(self.theta21) + 0.1)

            #pen 2
            ax1.plot([0, self.x12[i], self.x22[i]], [0, self.y12[i], self.y22[i]])
            self.circle0 = Circle((0, 0), r/2, fc='b', zorder=10)
            self.circle1 = Circle((self.x12[i], self.y12[i]), r, fc='navy', zorder=10)
            self.circle2 = Circle((self.x22[i], self.y22[i]), r, fc='deepskyblue', zorder=10)
            ax1.add_patch(self.circle0)
            ax1.add_patch(self.circle1)
            ax1.add_patch(self.circle2)
            ax1.set_xlabel(r'$x$')
            ax1.set_ylabel(r'$y$')
            ax1.set_xlim(-2.1, 2.1)
            ax1.set_ylim(-2.1, 2.1)
            ax1.set_aspect('equal', adjustable='box')

            line12.set_xdata(self.x12[:i])
            line12.set_ydata(self.y12[:i])
            line22.set_xdata(self.x22[:i])
            line22.set_ydata(self.y22[:i])
            ax2.set_xlabel(r'$x$')
            ax2.set_ylabel(r'$y$')
            ax2.set_xlim(-2.1, 2.1)
            ax2.set_ylim(-2.1, 2.1)
            
            line32.set_xdata(t[:i])
            line42.set_xdata(t[:i])
            line32.set_ydata(self.theta12[:i])
            line42.set_ydata(self.theta22[:i])
            ax3.set_xlabel(r'$t$')
            ax3.set_ylabel(r'$\theta$')
            ax3.set_xlim(0, max(t))
            ax3.set_ylim(min(self.theta22) - 0.1, max(self.theta22) + 0.1)


            #pen 3
            ax1.plot([0, self.x13[i], self.x23[i]], [0, self.y13[i], self.y23[i]])
            self.circle0 = Circle((0, 0), r/2, fc='b', zorder=10)
            self.circle1 = Circle((self.x13[i], self.y13[i]), r, fc='indigo', zorder=10)
            self.circle2 = Circle((self.x23[i], self.y23[i]), r, fc='orchid', zorder=10)
            ax1.add_patch(self.circle0)
            ax1.add_patch(self.circle1)
            ax1.add_patch(self.circle2)
            ax1.set_xlabel(r'$x$')
            ax1.set_ylabel(r'$y$')
            ax1.set_xlim(-2.1, 2.1)
            ax1.set_ylim(-2.1, 2.1)
            ax1.set_aspect('equal', adjustable='box')

            line13.set_xdata(self.x13[:i])
            line13.set_ydata(self.y13[:i])
            line23.set_xdata(self.x23[:i])
            line23.set_ydata(self.y23[:i])
            ax2.set_xlabel(r'$x$')
            ax2.set_ylabel(r'$y$')
            ax2.set_xlim(-2.1, 2.1)
            ax2.set_ylim(-2.1, 2.1)
            
            line33.set_xdata(t[:i])
            line43.set_xdata(t[:i])
            line33.set_ydata(self.theta13[:i])
            line43.set_ydata(self.theta23[:i])
            ax3.set_xlabel(r'$t$')
            ax3.set_ylabel(r'$\theta$')
            ax3.set_xlim(0, max(t))
            ax3.set_ylim(min(self.theta23) - 0.1, max(self.theta23) + 0.1)



            plt.pause(0.01)
            plt.draw_all()
            ax1.clear()  
            
        #return self.circle1, self.circle2


    def graph_output(self):
        '''Saves various figures'''
        '''
        fig1 = plt.figure(figsize=(8, 6), dpi=600) 

        plt.plot(self.x1, self.y1, color='blue', alpha=0.7, label=r'$m_{1}$')
        plt.plot(self.x2, self.y2, color='orange', alpha=0.7, label=r'$m_{2}$')
        plt.legend(loc='upper right')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.xlim(-L1 - L2 - 0.1, L1 + L2 + 0.1)
        plt.ylim(-L1 - L2 - 0.1, L1 + L2 + 0.1)

        plt.savefig('./img/double_pendulum_path.png')

 
        fig2 = plt.figure(figsize=(8, 6), dpi=600)

        plt.plot(t, self.theta1, color='blue', alpha=0.7, label=r'$\theta_{1}$')
        plt.plot(t, self.theta2, color='orange', alpha=0.7, label=r'$\theta_{2}$')
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\theta$')
        plt.xlim(0, max(t))
        plt.ylim(min(self.theta2) - 0.3, max(self.theta2) + 0.3)

        plt.savefig('./img/double_pendulum_angle.png')
        '''
        '''
        fig3 = plt.figure(figsize=(8, 6), dpi=600)

        plt.plot(self.theta11, self.theta21, color='blue')
        plt.xlabel(r'$\theta_{1}$')
        plt.ylabel(r'$\theta_{2}$')

        plt.savefig('./img/theta_theta_1.png')

        fig4 = plt.figure(figsize=(8, 6), dpi=600)

        plt.plot(self.theta12, self.theta22, color='red')
        plt.xlabel(r'$\theta_{3}$')
        plt.ylabel(r'$\theta_{4}$')

        plt.savefig('./img/theta_theta_2.png')

        fig5 = plt.figure(figsize=(8, 6), dpi=600)

        plt.plot(self.theta13, self.theta23, color='green')
        plt.xlabel(r'$\theta_{5}$')
        plt.ylabel(r'$\theta_{6}$')

        plt.savefig('./img/theta_theta_3.png')
        '''

        fig5 = plt.figure(figsize=(8, 6), dpi=600)

        plt.plot(t, self.theta11, color='maroon', label=r'$\theta_{1}$')
        plt.plot(t, self.theta12, color='navy', label=r'$\theta_{2}$')
        plt.plot(t, self.theta13, color='indigo', label=r'$\theta_{3}$')
        plt.plot(t, self.theta21, color='red', label=r'$\theta_{4}$')
        plt.plot(t, self.theta22, color='deepskyblue', label=r'$\theta_{5}$')
        plt.plot(t, self.theta23, color='orchid', label=r'$\theta_{6}$')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\theta$')
        plt.legend(loc='upper right')

        plt.savefig('./img/compare_double_pen_1.png')
    
    def init(self): 
        '''Set up animation'''
        gs = gridspec.GridSpec(2, 2)

        self.animation_fig = plt.figure(figsize=(10, 10), dpi=600)

        self.ax1 = plt.subplot(gs[0, 0]) # row 0, col 0
        plt.plot([0,1])

        ax2 = plt.subplot(gs[0, 1]) # row 0, col 1
        plt.plot([0,1])

        ax3 = plt.subplot(gs[1, :]) # row 1, span all columns
        plt.plot([0,1])

        ax2.clear()
        self.line1, = ax2.plot(self.x1, self.y1, '-', color='blue', alpha=0.7, label=r'$m_{1}$')
        self.line2, = ax2.plot(self.x2, self.y2, '-', color='orange', alpha=0.7, label=r'$m_{2}$')
        ax2.legend(loc='upper right')
        ax3.clear()
        self.line3, = ax3.plot(t, self.theta1, '-', color='blue', alpha=0.7, label=r'$\theta_{1}$')
        self.line4, = ax3.plot(t, self.theta2, '-', color='orange', alpha=0.7, label=r'$\theta_{2}$')
        ax3.legend(loc='upper right')


        self.ax1.set_xlabel(r'$x$')
        self.ax1.set_ylabel(r'$y$')
        self.ax1.set_aspect('equal', adjustable='box')
          
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_xlim(-2.1, 2.1)
        ax2.set_ylim(-2.1, 2.1)
        
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$\theta$')
        ax3.set_xlim(0, max(t))
        ax3.set_ylim(min(self.theta2) - 0.1, max(self.theta2) + 0.1)
        

    def animate_plot(self, i):
        '''Generates animation'''

        self.ax1.clear() 
        self.ax1.set_xlim(-2.1, 2.1)
        self.ax1.set_ylim(-2.1, 2.1)
        self.ax1.plot([0, self.x1[i], self.x2[i]], [0, self.y1[i], self.y2[i]])
        self.circle0 = Circle((0, 0), r/2, fc='b', zorder=10)
        self.circle1 = Circle((self.x1[i], self.y1[i]), r, fc='b', zorder=10)
        self.circle2 = Circle((self.x2[i], self.y2[i]), r, fc='orange', zorder=10)
        self.ax1.add_patch(self.circle0)
        self.ax1.add_patch(self.circle1)
        self.ax1.add_patch(self.circle2)

        self.line1.set_xdata(self.x1[:i])
        self.line1.set_ydata(self.y1[:i])
        self.line2.set_xdata(self.x2[:i])
        self.line2.set_ydata(self.y2[:i])

        self.line3.set_xdata(t[:i])
        self.line4.set_xdata(t[:i])
        self.line3.set_ydata(self.theta1[:i])
        self.line4.set_ydata(self.theta2[:i])


    def create_animation(self):
        '''Save animation'''

        animate = animation.FuncAnimation(self.animation_fig, self.animate_plot,
            frames=6000, interval=1, repeat=False)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Timothy Holmes'), bitrate=1800)

        animate.save('./vid/double_pendulum.mp4', writer=writer)


#----------- Run -----------
start = time.time()

double_pendulum = double_pendulum(L1, L2, m1, m2, g)
double_pendulum.differential_equation()
double_pendulum.graph_output()

#Output files
#double_pendulum.graph_output()
#double_pendulum.init()
#double_pendulum.create_animation()

print('It took', time.time()-start, 'seconds.')