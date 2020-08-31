from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import numpy as np
import scipy
import time

#------------ variables ----------------
L = 1        
omega = 2.5  
A = 0.1      
m = 1        
g = 9.81    

#------------ time step -----------------
time_max, dt = 40, 0.001
t = np.arange(0, time_max + dt, dt)

#------------ initial conditions --------
initial_q = [np.pi/2, 0]

#------------ Animation values ----------
fps = 10
time_max, dt = 50, 0.01
t = np.arange(0, time_max + dt, dt)
frame_step = int(1/fps/dt)

#------------ shape values --------------
r = 0.05


class simple_pendulum:
    ''' '''

    def __init__(self, L, m, g):
        self.L = L
        self.m = m
        self.g = g

        '''
        Parameters
        ----------
        L : scalar, float
            Pendulum rod length (m)
        omega : scalar, float
            Drive frequency (Hz)
        A : scalar, float
            Amplitude (m)
        m : scalar, float
            Mass of the pendulum bob (Kg)
        g : scalar, float
            Gravitational constant (ms^-2)
        '''


    def derivative(self, q, t, L, m, g):
        '''Returns the first derivatives of y'''
        theta, thetadot = q

        dtheta_dt = thetadot
        dthetadot_dt = -(g/L) * np.sin(theta)

        return dtheta_dt, dthetadot_dt


    def coordinate_transformation(self):
        '''Returns a numerical solution to the differential equation'''
        q = odeint(self.derivative, initial_q, t, args=(self.L, self.m, self.g))

        self.theta, self.thetadot = q[:,0], q[:,1]
        self.x = L * np.sin(self.theta) 
        self.y = - L * np.cos(self.theta)

        return self.x, self.y, self.theta, self.thetadot


    def show_plot(self):
        '''Animation of the plots and shows'''
        #----------- Set figure -----------
        gs = gridspec.GridSpec(2, 2)

        fig = plt.figure(figsize=(10, 10), dpi=80)
        ax1 = plt.subplot(gs[0, 0]) 
        plt.plot([0,1])

        
        ax2 = plt.subplot(gs[0, 1]) 
        plt.plot([0,1])

        ax3 = plt.subplot(gs[1, :]) 
        plt.plot([0,1])

        ax2.clear()
        ax3.clear()
        line1, = ax2.plot(self.x, self.y, '-', color='orange')
        line2, = ax3.plot(t, self.theta, '-', color='blue')


        for i in range(0, len(t), frame_step):

            ax1.plot([0, self.x[i]], [0, self.y[i]])
            self.circle0 = Circle((0, 0), r/2, fc='b', zorder=10)
            self.circle1 = Circle((self.x[i], self.y[i]), r/2, fc='orange', zorder=10)
            ax1.add_patch(self.circle0)
            ax1.add_patch(self.circle1)
            ax1.set_xlabel(r'$x$')
            ax1.set_ylabel(r'$y$')
            ax1.set_xlim(-L - 0.1, L + 0.1)
            ax1.set_ylim(-L - 0.1, L + 0.1)
            ax1.set_aspect('equal', adjustable='box')


            line1.set_xdata(self.x[:i])
            line1.set_ydata(self.y[:i])
            ax2.set_xlabel(r'$x$')
            ax2.set_ylabel(r'$y$')
            ax2.set_xlim(-L - 0.1, L + 0.1)
            ax2.set_ylim(-L - 0.1, L + 0.1)


            line2.set_xdata(t[:i])
            line2.set_ydata(self.theta[:i])
            ax3.set_xlabel(r'$t$')
            ax3.set_ylabel(r'$\theta$')
            ax3.set_xlim(0, max(t))
            ax3.set_ylim(min(self.theta) - 0.1, max(self.theta) + 0.1)

            plt.pause(0.01)
            plt.draw_all()
            ax1.clear()  
            
        return self.circle1

    
    def graph_output(self):
        '''Saves various figures'''
        fig1 = plt.figure(figsize=(8, 6), dpi=600) 

        plt.plot(self.x, self.y, color='blue')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.xlim(-L - 0.1, L + 0.1)
        plt.ylim(-L - 0.1, L + 0.1)

        plt.savefig('./img/simple_pendulum_path.png')


        fig2 = plt.figure(figsize=(8, 6), dpi=600)

        plt.plot(t, self.theta, color='blue')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\theta$')
        plt.xlim(0, max(t))
        plt.ylim(min(self.theta) - 0.1, max(self.theta) + 0.1)

        plt.savefig('./img/simple_pendulum_angle.png')
        


    def init(self): 
        '''Set up animation'''
        self.animation_fig = plt.figure(figsize=(12, 4.5), dpi=100)
        self.ax1 = self.animation_fig.add_subplot(1,2,1)
        self.ax2 = self.animation_fig.add_subplot(1,2,2)

        self.ax1.set_xlim(-np.max(L)-A, np.max(L)+A)
        self.ax1.set_ylim(-np.max(L)-A, np.max(L)+A)
        self.ax1.set_aspect('equal', adjustable='box')
            
        self.ax2.set_xlabel(r'$t\;/\mathrm{s}$')
        self.ax2.set_ylabel(r'$\theta$')
        self.ax2.set_xlim(0, max(t))
        self.ax2.set_ylim(min(self.theta) - 0.1, max(self.theta) + 0.1)
        

    
    def animate_plot(self, i):
        

        self.ax1.clear()
        self.ax1.set_xlim(-np.max(L)-A, np.max(L)+A)
        self.ax1.set_ylim(-np.max(L)-A, np.max(L)+A)
        self.x0 = A * np.cos(omega * t[i])
        self.ax1.plot([self.x0, self.x0 + self.x[i]], [0, self.y[i]])
        self.circle1 = Circle((self.x0, 0), r/2, fc='b', zorder=10)
        self.circle2 = Circle((self.x0 + self.x[i], self.y[i]), r, fc='orange', zorder=10)
        self.ax1.add_patch(self.circle1)
        self.ax1.add_patch(self.circle2)

        self.ax2.scatter(t[i], self.theta[i], lw=0.1, c='orange')

        #self.ax1.clear()  
    

    def create_animation(self):
        '''Save animation'''

        animate = animation.FuncAnimation(self.animation_fig, self.animate_plot,
            frames=2000, interval=1, repeat=False)

        #mywriter = animation.FFMpegWriter()
        #animate.save('~/Desktop/mymovie.mp4',writer=mywriter)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        animate.save('pendulum.mp4', writer=writer)
        

#----------- Run -----------
simple_pendulum = simple_pendulum(L, m, g)
simple_pendulum.coordinate_transformation()
simple_pendulum.show_plot()

