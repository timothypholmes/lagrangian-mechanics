from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation
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
time_max, dt = 40, 0.01
t = np.arange(0, time_max + dt, dt)

#------------ initial conditions --------
y0 = [0, 0]

#------------ Animation values ----------
fps = 10
time_max, dt = 40, 0.01
t = np.arange(0, time_max + dt, dt)
frame_step = int(1/fps/dt)

#------------ shape values --------------
r = 0.05


class harmonically_driven_pendulum:
    ''' '''

    def __init__(self, L, omega, A, m, g):
        self.L = L
        self.omega = omega
        self.A = A
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


    def derivative(self, y, t, L, omega, A, m, g):
        '''Returns the first derivatives of y'''
        theta, thetadot = y 

        dtheta_dt = thetadot
        dthetadot_dt = (A * omega**2 / L * np.cos(omega*t) * np.cos(theta) -
            g * np.sin(theta))

        return dtheta_dt, dthetadot_dt


    def differential_equation(self):
        '''Returns a numerical solution to the differential equation'''
        y = odeint(self.derivative, y0, t, args=(self.L, self.omega, self.A, self.m, self.g))

        self.theta, self.thetadot = y[:,0], y[:,1]
        self.x = L * np.sin(self.theta) 
        self.y = -L * np.cos(self.theta)

        return self.x, self.y, self.theta, self.thetadot


    def animate_plot(self):
        '''Animation of the plots and shows'''
        #----------- Set figure -----------
        fig = plt.figure(figsize=(12, 4.5), dpi=80)
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)


        for i in range(0, len(t), frame_step):

            self.x0 = A * np.cos(omega * t[i])
            ax1.plot([self.x0, self.x0 + self.x[i]], [0, self.y[i]])
            self.circle1 = Circle((self.x0, 0), r/2, fc='b', zorder=10)
            self.circle2 = Circle((self.x0 + self.x[i], self.y[i]), r, fc='orange', zorder=10)
            ax1.add_patch(self.circle1)
            ax1.add_patch(self.circle2)
            ax1.set_xlim(-np.max(L)-A, np.max(L)+A)
            ax1.set_ylim(-np.max(L)-A, np.max(L)+A)
            ax1.set_aspect('equal', adjustable='box')

            ax2.scatter(t[i], self.theta[i], lw=0.1, c='orange')
            ax2.set_xlabel(r'$t\;/\mathrm{s}$')
            ax2.set_ylabel(r'$\theta$')
            ax2.set_xlim(0, max(t))
            ax2.set_ylim(min(self.theta) - 0.1, max(self.theta) + 0.1)
            plt.pause(0.01)
            plt.draw_all()
            ax1.clear()  

        return self.x0, self.circle1, self.circle2

    
    def numerical_graph(self):
        '''Saves a numerical solution to the diff eq as a plot'''
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111)
        ax.plot(t, self.theta, lw=2, c='orange', alpha=0.7)
        
        ax.set_xlabel(r'$t\;/\mathrm{s}$')
        ax.set_ylabel(r'$\theta$')
        ax.set_xlim(0, time_max)
        ax.set_ylim(min(self.theta) - 0.1, max(self.theta) + 0.1)
        ax.legend()
        plt.savefig('driven-theta.png', dpi=1200)
        #plt.show()

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
        

    '''
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
    '''

    def create_animation(self):
        '''Save animation'''

        animate = animation.FuncAnimation(self.animation_fig, self.animate_plot,
            frames=2000, interval=1, repeat=False)

        #mywriter = animation.FFMpegWriter()
        #animate.save('~/Desktop/mymovie.mp4',writer=mywriter)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        animate.save('pendulum.mp4', writer=writer)
        


class other_pendulum:
    pass


#----------- Run -----------
hdp = harmonically_driven_pendulum(L, omega, A, m, g)
hdp.differential_equation()
hdp.animate_plot()
#hdp.numerical_graph()
#hdp.init()
#hdp.create_animation()


op = other_pendulum()