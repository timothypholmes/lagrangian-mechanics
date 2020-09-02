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
A = 0.1      
m = 1        
g = 9.81    

#------------ time step -----------------
time_max, dt = 50, 0.01
t = np.arange(0, time_max + dt, dt)
#omega = np.arange(0, 10,len(t))
omega = 3.13
#------------ initial conditions --------
y0 = [0, 0]

#------------ Animation values ----------
fps = 10
time_max, dt = 50, 0.01
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
            g/L * np.sin(theta))

        return dtheta_dt, dthetadot_dt


    def differential_equation(self):
        '''Returns a numerical solution to the differential equation'''
        y = odeint(self.derivative, y0, t, args=(self.L, self.omega, self.A, self.m, self.g))

        self.theta, self.thetadot = y[:,0], y[:,1]
        self.x = L * np.sin(self.theta) 
        self.y = -L * np.cos(self.theta)

        return self.x, self.y, self.theta, self.thetadot


    def show_plot(self):
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
        line1, = ax2.plot(self.x, self.y, '-', color='orange')
        line2, = ax3.plot(t, self.theta, '-', color='orange')
        

        for i in range(0, len(t), frame_step):

            self.x0 = A * np.cos(omega * t[i])
            ax1.plot([self.x0, self.x0 + self.x[i]], [0, self.y[i]])
            self.circle0 = Circle((self.x0, 0), r/2, fc='b', zorder=10)
            self.circle1 = Circle((self.x0 + self.x[i], self.y[i]), r/2, fc='orange', zorder=10)
            ax1.add_patch(self.circle0)
            ax1.add_patch(self.circle1)
            ax1.set_xlabel(r'$x$')
            ax1.set_ylabel(r'$y$')
            ax1.set_xlim(-L - 0.1, L + 0.1)
            ax1.set_ylim(-L - 0.1, L + 0.1)
            ax1.set_aspect('equal', adjustable='box')

            line1.set_xdata(A * np.cos(omega * t[:i]) + self.x[:i])
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
            ax3.set_ylim(min(self.theta) - 0.3, max(self.theta) + 0.3)

            plt.pause(0.01)
            plt.draw_all()
            ax1.clear()  
        

        return self.x0, self.circle0, self.circle1

    def show_plot_with_omega(self):
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

        ax3 = plt.subplot(gs[1, 0]) # row 1, span all columns
        plt.plot([0,1])

        ax4 = plt.subplot(gs[1, 1]) # row 1, span all columns
        plt.plot([0,1])


        ax2.clear()
        ax3.clear()
        ax4.clear()
        line1, = ax2.plot(self.x, self.y, '-', color='orange')
        line2, = ax3.plot(t, self.theta, '-', color='orange')
        line3, = ax4.plot(A*omega**2, omega)

        for i in range(0, len(t), frame_step):

            self.x0 = A * np.cos(omega * t[i])
            ax1.plot([self.x0, self.x0 + self.x[i]], [0, self.y[i]])
            self.circle0 = Circle((self.x0, 0), r/2, fc='b', zorder=10)
            self.circle1 = Circle((self.x0 + self.x[i], self.y[i]), r/2, fc='orange', zorder=10)
            ax1.add_patch(self.circle0)
            ax1.add_patch(self.circle1)
            ax1.set_xlabel(r'$x$')
            ax1.set_ylabel(r'$y$')
            ax1.set_xlim(-L - 0.1, L + 0.1)
            ax1.set_ylim(-L - 0.1, L + 0.1)
            ax1.set_aspect('equal', adjustable='box')

            line1.set_xdata(A * np.cos(omega[:i] * t[:i]) + self.x[:i])
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
            ax3.set_ylim(min(self.theta) - 0.3, max(self.theta) + 0.3)

            line3.set_xdata(omega[:i])
            line3.set_ydata(A*omega[:i]**2)
            ax4.set_xlabel(r'$Omega$')
            ax4.set_ylabel(r'$Amplitude$')
           
            #ax4.set_xlim(0, max(omega))
            #ax4.set_ylim(min(self.theta) - 0.3, max(self.theta) + 0.3)

            plt.pause(0.01)
            plt.draw_all()
            ax1.clear()  
        

        return self.x0, self.circle0, self.circle1

    def omega_output(self):

        time_max, dt = 50, 0.01
        t = np.arange(0, time_max + dt, dt)
        omega = np.arange(0, 50 + dt, dt)
        omega0 = np.sqrt(g/L)

        Amp = ((omega**2 / L * np.cos(omega*t) * np.cos(self.theta) -
            g/L * np.sin(self.theta)))

        fig = plt.figure(figsize=(12, 4.5), dpi=80)
        plt.plot(omega, Amp)


        plt.show()


    def graph_output(self):
        '''Saves various figures'''
        fig1 = plt.figure(figsize=(8, 6), dpi=600) 

        plt.plot(self.x, self.y, color='orange', alpha=0.7, label=r'$m_{2}$')
        plt.legend(loc='upper right')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.xlim(-L - 0.1, L + 0.1)
        plt.ylim(-L - 0.1, L + 0.1)

        plt.savefig('./img/harmonic_driven_pendulum_path.png')

 
        fig2 = plt.figure(figsize=(8, 6), dpi=600)

        plt.plot(t, self.theta, color='orange', alpha=0.7, label=r'$\theta$')
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\theta$')
        plt.xlim(0, max(t))
        plt.ylim(min(self.theta) - 0.3, max(self.theta) + 0.3)

        plt.savefig('./img/harmonic_driven_pendulum_angle.png')
        


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
harmonically_driven_pendulum = harmonically_driven_pendulum(L, omega, A, m, g)
harmonically_driven_pendulum.differential_equation()
harmonically_driven_pendulum.show_plot()

#harmonically_driven_pendulum.animate_plot(i)
#harmonically_driven_pendulum.graph_output()
#harmonically_driven_pendulum.init()
#harmonically_driven_pendulum.create_animation()
