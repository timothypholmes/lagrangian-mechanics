from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation
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
omega = 2

#------------ time step -----------------
time_max, dt = 40, 0.01
t = np.arange(0, time_max + dt, dt)

#------------ initial conditions --------
y0 = np.array([0, 0, 0, 0]) #theta1, thetadot1, theta2, thetadot2

#------------ Animation values ----------
fps = 10
time_max, dt = 40, 0.01
t = np.arange(0, time_max + dt, dt)
frame_step = int(1/fps/dt)

#------------ shape values --------------
r = 0.05
R = 0.5



class rotational_double_pendulum:
    ''' '''

    def __init__(self, L1, L2, omega, m1, m2, g, R):
        self.L1 = L1
        self.L2 = L2
        self.omega = omega
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.R = R


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


    def derivative(self, y, t, L1, L2, omega, m1, m2, g):
        '''Returns the first derivatives of y'''
        theta1, thetadot1, theta2, thetadot2 = y 


        dtheta_dt1 = thetadot1
        dtheta_dt2 = thetadot2
        dthetadot_dt1 = ((R/L) * (omega ** 2)) * np.sin((omega * t) + theta1) 
        - ((m2*g*np.sin(theta2)*np.cos(theta1-theta2) 
        - m2*np.sin(theta1-theta2)*(L1*thetadot1**2*np.cos(theta1-theta2) 
        + L2*thetadot2**2) - (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*np.sin(theta1-theta2)**2))


        dthetadot_dt2 = ((R/L) * (omega ** 2)) * np.sin((omega * t) + theta1) 
        - (((m1+m2)*(L1*thetadot1**2*np.sin(theta1-theta2) - g*np.sin(theta2) 
        + g*np.sin(theta1)*np.cos(theta1-theta2)) 
        + m2*L2*thetadot2**2*np.sin(theta1-theta2)*np.cos(theta1-theta2)) / L2 / 
        (m1 + m2*np.sin(theta1-theta2)**2))

        return dtheta_dt1, dthetadot_dt1, dtheta_dt2, dthetadot_dt2


    def differential_equation(self):
        '''Returns a numerical solution to the differential equation'''
        y = odeint(self.derivative, y0, t, args=(self.L1, self.L2, self.m1, 
        self.m2, self.omega, self.g))

        #self.theta, self.thetadot = y[:,0], y[:,1]
        self.theta1, self.thetadot1, self.theta2, self.thetadot2 = y[:,0], y[:,1], y[:,2], y[:,3]
        self.x1 = - (R * np.sin(omega * t))
        self.y1 = - (R * np.cos(omega * t))
        self.x2 = - L * np.sin(self.theta1) - R * np.sin(omega * t)
        self.y2 = - L * np.cos(self.theta1) - R * np.cos(omega * t)
        self.x3 = self.x2 - L * np.sin(self.theta2) - R * np.sin(omega * t)
        self.y3 = self.y2 - L * np.cos(self.theta2) - R * np.cos(omega * t)

        return self.x1, self.x2, self.y1, self.y2, self.theta1, self.thetadot1, self.theta2, self.thetadot2



    def generate_plot(self):
        '''Animation of the plots and shows'''
        #----------- Set figure -----------
        fig = plt.figure(figsize=(12, 4.5), dpi=80)
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)


        for i in range(0, len(t), frame_step):

            ax1.plot([self.x1[i], self.x2[i], self.x3[i]], [self.y1[i], self.y2[i], self.y3[i]])
            self.circle1 = Circle((self.x1[i], self.y1[i]), r/2, fc='b', zorder=10)
            self.circle2 = Circle((self.x2[i], self.y2[i]), r, fc='orange', zorder=10)
            self.circle3 = Circle((self.x3[i], self.y3[i]), r, fc='orange', zorder=10)
            ax1.add_patch(self.circle1)
            ax1.add_patch(self.circle2)
            ax1.add_patch(self.circle3)
            ax1.set_xlim(-3, 3)
            ax1.set_ylim(-3, 3)
            ax1.set_aspect('equal', adjustable='box')


            x1_ring = self.x1
            x2_ring = self.y1
            ax1.plot(x1_ring, x2_ring, color='black', linewidth=0.1)

            ax2.scatter(t[i], self.theta1[i], lw=0.01, c='orange')
            ax2.scatter(t[i], self.theta2[i], lw=0.01, c='orange')
            ax2.set_xlabel(r'$t$')
            ax2.set_ylabel(r'$\theta$')
            ax2.set_xlim(0, max(t))
            ax2.set_ylim(min(self.theta1) - 0.1, max(self.theta1) + 0.1)
            plt.pause(0.01)
            plt.draw_all()
            ax1.clear()  

        return self.circle1, self.circle2


    def init(self): 
        '''Set up animation'''
        self.animation_fig = plt.figure(figsize=(12, 4.5), dpi=600)
        self.ax1 = self.animation_fig.add_subplot(1,2,1)
        self.ax2 = self.animation_fig.add_subplot(1,2,2)

        self.ax1.set_xlim(-np.max(L)-A, np.max(L)+A)
        self.ax1.set_ylim(-np.max(L)-A, np.max(L)+A)
        self.ax1.set_aspect('equal', adjustable='box')
            
        self.ax2.set_xlabel(r'$t\;/\mathrm{s}$')
        self.ax2.set_ylabel(r'$\theta$')
        self.ax2.set_xlim(0, 20)
        self.ax2.set_ylim(min(self.theta) - 0.1, max(self.theta) + 0.1)
        


    def animate_plot(self, i):
        '''Generates animation'''

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
            frames=6000, interval=1, repeat=False)

        #mywriter = animation.FFMpegWriter()
        #animate.save('~/Desktop/mymovie.mp4',writer=mywriter)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=45, metadata=dict(artist='Timothy Holmes'), bitrate=1800)

        animate.save('rotational-pendulum.mp4', writer=writer)


start = time.time()

rotational_double_pendulum = rotational_double_pendulum(L1, L2, omega, m1, m2, g, R)
rotational_double_pendulum.differential_equation()
rotational_double_pendulum.generate_plot()
#rotational_double_pendulum.numerical_graph()
#rotational_double_pendulum.init()
#rotational_double_pendulum.create_animation()'


print('It took', time.time()-start, 'seconds.')