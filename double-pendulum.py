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

#------------ time step -----------------
time_max, dt = 40, 0.01
t = np.arange(0, time_max + dt, dt)

#------------ initial conditions --------
y0 = np.array([np.pi/2, 0, np.pi/2, 0]) #theta1, thetadot1, theta2, thetadot2

#------------ Animation values ----------
fps = 10
time_max, dt = 40, 0.01
t = np.arange(0, time_max + dt, dt)
frame_step = int(1/fps/dt)

#------------ shape values --------------
r = 0.05



class double_pendulum:
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
        theta1, thetadot1, theta2, thetadot2 = y 


        dtheta_dt1 = thetadot1
        dtheta_dt2 = thetadot2
        dthetadot_dt1 = ((m2*g*np.sin(theta2)*np.cos(theta1-theta2) 
        - m2*np.sin(theta1-theta2)*(L1*thetadot1**2*np.cos(theta1-theta2) 
        + L2*thetadot2**2) - (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*np.sin(theta1-theta2)**2))


        dthetadot_dt2 = (((m1+m2)*(L1*thetadot1**2*np.sin(theta1-theta2) - g*np.sin(theta2) 
        + g*np.sin(theta1)*np.cos(theta1-theta2)) 
        + m2*L2*thetadot2**2*np.sin(theta1-theta2)*np.cos(theta1-theta2)) / L2 / 
        (m1 + m2*np.sin(theta1-theta2)**2))

        return dtheta_dt1, dthetadot_dt1, dtheta_dt2, dthetadot_dt2


    def differential_equation(self):
        '''Returns a numerical solution to the differential equation'''
        y = odeint(self.derivative, y0, t, args=(self.L1, self.L2, self.m1, 
        self.m2, self.g))

        self.theta1, self.theta2 = y[:,0], y[:,2]
        self.x1 = L1 * np.sin(self.theta1)
        self.y1 = -L1 * np.cos(self.theta1)
        self.x2 = self.x1 + L2 * np.sin(self.theta2)
        self.y2 = self.y1 - L2 * np.cos(self.theta2)

        return self.x1, self.y1, self.theta1, self.x2, self.y2, self.theta2


    def generate_plot(self):
        '''Animation of the plots and shows'''
        #----------- Set figure -----------
        fig = plt.figure(figsize=(12, 4.5), dpi=80)
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)


        for i in range(0, len(t), frame_step):

            ax1.plot([0, self.x1[i], self.x2[i]], [0, self.y1[i], self.y2[i]])
            self.circle0 = Circle((0, 0), r/2, fc='b', zorder=10)
            self.circle1 = Circle((self.x1[i], self.y1[i]), r/2, fc='b', zorder=10)
            self.circle2 = Circle((self.x2[i], self.y2[i]), r, fc='orange', zorder=10)
            ax1.add_patch(self.circle0)
            ax1.add_patch(self.circle1)
            ax1.add_patch(self.circle2)
            ax1.set_xlim(-2.1, 2.1)
            ax1.set_ylim(-2.1, 2.1)
            ax1.set_aspect('equal', adjustable='box')

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
        self.ax1.plot([0, self.x0 + self.x[i]], [0, self.y[i]])
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

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=45, metadata=dict(artist='Timothy Holmes'), bitrate=1800)

        animate.save('rotational-pendulum.mp4', writer=writer)


#----------- Run -----------
start = time.time()

double_pendulum = double_pendulum(L1, L2, m1, m2, g)
double_pendulum.differential_equation()
double_pendulum.generate_plot()
#double_pendulum.numerical_graph()
#double_pendulum.init()
#double_pendulum.create_animation()'

print('It took', time.time()-start, 'seconds.')