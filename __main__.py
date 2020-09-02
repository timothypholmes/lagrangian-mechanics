from pendulum.pendulum_problems import (simple_pendulum, rotational_pendulum,
                                        spring_pendulum, harmonically_driven_pendulum,
                                        double_pendulum, multiple_double_pendulum,
                                        visualize)
import numpy as np


#------------ variables ----------------
L = 1        
omega = 2.5  
A = 0.1      
m = 1        
g = 9.81 

#------------ Animation values ----------
fps = 10
time_max, dt = 40, 0.01
t = np.arange(0, time_max + dt, dt)
frame_step = int(1/fps/dt)

#------------ shape values --------------
r = 0.05
R = 0.5

visualize = visualize()

if __name__ == "__main__":
    sp = simple_pendulum(L, m, g)
    sp.coordinate_transformation()
    sp.show_plot()

    rp = rotational_pendulum(L, omega, A, m, g, R)

    sprp = spring_pendulum(L, m, g)

    hp = harmonically_driven_pendulum(L, omega, A, m, g)

    dp = double_pendulum(L1, L2, m1, m2, g)

    mdp = multiple_double_pendulum(L1, L2, m1, m2, g)

    visualize.
