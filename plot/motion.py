from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection

import os
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Run vpfc visualization scipt')
parser.add_argument("-i",
    help="Input config file", metavar="input", required=True)
args = vars(parser.parse_args())


def read_init(fname, prefix = "vpfc->"):
     params = dict()

     with open(fname) as f:
          for line in f:
               if ':' in line:
                    line = line[line.startswith(prefix) and len(prefix):]
                    params[line.split(':')[0].strip()] = line.split(':')[1].strip()
     return params

# init_params = read_init(args["i"])
init_params = read_init(os.path.join(args["i"],"init.2d"))

def read_file(fname):
     particle_ids = set()
     # properties written at every time step
     t = 0
     prop_t = dict()
     t_all = []; prop_all = []

     with open(fname) as f:
          for line in f:
               if line.startswith("Time:"):
                    if len(prop_t) > 1:
                         prop_all += [[prop_t[i] if i in prop_t.keys() else [0.0]*6 for i in range(int(init_params["N"]))]]
                         t_all += [t]
                    t = float(line.split(":")[1].strip())
                    prop_t = dict()
               if line.startswith('\t'):
                    if len(line.strip().split(" ")) > 5:
                         prop_t[int(line.strip().split(" ")[0])] = [float(d) for d in line.strip().split(" ")[1:]]
                    else:
                         prop_t[int(line.strip().split(" ")[0])] = [float(d) for d in line.strip().split(" ")[1:]] + [0,0]

          if len(prop_t) > 1:
               prop_all += [[prop_t[i] if i in prop_t.keys() else [0.0]*6 for i in range(int(init_params["N"]))]]
               t_all += [t]
     
     return np.array(t_all), np.array(prop_all)

t_all, prop_all = read_file(os.path.join(args["i"],"particles.2d"))

# prop_all[:,:,4] = np.diff(prop_all[:,:,0], append=0, axis=0)
# prop_all[:,:,5] = np.diff(prop_all[:,:,1], append=0, axis=0)

lattice = float(init_params["lattice"])*math.pi/np.sqrt(3);
scale = float(init_params["Nl"])*lattice;
domain = [[-scale,scale], [-scale,scale]]
orient_line_len = 5

orientation_vector = np.moveaxis(np.array([(prop_all[:,:,2]+1)/2, (prop_all[:,:,3]+1)/2]), 0, -1)
velocity_real = prop_all[:,:,4:]

orient_ext_vector = np.array([[prop_all[:,:,0], prop_all[:,:,0]+np.sign(prop_all[:,:,2])*prop_all[:,:,2]**2*orient_line_len],[prop_all[:,:,1], prop_all[:,:,1]+np.sign(prop_all[:,:,3])*prop_all[:,:,3]**2*orient_line_len]])
orient_ext_vector = np.moveaxis(orient_ext_vector, [0, 1], [-2, -1])
# due to the different way of coordinates usage by plot and LineCollection
orient_ext_vector = np.transpose(orient_ext_vector, (0, 1, 3, 2))

realVelocity_vectors = np.array([[prop_all[:,:,0], prop_all[:,:,0]+np.sign(prop_all[:,:,4])*prop_all[:,:,4]**2*100],[prop_all[:,:,1], prop_all[:,:,1]+np.sign(prop_all[:,:,5])*prop_all[:,:,5]**2*100]])
realVelocity_vectors = np.moveaxis(realVelocity_vectors, [0, 1], [-2, -1])
# due to the different way of coordinates usage by plot and LineCollection
realVelocity_vectors = np.transpose(realVelocity_vectors, (0, 1, 3, 2))

n0 = velocity_real[:,:,0]/np.sqrt(velocity_real[:,:,0]**2 + velocity_real[:,:,1]**2+1E-16)
n1 = velocity_real[:,:,1]/np.sqrt(velocity_real[:,:,0]**2 + velocity_real[:,:,1]**2+1E-16)
translation_order = np.sum(prop_all[:,:,4:], axis=1)/prop_all.shape[1]
translation_order2 = np.sqrt(np.mean(n0,axis=1)**2 + np.mean(n1,axis=1)**2)

# ui
particle_cmap = plt.get_cmap('jet')(np.linspace(0.2, 0.8, int(init_params["N"])))

# fig = plt.figure()

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[:, :-1])
ax.set_aspect('equal', adjustable='box')
# f3_ax1.set_title('gs[0, :]')
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_aspect('auto', adjustable='box')
ax2.set_title('translational order')
# translation_order_line, = ax2.plot([], color="black")
translation_order_line2, = ax2.plot([], color="black")
ax2.set_ylabel(r'$\Phi_T$',fontsize=16)
ax2.set_xlabel('t (s)',fontsize=16)
ax2.grid()

ax3 = fig.add_subplot(gs[1, 2])
ax3.set_title('rotational order')
ax3.set_aspect('auto', adjustable='box')
ax3.set_ylabel(r'$\Phi_R$',fontsize=16)
ax3.set_xlabel('t',fontsize=16)

gui_artists = dict()
label_particles_gui = False
colormap_orientational = True

t, prop_t = t_all[0], prop_all[0]

for particle in range(len(prop_t)):
     # particle properties at time t
     p_prop_t = prop_t[particle]
     if colormap_orientational:
          circle = plt.Circle((p_prop_t[0],p_prop_t[1]), 3, color=(0, orientation_vector[0][particle][0], orientation_vector[0][particle][1]), linewidth=2, fill=False)
     else:
          circle = plt.Circle((p_prop_t[0],p_prop_t[1]), 3, color=particle_cmap[particle], linewidth=2, fill=False)

     patch = ax.add_patch(circle)
     patch = ax.add_artist(circle)
     if label_particles_gui:
          text = ax.text(p_prop_t[0],p_prop_t[1],str(particle), color = "black")
          gui_artists[particle] = (circle, text)
     else:
          gui_artists[particle] = (circle)

line_color = np.array([(0, orientation_vector[0][p][0], orientation_vector[0][p][1]) if colormap_orientational else particle_cmap[p] for p in range(len(prop_t))])
velocity_segments = LineCollection(orient_ext_vector[0],
                          linestyles="solid", color=line_color)
realVelocity_segments = LineCollection(realVelocity_vectors[0],
                          linestyles="solid", color="black")
# ax.add_collection(velocity_segments)
ax.add_collection(realVelocity_segments)


def simData():
     tmax = len(prop_all)
     ts = 1
     tc = 0
     while tc < tmax:
          t, prop_t = t_all[tc], prop_all[tc]
          patches = []
          for particle in range(len(prop_t)):
               p_prop_t = prop_t[particle]

               if label_particles_gui:
                    (circle, text) = gui_artists[particle]
                    text.set_position((p_prop_t[0],p_prop_t[1]))
                    circle.center = p_prop_t[0],p_prop_t[1]
               else:
                    (circle) = gui_artists[particle]
                    circle.center = p_prop_t[0],p_prop_t[1]

               if colormap_orientational:
                    circle.set_edgecolor((0, orientation_vector[tc][particle][0], orientation_vector[tc][particle][1]))

          # translation_order_line.set_data(t_all[:tc+1], translation_order[:tc+1,0]**2+translation_order[:tc+1,1]**2)
          translation_order_line2.set_data(t_all[:tc+1], translation_order2[:tc+1])
          ax2.relim()
          ax2.autoscale_view(True,True,True)
          # velocity_segments.set_segments(orient_ext_vector[tc])
          realVelocity_segments.set_segments(realVelocity_vectors[tc])

          # if colormap_orientational:
          #      velocity_segments.set_color(np.array([(0, orientation_vector[tc][p][0], orientation_vector[tc][p][1]) for p in range(len(prop_t))]))

          tc = tc + ts
          yield [t]

def simPoints(simData):
     t = simData[0]
     time_text.set_text(time_template%(t))
     return time_text


# ax.set_title("t = " + str(t))
ax.set(xlim=domain[0], ylim=domain[1])

time_template = 'Time = %.3f s'    # prints running simulation time
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

#run animation
# plt.gca().set_aspect('equal', adjustable='box')
ani = anim.FuncAnimation(fig, simPoints, simData, interval=1, repeat=False)
# ax2.autoscale(True)
ani.event_source.stop()
ax2.autoscale(enable=True, axis="y", tight=False)
plt.show()
