import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import math


def read_file(fname):
     particle_ids = set()
     # properties written at every time step
     t = 0
     prop_t = dict()
     prop_all = []  

     with open(fname) as f:
          for line in f:
               if line.startswith("Time:"):
                    if len(prop_t) > 1:
                         prop_all += [(t, prop_t)]
                    t = float(line.split(":")[1].strip())
                    prop_t = dict()
               if line.startswith('\t'):
                    prop_t[int(line.strip().split(" ")[0])] = [float(d) for d in line.strip().split(" ")[1:]] 
                    x, y = [float(d) for d in line.strip().split(" ")[1:3]]
     
     # print(prop_all)
     return prop_t, prop_all

def read_init(fname, prefix = "vpfc->"):
     params = dict()

     with open(fname) as f:
          for line in f:
               if ':' in line:
                    line = line[line.startswith(prefix) and len(prefix):]
                    params[line.split(':')[0].strip()] = line.split(':')[1].strip()
     return params

_, prop_all = read_file("output/particles.2d")
init_params = read_init("init/vpfc.2d")

lattice = float(init_params["lattice"])*math.pi/np.sqrt(3);
scale = float(init_params["Nl"])*lattice;
domain = [[-scale,scale], [-scale,scale]]

# ui
particle_cmap = plt.get_cmap('jet')(np.linspace(0.2, 0.8, 42))
arrow_size = 5

fig = plt.figure()
ax=fig.add_subplot(111)
ims=[]; gui_artists = dict()

(t, prop_t) = prop_all[0]
for _id, prop in prop_t.items():
     circle = plt.Circle((prop[0],prop[1]), 3, color=particle_cmap[_id-1])
     patch = ax.add_patch(circle)
     text = ax.text(prop[0],prop[1],str(_id), color = "black")
     line, = ax.plot([prop[0], prop[0]+np.sign(prop[2])*prop[2]**2*arrow_size], [prop[1], prop[1]+np.sign(prop[3])*prop[3]**2*arrow_size], color="black")

     gui_artists[_id] = (circle, patch, line, text)


def simData():
     tmax = len(prop_all)
     ts = 1
     tc = 0
     while tc < tmax:
          (t, prop_t) = prop_all[int(tc)]
          for _id, prop in prop_t.items():
               (circle, patch, line, text) = gui_artists[_id]
               circle.center = prop[0],prop[1]
               text.set_position((prop[0],prop[1]))
               line.set_data([prop[0], prop[0]+np.sign(prop[2])*prop[2]**2*arrow_size], [prop[1], prop[1]+np.sign(prop[3])*prop[3]**2*arrow_size])
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
plt.gca().set_aspect('equal', adjustable='box')
# ani = anim.ArtistAnimation(fig,ims, interval=50,blit=False)
ani = anim.FuncAnimation(fig, simPoints, simData, blit=False,\
     interval=10, repeat=True)
plt.show()
