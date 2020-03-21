import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly


'''
# Parameter
#############

D = 7 * 10e-5                # diffusion coefficient
dt = 1                       # time step
N = 100                      # number of particles that should be simulated
x_len = 2                    # length of the domain
y_len = 1                    # width of the domain
k = 500                      # number of simulated time-steps


# Seeding of particles
######################

pos_x = np.random.uniform(size=N)*x_len/2
pos_y = np.random.uniform(size=N)*y_len


# Loop for k displacement iterations
####################################

anm_figure = plt.figure()

plt_anm = []

for i in range(k):

    # for each time step a sample of displacements (according to Einstein's theory) is assigned
    # in x- and in y-direction
    pos_x = pos_x + np.random.normal(0, np.sqrt(2*D*dt), N)
    pos_y = pos_y + np.random.normal(0, np.sqrt(2*D*dt), N)

    # conditions for closed boundaries
    for j in range(len(pos_x)):

        if pos_x[j]<0:
            pos_x[j] = 0 - pos_x[j]
        if pos_x[j]>x_len:
            pos_x[j] = x_len - (pos_x[j]-x_len)


        if pos_y[j]<0:
            pos_y[j] = 0 - pos_y[j]
        if pos_y[j]>y_len:
            pos_y[j] = y_len -  (pos_y[j]-y_len)

    # collect plots for animation
    plt_anm.append(plt.plot(pos_x, pos_y, 'o', color='b', animated=True))
    plt.axis([0, x_len, 0, y_len])
    plt.title('Brownian motion', fontweight='bold')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.text(x_len-0.45,y_len-0.2, "D = "+str(D)+" m^2/s \n"+"t = "+str(round(dt*k/60,2))+" min", bbox=dict(facecolor='white', alpha=0.1), fontsize=12)


BM_anm = animation.ArtistAnimation(anm_figure, plt_anm, interval=100, repeat=False, blit=True)
'''

# Parameter
#############

D = 7 * 10e-5                # diffusion coefficient
dt = 1                       # time step
N = 100                      # number of particles that should be simulated
x_len = 2                    # length of the domain
y_len = 1                    # width of the domain
k = 500                      # number of simulated time-steps
x_bar = 1                    # x-position of the barrier
bar_length = 0.7             # defining length of barrier
# setting start and end point of barrier
y_down_bar = y_len/2 - bar_length/2
y_up_bar = y_len/2 + bar_length/2



# Seeding of particles
######################

pos_x = np.random.uniform(size=N)*x_len/2
pos_y = np.random.uniform(size=N)*y_len


# Loop for k displacement iterations
####################################

anm_figure = plt.figure()

plt_anm = []

for i in range(k):

    # save old positions for later check concerning the barrier
    pos_x_old = pos_x
    pos_y_old = pos_y

    # for each time step a sample of displacements (according to Einstein's theory) is assigned
    # in x- and in y-direction
    pos_x = pos_x + np.random.normal(0, np.sqrt(2*D*dt), N)
    pos_y = pos_y + np.random.normal(0, np.sqrt(2*D*dt), N)

    # conditions for closed boundaries
    for j in range(len(pos_x)):

        if pos_x[j]<0:
            pos_x[j] = 0 - pos_x[j]
        if pos_x[j]>x_len:
            pos_x[j] = x_len - (pos_x[j]-x_len)


        if pos_y[j]<0:
            pos_y[j] = 0 - pos_y[j]
        if pos_y[j]>y_len:
            pos_y[j] = y_len -  (pos_y[j]-y_len)


        if pos_x_old[j]<x_bar and pos_x[j]>x_bar:

            if pos_y_old[j]<y_up_bar and pos_y_old[j]>y_down_bar:

                pos_x[j] = x_bar - (pos_x[j]-x_bar)


        if pos_x_old[j]>x_bar and pos_x[j]<x_bar:

            if pos_y_old[j]<y_up_bar and pos_y_old[j]>y_down_bar:

                pos_x[j] = x_bar + (x_bar-pos_x[j])


    # collect plots for animation
    plt_anm.append(plt.plot(pos_x, pos_y, 'bo', [x_bar, x_bar], [y_up_bar, y_down_bar],'r', animated=True))
    plt.axis([0, x_len, 0, y_len])
    plt.title('Brownian motion with barrier', fontweight='bold')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.text(x_len-0.45,y_len-0.2, "D = "+str(D)+" m^2/s \n"+"t = "+str(round(dt*k/60,2))+" min", bbox=dict(facecolor='white', alpha=0.1), fontsize=12)
    plt.text(x_len-0.45,y_len-0.4, "length of barrier: \n "+str(bar_length)+" m", bbox=dict(facecolor='white', alpha=0.1), fontsize=12)


BM_anm = animation.ArtistAnimation(anm_figure, plt_anm, interval=100, repeat=False, blit=True)

'''
PLOTTEN

# Plot histrograms for different times and Diffusioncoefficients
################################################################

plt.figure(figsize=[10,10])

# t = 50 s

plt.subplot(331)
plt.hist(pos_data[0][49][0], bins=30)
plt.title('D = '+str(D[0])+' m^2/s')
plt.ylabel('Occurances', fontsize=12)
plt.text(1.25,10,'t = 50s')
plt.axis([0, x_len, 0, 12])
plt.subplot(332)
plt.hist(pos_data[1][49][0], bins=30)
plt.title('D = '+str(D[1])+' m^2/s')
plt.text(1.25,10,'t = 50s')
plt.axis([0, x_len, 0, 12])
plt.subplot(333)
plt.hist(pos_data[2][49][0], bins=30)
plt.title('D = '+str(round(D[2], 5))+' m^2/s')
plt.text(1.25,10,'t = 50s')
plt.axis([0, x_len, 0, 12])

# t = 250 s

plt.subplot(334)
plt.hist(pos_data[0][249][0], bins=30)
plt.ylabel('Occurances', fontsize=12)
plt.text(1.25,10,'t = 250s')
plt.axis([0, x_len, 0, 12])
plt.subplot(335)
plt.hist(pos_data[1][249][0], bins=30)
plt.text(1.25,10,'t = 250s')
plt.axis([0, x_len, 0, 12])
plt.subplot(336)
plt.hist(pos_data[2][249][0], bins=30)
plt.text(1.25,10,'t = 250s')
plt.axis([0, x_len, 0, 12])

# t = 500 s

plt.subplot(337)
plt.hist(pos_data[0][499][0], bins=30)
plt.xlabel('x-position', fontsize=12)
plt.ylabel('Occurances', fontsize=12)
plt.text(1.25,10,'t = 500s')
plt.axis([0, x_len, 0, 12])
plt.subplot(338)
plt.hist(pos_data[1][499][0], bins=30)
plt.xlabel('x-position', fontsize=12)
plt.text(1.25,10,'t = 500s')
plt.axis([0, x_len, 0, 12])
plt.subplot(339)
plt.hist(pos_data[2][499][0], bins=30)
plt.xlabel('x-position', fontsize=12)
plt.text(1.25,10,'t = 500s')
plt.axis([0, x_len, 0, 12])


plt.show()
'''


'''

PLOTTEN

# Plot the histograms for different barrier lengths at different times
######################################################################

plt.figure(figsize=[10,10])

# t = 100 s

plt.subplot(331)
plt.hist(pos_data[0][99][0], bins=30)
plt.title('barrier length: '+str(bar_length[0])+' m')
plt.ylabel('Occurances', fontsize=12)
plt.text(1.25,10,'t = 100s')
plt.axis([0, x_len, 0, 12])
plt.subplot(332)
plt.hist(pos_data[1][99][0], bins=30)
plt.title('barrier length: '+str(bar_length[1])+' m')
plt.text(1.25,10,'t = 100s')
plt.axis([0, x_len, 0, 12])
plt.subplot(333)
plt.hist(pos_data[2][99][0], bins=30)
plt.title('barrier length: '+str(bar_length[2])+' m')
plt.text(1.25,10,'t = 100s')
plt.axis([0, x_len, 0, 12])

# t = 250 s

plt.subplot(334)
plt.hist(pos_data[0][249][0], bins=30)
plt.ylabel('Occurances', fontsize=12)
plt.text(1.25,10,'t = 250s')
plt.axis([0, x_len, 0, 12])
plt.subplot(335)
plt.hist(pos_data[1][249][0], bins=30)
plt.text(1.25,10,'t = 250s')
plt.axis([0, x_len, 0, 12])
plt.subplot(336)
plt.hist(pos_data[2][249][0], bins=30)
plt.text(1.25,10,'t = 250s')
plt.axis([0, x_len, 0, 12])

# t = 500 s

plt.subplot(337)
plt.hist(pos_data[0][499][0], bins=30)
plt.xlabel('x-position', fontsize=12)
plt.ylabel('Occurances', fontsize=12)
plt.text(1.25,10,'t = 500s')
plt.axis([0, x_len, 0, 12])
plt.subplot(338)
plt.hist(pos_data[1][499][0], bins=30)
plt.xlabel('x-position', fontsize=12)
plt.text(1.25,10,'t = 500s')
plt.axis([0, x_len, 0, 12])
plt.subplot(339)
plt.hist(pos_data[2][499][0], bins=30)
plt.xlabel('x-position', fontsize=12)
plt.text(1.25,10,'t = 500s')
plt.axis([0, x_len, 0, 12])


plt.show()
'''
'''
# Count how many particles are on the other side of the barrier at a certain time
#################################################################################

part_cross = []

for n in range(len(bar_length)):

    part_cross_temp = []

    for i in [99, 249, 449]:

        part_count = 0

        for j in range(len(pos_data[n][i][0])):

            if pos_data[n][i][0][j]>1:

                part_count += 1

        part_cross_temp.append(part_count)

    part_cross.append(part_cross_temp)

# simple output in source code, formatting in report
####################################################

print(part_cross)
'''
