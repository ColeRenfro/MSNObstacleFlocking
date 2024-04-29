import numpy as np
import matplotlib.pyplot as plt


#==============Parameters start===========
X = 70
Y = 70
EPSILON = 0.5
H = 0.2
C1ALPHA = 20
C2ALPHA = 2 * np.sqrt(C1ALPHA)
N = 150  # Number of sensor nodes
M = 2  # Space dimensions
D = 15  # Desired distance among sensor node
K = 1.2  # Scaling factor
R = K*D  # Interaction range
DELTAT = 0.009
A = 5
B = 5
C = np.abs(A-B)/np.sqrt(4*A*B)

snapPoints = np.arange(0,7,DELTAT)
iterations = 900 #total time of simulation
interval = 50 #interval of snap time 
# if interval is 10 and iterations is 100, then it runs for 0.1 secs and takes 10 snapshots

posx = np.zeros([N, iterations]) # x pos of nodes
posy = np.zeros([N, iterations]) # y pos of nodes

nodes = np.random.rand(N, M) * X
nodesvel = np.zeros([N, M])
velmag = np.zeros([N, iterations])
connectivity = np.zeros([iterations, 1])

fig = plt.figure()
gammaAgent = np.array([40, 25])
gammaAgentPath = np.zeros([iterations, 2])  # Path of gamma agent
gammaAgentPath[0] = gammaAgent
gammaAgentEnd = np.array([250,25])

# Define sine-wave trajectory parameters
amplitude = 50
frequency = 0.1  

# Calculate the x-coordinate for each frame
num_frames = 100  
x_coordinates = np.linspace(gammaAgent[0], 250, iterations)

# Calculate the y-coordinate using the sine wave
y_coordinates = gammaAgent[1] + amplitude * np.sin(frequency * x_coordinates)


c1mt = 1.1
c2mt = 2 * np.sqrt(c1mt)

c1beta = 1500
c2beta = 2*np.sqrt(c1beta)
rPrime = 0.22 * K * R
dPrime = 15
 
obstaclesLoc = np.array([(100, 20), (110,60)]) # 200, 25
obstacleRadius = np.array([15, 10])# add 15
numObstacles = obstaclesLoc.shape[0]


centerOfMass = np.zeros([iterations, M])


#==================Parameters end===================

def simulate():
    
    for t in range(0, iterations):
        # print(np.linalg.matrix_rank(adjacencyMatrix))
        AdjMat = MakeAdjMat()
        # print(np.linalg.matrix_rank(adjacencyMatrix))
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(AdjMat)
        centerOfMass[t] = np.array([np.mean(nodes[:, 0]), np.mean(nodes[:, 1])])
        
         
        gammaAgent[0] = x_coordinates[t]
        gammaAgent[1] = y_coordinates[t]
        # print(t)
        if t == 0:
            plotNeighbors(t,gammaAgent)
            for i in range(0, N):
                posx[i, t] = nodes[i, 0]
                posy[i, t] = nodes[i, 1]
        else:
            if t == 3:
                o = 9
                pass
            for i in range(0, N):
                # ui = getUi(i)
                oldVelocity = nodesvel[i, :]
                oldPosition = np.array([posx[i, t-1],
                                         posy[i, t-1]])
                # newVelocity = oldVelocity + ui * DELTAT
               
                
                for BetaLoc in obstaclesLoc:
                    mu = obstacleRadius / np.linalg.norm(oldPosition - BetaLoc)
                    ak = (oldPosition - BetaLoc) / np.linalg.norm(oldPosition - BetaLoc)
                    P = np.eye(2) - np.outer(ak, ak)
                    #print(f"P: {P.shape} OldVel: {oldVelocity.shape} MU: {mu.shape} ")
                    pik = mu * P @ oldVelocity
                    
                    qik = mu * oldPosition + (1 - mu) * BetaLoc
                    bik = bump(sigmaNorm(np.linalg.norm(qik - oldPosition)) / dBeta)
                    nik = (qik - oldPosition) / np.sqrt(1 + EPSILON * np.linalg.norm(qik - oldPosition) ** 2)

                    ui = getUi(i, mu, ak, P, pik, qik, bik, nik, oldPosition,gammaAgent)
                    newPosition = oldPosition + DELTAT * oldVelocity + (DELTAT ** 2 / 2) * ui
                    newVelocity = (newPosition - oldPosition) / DELTAT

                    # Update node properties
                    [posx[i, t], posy[i, t]] = newPosition
                    nodesvel[i, :] = newVelocity
                    nodes[i, :] = newPosition
                    velmag[i, t] = np.linalg.norm(newVelocity)

        if (t+1) % interval == 0:
            plotNeighbors(t,gammaAgent)

def sigmaNorm(z):
    val = EPSILON*(z**2)
    val = np.sqrt(1 + val) - 1
    val = val/EPSILON
    return val

rBeta = sigmaNorm(np.linalg.norm(rPrime))
dBeta = sigmaNorm(np.linalg.norm(dPrime))
s = 1

def MakeAdjMat():
    adjacencyMatrix = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                distance = np.linalg.norm(nodes[i] - nodes[j])
                if distance <= R:
                    adjacencyMatrix[i, j] = 1
    return adjacencyMatrix


def plotNeighbors(t, gammaAgent):
    plt.figure(figsize = (10,5))
    #plt.plot(gammaAgentPath[:t, 0], gammaAgentPath[:t, 1], 'ro', color = 'orange')
   
    # plt.plot(centerOfMass[0:t, 0], centerOfMass[0:t, 1], color='black')
    for i in range(0, numObstacles):
        plt.gcf().gca().add_artist(plt.Circle((obstaclesLoc[i, 0], obstaclesLoc[i, 1]), obstacleRadius[i], color='green'))
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    for i in range(0, N):
        for j in range(0, N):
            distance = np.linalg.norm(nodes[j] - nodes[i])
            if distance <= R:
                plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-', lw=0.5)

    plt.plot(gammaAgent[0], gammaAgent[1], 'ro', color='orange')
    plt.plot(gammaAgentEnd[0],gammaAgentEnd[1], 'ro', color = 'purple')
    plt.show()

def bump(x):
    if 0 <= x < H:
        return 1
    elif H <= x < 1:
        val = (x-H)/(1-H)
        val = np.cos(np.pi*val)
        val = (1+val)/2
        return val
    else:
        return 0

def sigma1(z):
    val = 1 + z **2
    val = np.sqrt(val)
    val = z/val
    return val

def phi(z):
    val1 = A + B
    val2 = sigma1(z + C)
    val3 = A - B
    val = val1 * val2 + val3
    val = val / 2
    return val

def phiAlpha(z):
    input1 = z/sigmaNorm(R) 
    input2 = z - sigmaNorm(D)  #
    val1 = bump(input1)
    val2 = phi(input2)
    val = val1 * val2
    return val

def phiBeta(z):
    val1 = bump(z/dBeta)
    val2 = sigma1(z-dBeta) - 1
    return val1 * val2

def getAij(i, j):
    val1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val1)
    val2 = sigmaNorm(norm)/sigmaNorm(R)
    val = bump(val2)
    return val

def getNij(i, j):
    val1 = nodes[j] - nodes[i]
    norm = np.linalg.norm(val1)
    val2 = 1 + EPSILON * norm**2
    val = val1/np.sqrt(val2)
    return val

def getUi(i, mu, ak, P, pik, qik, bik, nik, oldPosition, gammaAgent):
    sum1 = np.array([0.0, 0.0])
    sum2 = np.array([0.0, 0.0])
    for j in range(0, N):
        distance = np.linalg.norm(nodes[j] - nodes[i])
        if distance <= R:
            val1 = nodes[j] - nodes[i]
            norm = np.linalg.norm(val1)
            phiAlphaVal = phiAlpha(sigmaNorm(norm))
            val = phiAlphaVal * getNij(i, j)
            sum1 += val

            val2 = nodesvel[j] - nodesvel[i]
            sum2 += getAij(i, j) * val2
    val = C1ALPHA * sum1 + C2ALPHA * sum2 -  c1mt * (nodes[i] - gammaAgent) + \
          c1beta * phiBeta(sigmaNorm(np.linalg.norm(qik - oldPosition))) * nik + \
          c2beta * bik * (pik - nodesvel[i])
   

    return val



def plotTrajectory():
    
    for i in range(0, N):
        plt.plot(posx[i, :], posy[i, :])

    plt.xlabel('Time')
    plt.ylabel('Trajectories')
    plt.show()



def plotVelocity():
    for i in range(0, N):
        velocity_i = velmag[i, :]

        plt.plot(velocity_i)
    plt.xlabel('Time')
    plt.ylabel('Velocities')
    plt.show()



def plotConnectivity():
    plt.plot(connectivity)
    plt.xlabel('Time')
    plt.ylabel('Connectivity')
    plt.show()


def plotCenterOfMass():
    plt.plot(centerOfMass[:, 0], centerOfMass[:, 1])
    #plt.xlabel('')
    plt.ylabel('COM')
    plt.show()



simulate()
plotTrajectory()
plotVelocity()
plotConnectivity()
plotCenterOfMass()