import numpy as np
import matplotlib.pyplot as plt
from Nodes import Nodes 
from Math import Math

#==============Parameters start===========
x = 70
y = 70
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
a = 5
b = 5
c = np.abs(a-b)/np.sqrt(4*a*b)

gammaAgent = Nodes(250,40)



iterations = 900 #total time of simulation
interval = 50 #interval of snap time 
# if interval is 10 and iterations is 100, then it runs for 0.1 secs and takes 10 snapshots


nodes = np.zeros(N, dtype=object)
for i in range(N):
    nodes[i] =  Nodes(np.random.uniform(high=x, low=0), np.random.uniform(high=y, low=0)) #np.random.rand(N, M) * x
#nodesvel = np.zeros([N, M])
velmag = np.zeros([N, iterations])
connectivity = np.zeros([iterations, 1])

fig = plt.figure()



c1mt = 1.1
c2mt = 2 * np.sqrt(c1mt)
c1beta = 1500
c2beta = 2*np.sqrt(c1beta)
rPrime = 0.22 * K * R
dPrime = 15
 
obstaclesLoc = np.array([(100, 10), (110,60)])
obstacleRadius = np.array([15, 10])
numObstacles = obstaclesLoc.shape[0]




centerOfMass = np.zeros([iterations, M])

#==================Parameters end===================

def simulate():
    print(nodes[0].posx)
    for t in range(0, iterations):
        # print(np.linalg.matrix_rank(adjacencyMatrix))
        AdjMat = MakeAdjMat()
        # print(np.linalg.matrix_rank(adjacencyMatrix))
        connectivity[t] = (1 / N) * np.linalg.matrix_rank(AdjMat)
        posxs = [node.posx for node in nodes]
        posys = [node.posy for node in nodes]
        centerOfMass[t] = np.array([np.mean(posxs), np.mean(posys)])

        # print(t)
        if t == 0:
            plotNeighbors(t)
            #for i in range(0, N):
            #    posx[i, t] = nodes[i, 0]
            #    posy[i, t] = nodes[i, 1]
        else:
            if t == 3:
                o = 9
                pass
            for i in range(0, N):
                # ui = getUi(i)
                oldVelocity = nodes[i].vel
                oldPosition = np.array([nodes[i].posx,nodes[i].posy])
                # newVelocity = oldVelocity + ui * DELTAT
                
                for BetaLoc in obstaclesLoc:
                    mu = obstacleRadius / np.linalg.norm(oldPosition - BetaLoc)
                    ak = (oldPosition - BetaLoc) / np.linalg.norm(oldPosition - BetaLoc)
                    P = np.eye(2) - np.outer(ak, ak)
                    #print(f"P: {P.shape} OldVel: {oldVelocity.shape} MU: {mu.shape} ")
                    pik = mu * P @ oldVelocity
                    
                    qik = mu * oldPosition + (1 - mu) * BetaLoc
                    bik = Math.bump(Math.sigmaNorm(np.linalg.norm(qik - oldPosition)) / dBeta)
                    nik = (qik - oldPosition) / np.sqrt(1 + EPSILON * np.linalg.norm(qik - oldPosition) ** 2)

                    ui = Math.getUi(i,pik, qik, bik, nik, oldPosition, nodes, gammaAgent)
                    newPosition = oldPosition + DELTAT * oldVelocity + (DELTAT ** 2 / 2) * ui
                    newVelocity = (newPosition - oldPosition) / DELTAT

                    # Update node properties
                    nodes[i].posx = newPosition[0]
                    nodes[i].posy = newPosition[1]
                    nodes[i].vel = newVelocity
                    #nodes[i, :] = newPosition
                    velmag[i, t] = np.linalg.norm(newVelocity)

        if (t+1) % interval == 0:
            plotNeighbors(t)



rBeta = Math.sigmaNorm(np.linalg.norm(rPrime))
dBeta = Math.sigmaNorm(np.linalg.norm(dPrime))
s = 1

def MakeAdjMat():
    adjacencyMatrix = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                distance = Math.distanceBetweenNodes(nodes[i], nodes[j])
                #print(distance)
                if distance <= R:
                    adjacencyMatrix[i, j] = 1
                    #print("Node connected")
    return adjacencyMatrix


def plotNeighbors(t):
    plt.figure(figsize = (10,5))
    plt.plot(gammaAgent.posx, gammaAgent.posy, 'ro', color='purple')
    # plt.plot(centerOfMass[0:t, 0], centerOfMass[0:t, 1], color='black')
    for i in range(0, numObstacles):
        plt.gcf().gca().add_artist(plt.Circle((obstaclesLoc[i, 0], obstaclesLoc[i, 1]), obstacleRadius[i], color='green'))
    #plt.plot(nodes)
    for i in range(0, N):
        plt.plot(nodes[i].posx,nodes[i].posy, 'ro')
        for j in range(0, N):
            distance = Math.distanceBetweenNodes(nodes[i],nodes[j])
            if distance <= R:
                plt.plot([nodes[i].posx, nodes[j].posx], [nodes[i].posy, nodes[j].posy], 'b-', lw=0.5)
    plt.show()




def plotTrajectory():
    
    for i in range(0, N):
        plt.plot(nodes[i].posx, nodes[i].posy)

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
    #plt.xlabel('Time')
    plt.ylabel('COM')
    plt.show()



simulate()
plotTrajectory()
plotVelocity()
plotConnectivity()
plotCenterOfMass()