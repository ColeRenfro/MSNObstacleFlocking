import matplotlib.pyplot as plt
from Math import Math

class Plot:

    def plotNeighbors(t,gammaAgent,numObstacles,obstaclesLoc,obstacleRadius,nodes,N,R):
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




    def plotTrajectory(nodes,N):
    
        for i in range(0, N):
            plt.plot(nodes[i].posx, nodes[i].posy)

        plt.xlabel('Time')
        plt.ylabel('Trajectories')
        plt.show()



    def plotVelocity(velmag,N):
        for i in range(0, N):
            velocity_i = velmag[i, :]

            plt.plot(velocity_i)
        plt.xlabel('Time')
        plt.ylabel('Velocities')
        plt.show()



    def plotConnectivity(connectivity):
        plt.plot(connectivity)
        plt.xlabel('Time')
        plt.ylabel('Connectivity')
        plt.show()


    def plotCenterOfMass(centerOfMass):
        plt.plot(centerOfMass[:, 0], centerOfMass[:, 1])
        #plt.xlabel('Time')
        plt.ylabel('COM')
        plt.show()

