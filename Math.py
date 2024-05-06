import numpy as np
from Nodes import Nodes

class Math:
    
    
    EPSILON = 0.1
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
    c1mt = 1.1
    c2mt = 2 * np.sqrt(c1mt)
    c1beta = 1500
    c2beta = 2*np.sqrt(c1beta)
    rPrime = 0.22 * K * R
    dPrime = 15
    

    
    def bump(z):
        if 0 <= z < Math.H:
            return 0
        elif Math.H <= z < 1:
            val = (z - Math.H)/(1 - Math.H)
            val = np.cos(np.pi * val)
            val = (1+val)/2
            return val
        else:
            return 0
    
    def distanceBetweenNodes(node1,node2):
        difx = node1.posx - node2.posx
        dify = node1.posx - node2.posy
        return np.sqrt(difx ** 2 + dify ** 2)
    

    def sigma1(z):
        val = 1 + z **2
        val = np.sqrt(val)
        val = z/val
        return val
    


    def phi(z):
        val1 = Math.a + Math.b
        val2 = Math.sigma1(z + Math.c)
        val3 = Math.a - Math.b
        val = val1 * val2 + val3
        val = val / 2
        return val

    def phiAlpha(z):
        input1 = z/Math.sigmaNorm(Math.R) 
        input2 = z - Math.sigmaNorm(Math.D)  #
        val1 = Math.bump(input1)
        val2 = Math.phi(input2)
        val = val1 * val2
        return val

    def phiBeta(z):
        val1 = Math.bump(z/Math.dBeta)
        val2 = Math.sigma1(z-Math.dBeta) - 1
        return val1 * val2

    def getAij(i, j,nodes):
        val1 = Math.distanceBetweenNodes(nodes[j],nodes[i])
        norm = np.linalg.norm(val1)
        val2 = Math.sigmaNorm(norm)/Math.sigmaNorm(Math.R)
        val = Math.bump(val2)
        return val

    def getNij(i, j, nodes):
        val1 = Math.distanceBetweenNodes(nodes[j],nodes[i])
        norm = np.linalg.norm(val1)
        val2 = 1 + Math.EPSILON * norm**2
        val = val1/np.sqrt(val2)
        return val

    def sigmaNorm(z):
        val = 0.1*(z**2)
        val = np.sqrt(1 + val) - 1
        val = val/0.1
        return val
    
    rBeta = sigmaNorm(np.linalg.norm(rPrime))
    dBeta = sigmaNorm(np.linalg.norm(dPrime))
    s = 1

    def getUi(i, pik, qik, bik, nik, oldPosition, nodes, gammaAgent):
        sum1 = np.array([0.0, 0.0])
        sum2 = np.array([0.0, 0.0])
        for j in range(0, 1):
            distance = Math.distanceBetweenNodes(nodes[i], nodes[j])
            if distance <= Math.R:
                val1 = Math.distanceBetweenNodes(nodes[j], nodes[i])
                norm = np.linalg.norm(val1)
                phiAlphaVal = Math.phiAlpha(Math.sigmaNorm(norm))
                val = phiAlphaVal * Math.getNij(i, j, nodes)
                sum1 += val

                val2 = nodes[j].vel - nodes[i].vel
                sum2 += Math.getAij(i, j, nodes) * val2
        val = Math.C1ALPHA * sum1 + Math.C2ALPHA * sum2 - Math.c1mt * (Math.distanceBetweenNodes(nodes[i],gammaAgent)) + \
            Math.c1beta * Math.phiBeta(Math.sigmaNorm(np.linalg.norm(qik - oldPosition))) * nik + \
            Math.c2beta * bik * (pik - nodes[i].vel)

        return val

