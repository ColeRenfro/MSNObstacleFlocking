import unittest
import numpy as np
from Nodes import Nodes
from Math import Math


class Tester(unittest.TestCase):
    
    obstaclesLoc = np.array([(100, 10)])
    obstacleRadius = np.array([15])
    numObstacles = obstaclesLoc.shape[0]
    gammaAgent = Nodes(250,40)
    nodes = np.zeros(2, dtype=object)
    nodes[0] = Nodes(100,40)
    nodes[1] = Nodes(105,42)
    i = 0
    j = 1
    oldPosition = np.array([nodes[0].posx,nodes[0].posy])
    mu = obstacleRadius / np.linalg.norm(oldPosition - obstaclesLoc)
    qik = mu * oldPosition + (1 - mu) * obstaclesLoc
    bik = Math.bump(Math.sigmaNorm(np.linalg.norm(qik - oldPosition)) /  Math.dBeta)
    nik = (qik - oldPosition) / np.sqrt(1 + 0.1 * np.linalg.norm(qik - oldPosition) ** 2)
    ak = (oldPosition - obstaclesLoc) / np.linalg.norm(oldPosition - obstaclesLoc)
    P = np.eye(2) - np.outer(ak, ak)
    pik = mu * P @ [1,1]
    ui = Math.getUi(i ,pik, qik, bik, nik, oldPosition, nodes, gammaAgent)
     
    def test_MathDistance(self):
        node1 = Nodes(1,1)
        node2 = Nodes(1,5)
        
        dis = Math.distanceBetweenNodes(node1,node2)
        #print(dis)
        self.assertEqual(dis, 4)
        
    def test_phiCalculation(self):
        z = 1
        self.assertAlmostEqual(np.round(Math.phi(z), decimals=4), 3.5355, places = 3)
        
    def test_phiAplhaCalculation(self):
        z = 1
        self.assertAlmostEqual(np.round(Math.phiAlpha(z), decimals=4), -0.0, places = 3)
    
    def test_phiBetaCalculation(self):
        z = 1
        self.assertAlmostEqual(np.round(Math.phiBeta(z), decimals=4), -0.0, places = 3)
  
    def test_getNij(self):
        val = 3.1586
        self.assertAlmostEqual(np.round(Math.getNij(0,1,self.nodes), decimals=4), val, places = 4)
        
    def test_bik(self):

        #print(self.bik)
        self.assertAlmostEqual(self.bik, 0.0, places=2)
        
    def test_sigmaNorm(self):
        z = 1
        self.assertAlmostEqual(np.round(Math.sigmaNorm(z),decimals=4),0.488,places = 3)
        
       
    
    def test_getUI(self):
        print(self.ui[0,:])
        #self.assertAlmostEqual(self.ui.any(), -177.71, places=2)
        self.assertEqual(np.allclose(self.ui,[-177.71,-177.71]), True)


     
if __name__ == "__main__":
    unittest.main
    print("All tests passed") 