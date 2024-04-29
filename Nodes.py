import numpy as np

class Nodes:
    def __init__(self, posx, posy):
        self.posx = posx
        self.posy = posy
        self.vel = np.array([0,0])
    
    def move(self):
        self.posx += self.vel[0]
        self.posy += self.vel[1]
        return self.posx, self.posy
        