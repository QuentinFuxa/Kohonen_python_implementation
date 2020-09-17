import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation


class SOM_map:
    def __init__(self, height, width, alpha, x_size,dataset):
        self.prototypes= np.random.random((height, width, x_size))
        self.height = height
        self.width = width
        self.x_size = x_size
        self.alpha = alpha
        self.norm = 1
        self.curr_step = 0
        self.dataset = dataset
            
    def set_h_radius(self, r):
        self.norm = 1/float(r)
        
    def distance(self, prototype, X):
        return np.linalg.norm(prototype-X)

    def h(self, i, j, closest_prototype_row, closest_prototype_column):
        res  = max(0,(1-(abs(i-closest_prototype_row) + abs(j-closest_prototype_column))*self.norm))
        return res
    
    def learn(self, X, closest_prototype_row, closest_prototype_column):
        for i in range(self.height) :
            for j in range(self.width):
                self.prototypes[i,j] += self.alpha*self.h(i, j, closest_prototype_row, closest_prototype_column)*(X-self.prototypes[i,j])     
        
    def competition(self, X):
        dists = np.array([[np.linalg.norm(prototype-X) for prototype in self.prototypes[i]] for i in range(self.width)]) 
        closest_prototype_row, closest_prototype_column = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
        return closest_prototype_row, closest_prototype_column 
    
    def build(self, step):
        Rs = {0 : 40, 100: 20, 1000: 10, 5000: 5, 25000: 3}
        for i in range(step):
            X = self.dataset[self.curr_step]
            closest_prototype_row, closest_prototype_column = self.competition(X)
            self.learn(X, closest_prototype_row, closest_prototype_column)
            if self.curr_step in Rs:
                self.set_h_radius(Rs[self.curr_step%len(self.dataset)])
            self.curr_step += 1
        return self.prototypes
            
    def compute_err(self, X):
        closest_prototype_row, closest_prototype_column = self.competition(X)
        return (self.distance(self.prototypes[closest_prototype_row][closest_prototype_column], X), closest_prototype_row, closest_prototype_column)
    
    def build_err(self, dataset, threshold=500):
        err = []
        suspect = []
        for i in range(len(dataset)):
            X = dataset[i]
            dist, row, col = self.compute_err(X)
            err.append(dist)
            if dist > threshold:
                suspect.append(i)
        return (err, suspect)

class prototypes_animation:
    
    def __init__(self, size, dataset, alpha, nb_it,x_size):
        np.random.shuffle(dataset)
        self.size = size
        self.dataset = dataset
        self.nb_it = nb_it
        self.lines = []
        self.speed = 1
        self.x = np.arange(len(self.dataset[0]))
        #self.km = SOM_map(size, dataset, radius=radius, alpha=alpha, nb_it=nb_it)
        self.km = SOM_map(size,size,alpha,x_size,dataset)
        
    def init(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def animate(self, i):
        
#         clear_output(wait=True)
        y = self.km.build(self.speed)
        #y = self.km.build(self.speed)/np.max(self.km.build(self.speed))
        print(i)
        for ind, line in enumerate(self.lines):
            line.set_data(self.x, y[ind//self.size, ind%self.size])

        return self.lines
    
    def run_animation(self, speed):
        lines = []
        fig, axes = plt.subplots(nrows=self.size, ncols=self.size, figsize=(self.size*5, self.size*5))
        axes = axes.ravel()
        for ax in axes:
            lines.append(ax.plot([], [])[0])
            ax.set_xlim(0, len(self.dataset[0]))
            ax.set_ylim(0, 1)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
        self.lines = lines
        self.speed = speed
            
        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init, frames=self.nb_it, interval=0, blit=True)
        plt.show()
