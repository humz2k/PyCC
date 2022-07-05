import numpy as np
from scipy import spatial
from scipy import constants
import warnings
import time

warnings.filterwarnings('ignore')

class Node:
    def __init__(self,particles=None,masses=None,vol=None,pos=None,box=None):
        self.particles = particles
        self.n_particles = 0
        self.masses = masses
        self.vol = vol
        self.pos = pos
        self.box = box
        self.children = []

class DirectSum(object):
    @staticmethod
    def dists(pos,particles):
        return spatial.distance.cdist(particles,np.reshape(pos,(1,)+pos.shape))

    @staticmethod
    def phi(pos,particles,masses,eps=0):
        dists = DirectSum.dists(pos,particles).flatten()
        masses = masses[dists != 0]
        dists = dists[dists != 0]
        if eps == 0:
            potentials = (-1) * constants.G * (masses)/dists
        else:
            potentials = (-1) * constants.G * (masses)/((dists**2+eps**2)**(1/2))
        return np.sum(potentials)

class Tree:
    def __init__(self,particles,masses):
        self.particles = particles
        self.masses = masses
        self.base_node = None
        self.truncations = 0
        self.full = 0
        self.dist_calculations = 0

    def get_box(self):
        max_x = np.max(self.particles[:,0])
        min_x = np.min(self.particles[:,0])
        max_y = np.max(self.particles[:,1])
        min_y = np.min(self.particles[:,1])
        max_z = np.max(self.particles[:,2])
        min_z = np.min(self.particles[:,2])
        x = max([abs(max_x),abs(min_x)])
        y = max([abs(max_y),abs(min_y)])
        z = max([abs(max_z),abs(min_z)])
        size = max([x,y,z])
        return np.array([[-size,size],[-size,size],[-size,size]])

    def divide_box(self,box):
        new_boxes = [[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]]
        for i in range(0,4):
            new_boxes[i][0][0] = box[0][0]
            new_boxes[i][0][1] = sum(box[0])/2
        for i in range(4,8):
            new_boxes[i][0][0] = sum(box[0])/2
            new_boxes[i][0][1] = box[0][1]
        for i in range(0,4):
            new_boxes[i*2][1][0] = sum(box[1])/2
            new_boxes[i*2][1][1] = box[1][1]
        for i in range(0,4):
            new_boxes[i*2+1][1][0] = box[1][0]
            new_boxes[i*2+1][1][1] = sum(box[1])/2
        new_boxes[0][2][0] = box[2][0]
        new_boxes[0][2][1] = sum(box[2])/2
        new_boxes[1][2][0] = box[2][0]
        new_boxes[1][2][1] = sum(box[2])/2

        new_boxes[2][2][0] = sum(box[2])/2
        new_boxes[2][2][1] = box[2][1]
        new_boxes[3][2][0] = sum(box[2])/2
        new_boxes[3][2][1] = box[2][1]

        new_boxes[4][2][0] = box[2][0]
        new_boxes[4][2][1] = sum(box[2])/2
        new_boxes[5][2][0] = box[2][0]
        new_boxes[5][2][1] = sum(box[2])/2

        new_boxes[6][2][0] = sum(box[2])/2
        new_boxes[6][2][1] = box[2][1]
        new_boxes[7][2][0] = sum(box[2])/2
        new_boxes[7][2][1] = box[2][1]
        boxes = []
        for i in new_boxes:
            boxes.append(np.array(i))
        return boxes

    def build_tree(self,recursive = False):
        first = time.perf_counter()
        if recursive:
            box = self.get_box()
            vol = abs(box[0][1] - box[0][0]) * abs(box[1][1] - box[1][0]) * abs(box[2][1] - box[2][0])
            self.base_node,_,_ = self.make_node_recursive(self.particles,self.masses,box,vol)
            return self.base_node
        self.make_nodes()
        second = time.perf_counter()
        return second-first


    def particle_in_box(self,particle,box):
        x = particle[0]
        y = particle[1]
        z = particle[2]
        if (x < box[0][0] or x > box[0][1]):
            return False
        if (y < box[1][0] or y > box[1][1]):
            return False
        if (z < box[2][0] or z > box[2][1]):
            return False
        return True

    def make_nodes(self):
        box = self.get_box()
        diff = box[:,1] - box[:,0]
        abs_diff = np.abs(diff)
        vol = abs_diff[0] * abs_diff[1] * abs_diff[2]
        pos = pos = (diff)/2 + box[:,0]
        self.base_node = Node(particles=self.particles,masses=self.masses,vol=vol,pos=pos)

        stack = []
        parts = []
        masses = []

        new_parts = self.particles.tolist()
        new_masses = self.masses.tolist()
        for i in self.divide_box(box):
            temp = Node(particles=new_parts,masses = new_masses,box=i)
            self.base_node.children.append(temp)
            stack.append(temp)
            parts.append(new_parts)
            masses.append(new_masses)

        while len(stack) != 0:
            current_node = stack.pop()
            current_parts = parts.pop()
            current_masses = masses.pop()
            new_parts = []
            new_masses = []
            to_remove = []
            box = current_node.box
            for idx,i in enumerate(current_parts):
                if self.particle_in_box(i,box):
                    new_parts.append(i)
                    new_masses.append(current_masses[idx])
                    to_remove.append(idx)

            current_node.particles = np.array(new_parts)
            current_node.masses = np.array(new_masses)
            current_node.n_particles = len(current_node.particles)

            diff = box[:,1] - box[:,0]
            abs_diff = np.abs(diff)
            vol = abs_diff[0] * abs_diff[1] * abs_diff[2]
            pos = pos = (diff)/2 + box[:,0]

            current_node.vol = vol
            current_node.pos = pos

            if current_node.n_particles > 1:
                new_parts = current_node.particles.tolist()
                new_masses = current_node.masses.tolist()
                for i in self.divide_box(box):
                    temp = Node(particles=new_parts,masses = new_masses,box=i)
                    current_node.children.append(temp)
                    stack.append(temp)
                    parts.append(new_parts)
                    masses.append(new_masses)

            for i in to_remove[::-1]:
                current_parts.pop(i)
                current_masses.pop(i)
        return self.base_node

    def make_node_recursive(self,particles,particle_masses,box,vol):
        indexes = np.arange(len(particles))

        inside = indexes[np.take(particles,indexes,axis=0)[:,0] >= box[0][0]]

        if len(inside) == 0:
            return Node([],None,None,None),particles,particle_masses

        inside = inside[np.take(particles,inside,axis=0)[:,0] <= box[0][1]]

        if len(inside) == 0:
            return Node([],None,None,None),particles,particle_masses

        inside = inside[np.take(particles,inside,axis=0)[:,1] >= box[1][0]]

        if len(inside) == 0:
            return Node([],None,None,None),particles,particle_masses

        inside = inside[np.take(particles,inside,axis=0)[:,1] <= box[1][1]]

        if len(inside) == 0:
            return Node([],None,None,None),particles,particle_masses

        inside = inside[np.take(particles,inside,axis=0)[:,2] >= box[2][0]]

        if len(inside) == 0:
            return Node([],None,None,None),particles,particle_masses

        inside = inside[np.take(particles,inside,axis=0)[:,2] <= box[2][1]]

        if len(inside) == 0:
            return Node([],None,None,None),particles,particle_masses

        parts = np.take(particles,inside,axis=0)
        masses = np.take(particle_masses,inside,axis=0)

        if len(parts) == len(particles):
            remaining_parts = []
            remaining_masses = []
        else:
            outside = indexes[np.logical_not(np.isin(indexes,inside))]
            remaining_parts = np.take(particles,outside,axis=0)
            remaining_masses = np.take(particle_masses,outside,axis=0)

        pos = (box[:,1] - box[:,0])/2 + box[:,0]

        node = Node(parts,masses,vol,pos)
        if len(parts) > 1:
            for subbox in self.divide_box(box):
                next_node,parts,masses = self.make_node_recursive(parts,masses,subbox,vol/8)
                node.children.append(next_node)
                next_node.parent = node
                if len(parts) == 0:
                    break

        return node,remaining_parts,remaining_masses

    def evaluate(self,evaluate_at,eps=0,theta=1):
        #the output array for phis
        out = np.zeros(len(evaluate_at),dtype=float)
        acc = np.zeros((len(evaluate_at),3),dtype=float)

        #indexes of the phis
        indexes = np.arange(len(evaluate_at))

        stack = [self.base_node]
        positions = [indexes]

        truncations = 0
        direct = 0

        while len(stack) != 0:

            node = stack.pop()
            pos_indexes = positions.pop()

            pos = np.take(evaluate_at,pos_indexes,axis=0)

            if node.n_particles == 1:
                direct += len(pos)
                dists = spatial.distance.cdist(pos,node.particles).flatten()

                to_change = pos_indexes[dists != 0]
                parts = np.take(evaluate_at,to_change,axis=0)
                dists = dists[dists != 0]
                if len(dists) > 0:
                    if eps == 0:
                        delta_phi = (-1) * constants.G * (node.masses[0])/dists
                        muls = (constants.G * ((node.masses[0]) / (dists**3)))
                        accelerations = (node.particles - parts) * np.reshape(muls,(1,) + muls.shape).T
                    else:
                        delta_phi = (-1) * constants.G * (node.masses[0])/((dists**2+eps**2)**(1/2))
                        muls = (constants.G * node.masses[0] / (((dists**2+eps**2)**(1/2))**3))
                        accelerations = (node.particles - parts) * np.reshape(muls,(1,) + muls.shape).T
                    out[to_change] += delta_phi
                    acc[to_change] += accelerations
            else:
                dists = spatial.distance.cdist(pos,np.reshape(node.pos,(1,)+node.pos.shape)).flatten()
                check = ((node.vol/dists) <= theta)
                nexts = pos_indexes[np.logical_not(check)]
                finished = pos_indexes[check]
                if len(finished) != 0:
                    truncations += len(finished)
                    mass = np.sum(node.masses)
                    dists = dists[check]
                    to_change = finished[dists != 0]
                    parts = np.take(evaluate_at,to_change,axis=0)
                    dists = dists[dists != 0]
                    if eps == 0:
                        delta_phi = (-1) * constants.G * (mass)/dists
                        muls = (constants.G * ((mass) / (dists**3)))
                        accelerations = (-(parts - node.pos)) * np.reshape(muls,(1,) + muls.shape).T
                    else:
                        delta_phi = (-1) * constants.G * (mass)/((dists**2+eps**2)**(1/2))
                        muls = (constants.G * mass / (((dists**2+eps**2)**(1/2))**3))
                        accelerations = (-(parts - node.pos)) * np.reshape(muls,(1,) + muls.shape).T
                    out[to_change] += delta_phi
                    acc[to_change] += accelerations
                if len(nexts) > 0:
                    for child in node.children:
                        if child.n_particles > 0:
                            stack.append(child)
                            positions.append(nexts)
        return acc,out,{"truncations":truncations,"directs":direct}
