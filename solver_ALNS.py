#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import random
from copy import copy, deepcopy
from time import time
from tsp_implementation import tabu_search as tsp_tabu_search

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def compute_obj(vehicle_tours, D):

    vehicle_count=len(vehicle_tours)
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 1:
            for i in range(len(vehicle_tour)):
                c1=vehicle_tour[i]
                c2=vehicle_tour[(i+1)%len(vehicle_tour)]
                obj+=D[c1, c2]
    return obj

class VRP():

    def length(self, customer1, customer2):
        return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

    def compute_D(self):
        N=len(self.customers)
        D=np.zeros((N,N))
        for n1, c1 in enumerate(self.customers):
            for n2, c2 in enumerate(self.customers):
                D[n1, n2] = self.length(c1, c2)
        return D

    def LNS(self, time_lim):

        temperature0=self.temperature
        T0=time()

        curr_obj=self.obj
        curr_tours=[[n_c for n_c in tour] for tour in self.vehicle_tours]
        curr_loads=[load for load in self.vehicle_loads]

        it_no_improvement=0
        while ( (time() - T0) < time_lim and self.obj > self.stop):
            print((time() - T0))
            u=random.uniform(0, 1)
            rem_num=random.randint(2,max(2,np.ceil(self.vehicle_count * self.fraction_destroy)  ))
            if u < self.cdf_rem_types[0]:
                rem_type=1
            elif u < self.cdf_rem_types[1]:
                rem_type=2
            else:
                rem_type=3
            it_no_improvement+=1
            temp_tours, temp_loads, missing_nodes, influenced_route_idxs=self.remove_part(curr_tours, curr_loads, rem_type, rem_num)
            temp_tours, temp_loads=self.complete_routes(temp_tours, temp_loads, self.repair_tries, missing_nodes)
            for idx in influenced_route_idxs:
                if len(temp_tours[idx]) > 3:
                    temp_tours[idx]=self.improve_tour(temp_tours[idx])  
            temp_obj=compute_obj(temp_tours, self.D)
            
            if temp_obj < self.obj - 10**(-5):
                old_obj=self.obj
                self.obj=temp_obj
                self.vehicle_tours=[[n_c for n_c in tour] for tour in temp_tours]
                self.vehicle_loads=[load for load in temp_loads]
                print(old_obj, '-->', self.obj )

            if random.uniform(0, 1) < (np.exp(-(temp_obj - self.obj) / temperature0)):
                curr_obj=temp_obj
                curr_tours=[[n_c for n_c in tour] for tour in temp_tours]
                curr_loads=[load for load in temp_loads]
                it_no_improvement=0

            temperature0=temperature0*self.alpha
            if it_no_improvement > 1000:
                temperature0=self.temperature

        if self.obj > self.stop:
            temp1=self.time_tabu; temp2=self.dt_dive_tabu; temp3=self.dt_renew_tabu;
            self.time_tabu=10
            self.dt_dive_tabu=1
            self.dt_renew_tabu=2
            self.improve_full_solution_tsp()
            self.time_tabu=temp1; self.dt_dive_tabu=temp2; self.dt_renew_tabu=temp3;
        return

    def remove_part(self, vehicle_tours, vehicle_loads, rem_type, rem_num):
        # rem_type == 1 --> remove rem_num randomly selected vehicles
        # rem_type == 2 --> remove route with longest edge and (rem_num-1) other random routes
        # rem_type == 3 --> remove random route and the (rem_num-1)-nearest by routes.

        return_vehicle_tours=deepcopy(vehicle_tours)
        return_vehicle_loads=deepcopy(vehicle_loads)
        influenced_routes=list()

        missing_nodes=list()
        if rem_type == 1:
            for v in random.sample(range(self.vehicle_count), rem_num):
                missing_nodes.extend(return_vehicle_tours[v][1:])
                return_vehicle_tours[v]=[0]
                return_vehicle_loads[v]=0
                influenced_routes.append(v)
        elif rem_type == 2:
            longest_dist=0
            worst_veh=0
            for v in range(self.vehicle_count):
                for i in range(len(vehicle_tours[v])):
                    if self.D[vehicle_tours[v][i], vehicle_tours[v][(i+1) % len(vehicle_tours[v])]]:
                        longest_dist=self.D[vehicle_tours[v][i], vehicle_tours[v][(i+1) % len(vehicle_tours[v])]]
                        worst_veh=v
            for v in random.sample(range(self.vehicle_count-1), rem_num-1):
                if v == worst_veh:
                    veh=self.vehicle_count-1
                else:
                    veh=v
                missing_nodes.extend(return_vehicle_tours[v][1:])
                return_vehicle_tours[v]=[0]
                return_vehicle_loads[v]=0
                influenced_routes.append(v)
        elif rem_type == 3:
            v1=random.randint(0, self.vehicle_count-1)
            mean_v1=0
            for n_c in self.vehicle_tours[v1]:
                mean_v1+=self.D[0, n_c]
            mean_v1/=len(self.vehicle_tours[v1])
            dist_to_v1=[0 for v in range(self.vehicle_count)]
            for v2 in range(self.vehicle_count):
                mean_v2=0
                for n_c in self.vehicle_tours[v2]:
                    mean_v2+=self.D[0,n_c]
                mean_v2/=len(self.vehicle_tours[v2])
                dist_to_v1[v2]=np.abs(mean_v2 - mean_v1)
            for i, v in enumerate(np.argsort(dist_to_v1)):
                missing_nodes.extend(return_vehicle_tours[v][1:])
                return_vehicle_tours[v]=[0]
                return_vehicle_loads[v]=0
                influenced_routes.append(v)
                if i >= rem_num:
                    break
        return return_vehicle_tours, return_vehicle_loads, missing_nodes, influenced_routes

    def complete_routes(self, input_vehicle_tours, input_vehicle_loads, nr_tries, missing_nodes):

        best_obj=9999999
        for nr_try in range(nr_tries):
            while True:
                vehicle_loads=[load for load in input_vehicle_loads]
                vehicle_tours=[[n_c for n_c in tour] for tour in input_vehicle_tours]
                repeat=False
                iterator_customers=list(missing_nodes)
                random.shuffle(iterator_customers)
                for n_c in iterator_customers:
                    c=self.customers[n_c]
                    cost_add_to_v=[0 for v in range(self.vehicle_count)]
                    all_bad=True
                    for v in range(self.vehicle_count):
                        cost_add_to_v[v]=999999
                        if (c.demand + vehicle_loads[v]) <= self.vehicle_capacity:
                            all_bad=False
                            for n in range(len(vehicle_tours[v])):
                                cost1=self.D[vehicle_tours[v][n], n_c]
                                cost2=self.D[n_c,vehicle_tours[v][(n+1)%len(vehicle_tours[v])]]
                                cost_add_to_v[v]=min(cost_add_to_v[v], cost1+cost2)
                    if all_bad:
                        repeat=True
                        break

                    chosen_v=np.argmin(cost_add_to_v)
                    chosen_place=0
                    c1=vehicle_tours[chosen_v][0]
                    c2=vehicle_tours[chosen_v][(0+1) % len(vehicle_tours[chosen_v])]
                    cost_place=self.D[c1, n_c] + self.D[n_c, c2] - self.D[c1, c2]

                    for n in range(1, len(vehicle_tours[chosen_v])):
                        c1=vehicle_tours[chosen_v][n]
                        c2=vehicle_tours[chosen_v][(n+1) % len(vehicle_tours[chosen_v])]
                        if self.D[c1, n_c] + self.D[n_c, c2] - self.D[c1, c2] < cost_place:
                            cost_place=self.D[c1, n_c] + self.D[n_c, c2] - self.D[c1, c2]
                            chosen_place=n

                    vehicle_tours[chosen_v].insert(chosen_place+1, n_c)
                    vehicle_loads[chosen_v]+=c.demand
                if not repeat:
                    break
            obj=compute_obj(vehicle_tours, self.D)
            if obj < best_obj:
                best_obj=obj
                best_tours=[[n_c for n_c in tour] for tour in vehicle_tours]
                best_loads=[load for load in vehicle_loads]
        return best_tours, best_loads

    def plot_routes(self):
        x = [self.customers[i][2] for i in range( len(self.customers))]
        y = [self.customers[i][3] for i in range( len(self.customers))]
        plt.scatter(x, y)
        for route in self.vehicle_tours:
            x = [self.customers[0][2]] + [self.customers[i][2] for i in route] + [self.customers[0][2]]
            y = [self.customers[0][3]] + [self.customers[i][3] for i in route] + [self.customers[0][3]]
            plt.plot( x, y)
        plt.show()
        return

    def improve_tour(self, vehicle_tour):
        # Use Tabu Search for TSP to improve one route.

        len_tour=len(vehicle_tour)
        obj=0
        for i in range(len_tour):
            obj+=self.D[vehicle_tour[i], vehicle_tour[(i+1) % len_tour]]

        mapping=[0 for c in self.customers]
        inv_mapping=[n_c for n_c in vehicle_tour]
        for i, n_c in enumerate(vehicle_tour):
            mapping[n_c]=i

        solution=[i for i in range(len_tour)]
        temp_D=np.zeros((len_tour, len_tour))
        for i in range(len_tour):
            for j in range(len_tour):
                if i==j:
                    temp_D[i,j]=9999999
                else:
                    temp_D[i,j]=self.D[inv_mapping[i], inv_mapping[j]]

        tabu_length=len(solution); short_tabu_length=np.floor(len(solution)*0.9)

        solution, obj = tsp_tabu_search(solution, obj, temp_D, self.time_tabu, self.dt_dive_tabu, self.dt_renew_tabu, tabu_length, short_tabu_length)
        
        place_0_sol=-1
        for i, s in enumerate(solution):
            if s == 0 :
                place_0_sol = i
                break
        solution=[solution[(i + place_0_sol) % len(solution)] for i in range(len(solution))]
        tour=[inv_mapping[s] for s in solution]
        obj=0
        for i in range(len_tour):
            obj+=self.D[tour[i], tour[(i+1) % len_tour]]
        return tour

    def improve_full_solution_tsp(self):

        for i, vehicle_tour in enumerate(self.vehicle_tours):
            if len(vehicle_tour) > 3:
                self.vehicle_tours[i]=self.improve_tour(vehicle_tour)
        self.obj=compute_obj(self.vehicle_tours, self.D)
        return

    def __init__(self, customers, vehicle_count, vehicle_capacity, temperature, alpha, time_lim, time_tabu, dt_dive_tabu, dt_renew_tabu):

        # Set hyperparemeters for each test case
        if len(customers) == 16:
            # vrp_16_3_1
            self.stop=280
            self.nr_tries=400
            self.time_tabu=1; self.dt_dive_tabu=0.1; self.dt_renew_tabu=0.2
            self.repair_tries=400; self.cdf_rem_types=[1/3, 2/3, 1]
            self.fraction_destroy=1/3
        elif len(customers) == 26:
            #vrp_26_8_1
            self.stop=630
            self.nr_tries=400
            self.time_tabu=1; self.dt_dive_tabu=0.1; self.dt_renew_tabu=0.2
            self.repair_tries=400; self.cdf_rem_types=[1/3, 2/3, 1]
            self.fraction_destroy=1/3
        elif len(customers) == 51:
            #vrp_51_5_1
            self.stop=540
            self.nr_tries=400
            self.time_tabu=1; self.dt_dive_tabu=0.1; self.dt_renew_tabu=0.2
            self.repair_tries=400; self.cdf_rem_types=[1/3, 1/3, 1]
            self.fraction_destroy=1/3
        elif len(customers) == 101:
            #vrp_101_10_1
            self.stop=830
            self.nr_tries=400
            self.time_tabu=1; self.dt_dive_tabu=0.1; self.dt_renew_tabu=0.2
            self.repair_tries=400; self.cdf_rem_types=[1/10, 2/10, 1]
            self.fraction_destroy=1/3
        elif len(customers) == 200:
            #vrp_200_16_1
            self.stop=1400
            self.nr_tries=1
            self.time_tabu=2; self.dt_dive_tabu=0.1; self.dt_renew_tabu=0.2
            self.repair_tries=400; self.cdf_rem_types=[1/3, 2/3, 1]
            self.fraction_destroy=1/3
        else:
            #vrp_421_41_1
            self.stop=2392
            self.nr_tries=400
            self.time_tabu=2; self.dt_dive_tabu=0.1; self.dt_renew_tabu=0.2
            self.repair_tries=400; self.cdf_rem_types=[1/3, 2/3, 1]
            self.fraction_destroy=1/10

        self.depot=0
        self.temperature=temperature
        self.alpha=alpha
        self.customers=customers
        self.vehicle_count=vehicle_count
        self.vehicle_capacity=vehicle_capacity
        self.vehicle_loads=[0 for v in range(vehicle_count)]
        self.D=self.compute_D()
        self.N=len(self.D)
        self.vehicle_tours=[[0] for v in range(self.vehicle_count)]
        self.vehicle_tours, self.vehicle_loads=self.complete_routes(self.vehicle_tours, self.vehicle_loads, self.nr_tries, range(1,self.N))
        self.obj=compute_obj(self.vehicle_tours, self.D)
        print('Initial Solution pre Tabu : ', self.obj)
        self.improve_full_solution_tsp()
        self.obj=compute_obj(self.vehicle_tours, self.D)
        print('Initial Solution after Tabu : ', self.obj)
        return

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = [] #customers[0] is the depot
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0] 

    # Set hyperparameters
    temperature=100
    alpha=0.99
    time_lim=60*60

    time_tabu=1
    dt_dive_tabu=0.1
    dt_renew_tabu=0.2

    model=VRP(customers, vehicle_count, vehicle_capacity, temperature, alpha, time_lim, time_tabu, dt_dive_tabu, dt_renew_tabu)
    model.LNS(time_lim)
    model.plot_routes()

    # prepare the solution in the specified output format
    outputData = '%.2f' % model.obj + ' ' + str(0) + '\n'
    for v in range(0, model.vehicle_count):
        outputData += ' '.join([str(c) for c in model.vehicle_tours[v]]) + ' ' + str(model.depot) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

