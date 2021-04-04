#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import random
from copy import copy, deepcopy
from time import time

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def compute_D(customers):
    N=len(customers)
    D=np.zeros((N,N))
    for n1, c1 in enumerate(customers):
        for n2, c2 in enumerate(customers):
            D[n1, n2] = length(c1, c2)
    return D

def init_solver(D, part_solution):
    # part_solution is a part of the solution with
    # len(part_solution) < len(solution).

    n=len(D)
    if not part_solution:
        node_degree=[0 for i in range(n)]
        connected=np.zeros(np.shape(D), dtype = bool)
        connect_list=dict()
        for i in range(n):
            connect_list[i]=[i]
    else:
        node_degree=[0 for i in range(n)]
        node_degree[part_solution[0]]=1
        node_degree[part_solution[-1]]=1
        for i in range(1, len(part_solution)-1):
            node_degree[part_solution[i]]=2
        connected=np.zeros(np.shape(D), dtype = bool)
        for v1 in part_solution:
            for v2 in part_solution:
                if (v1 != v2):
                    connected[v1, v2]=True
        connect_list=dict()
        for i in range(n):
            connect_list[i]=[i]

        for v in part_solution:
            connect_list[v]=part_solution

    D_flat=np.ndarray.flatten(D)
    sort_idxs=np.argsort(D_flat)
    edges=dict()
    for i in range(n):
        edges[i]=list()
    if part_solution:
        for i, v in enumerate(part_solution):
            if i == 0:
                edges[v].append(part_solution[i+1])
            elif i == len(part_solution)-1:
                edges[v].append(part_solution[i-1])
            else:
                edges[v].append(part_solution[i-1])
                edges[v].append(part_solution[i+1])

    j=0
    if part_solution:
        mm=len(part_solution)-1
    else:
        mm=0

    for qqq in range(n-1-mm):
        while True:
            sort_idx=sort_idxs[j]
            double_idx=np.unravel_index(sort_idx, np.shape(D))
            i1=double_idx[0]
            i2=double_idx[1]
            if (i1 == i2):
                continue

            if (node_degree[i1] == 2) or (node_degree[i2] == 2):
                j+=1
                continue
            if connected[i1, i2]:
                j+=1
                continue
            node_degree[i1]+=1
            node_degree[i2]+=1
            edges[i1].append(i2)
            edges[i2].append(i1)
            connected[i1, i2] = True
            connected[i2, i1] = True
            connect_list[i1].append(i2)
            connect_list[i2].append(i1)
            for i3 in range(n):
                if connected[i2, i3]:
                    for i4 in connect_list[i1]:
                        if not connected[i3, i4]:
                            connected[i4, i3] = True
                            connected[i3, i4] = True
                            connect_list[i3].append(i4)
                            connect_list[i4].append(i3)

                if connected[i1, i3]:
                    for i4 in connect_list[i2]:
                        if not connected[i3, i4]:
                            connected[i4, i3] = True
                            connected[i3, i4] = True
                            connect_list[i3].append(i4)
                            connect_list[i4].append(i3)
            j+=1
            break

    last2=list()
    for i in range(n):
        if len(edges[i]) == 1:
            last2.append(i)
    edges[last2[0]].append(last2[1])
    edges[last2[1]].append(last2[0])

    in_sol=[False for v in range(n)]
    obj=0
    solution=list()
    
    v1=0
    in_sol[v1]=True
    solution.append(v1)

    for i in range(n-1):
        for v2 in edges[v1]:
            if not in_sol[v2]:
                in_sol[v2]=True
                solution.append(v2)
                obj+=D[v1, v2]
                v1=v2
                break
    obj+=D[solution[0], solution[-1]]

    return solution, obj

def replace_out_piece(solution, D, i1, i2, shift):
    # Assume v1 < v2
    # We shift the solution by "shift" and then only keep the path from v1 to v2
    
    n=len(solution)
    shifted_solution=[-1 for i in range(n)]
    for i, val in enumerate(solution):
        shifted_solution[(i + shift) % n]=val
    if (i2 - i1) > 1:
        part_solution=list()
        for i in shifted_solution[i1:i2]:
            part_solution.append(i)
    else:
        part_solution=False

    solution, obj=init_solver(D, part_solution)
    return solution, obj

def two_opt(solution, obj, inv_solution, D, meanD, sigD, nearby_list, diving, tabu_list, tabu_length, short_tabu_list, short_tabu_length):

    n = len(solution)

    Dsolution=np.array([0 for i in range(n)])
    for i in range(n):
        Dsolution[i]=D[solution[i], solution[(i+1)%n]]


    indxs_order=np.argsort(-Dsolution)

    j=0
    while True:
        i1=indxs_order[j%n]#randint(0, n-1)
        j+=1
        i2=(i1+1) % n

        #print(np.random.normal(loc=(D[solution[i1], solution[i2]] - meanD)/sigD, scale=1.0))
        if True or np.random.normal(loc=(D[solution[i1], solution[i2]] - meanD)/sigD, scale=1.0) > 0.4:
            v1=solution[i1]
            v2=solution[i2]
            if (v1, v2) in short_tabu_list:
                continue
            short_tabu_list.append((v1, v2))
            while len(short_tabu_list) > short_tabu_length:
                short_tabu_list.pop(0)
            break

    for v4 in nearby_list[v2]:
        i4 = inv_solution[v4]
        i3=(i4-1)%n
        v3 = solution[i3]
        if (v1, v2, v3, v4, np.floor(obj)) in tabu_list:
            continue
        if (v2 == v3) or (v1 == v3) or (v2 == v4):
            continue
        obj_diff = D[v1, v3] + D[v2, v4] - D[v1, v2] - D[v3, v4]
        if (obj_diff < 0) or (not diving and np.random.normal(loc=-obj_diff/sigD, scale=1.0) > 0.4):
            temp_solution=copy(solution)
            if i2 <= i4:
                num_change=i4-i2
            else:
                num_change=i4-i2+n
            for i in range(num_change):
                solution[(i2+i)%n] = temp_solution[(i3-i)%n]

            tabu_list.append((v1, v2, v3, v4, np.floor(obj)))
            obj=obj+obj_diff
            while len(tabu_list) > tabu_length:
                tabu_list.pop(0)
            inv_solution=invert_solution(solution)
            temp=0
            for i in range(n):
                temp+=D[solution[i], solution[(i+1)%n]]
            return solution, inv_solution, obj, tabu_list, short_tabu_list

    return solution, inv_solution, obj, tabu_list, short_tabu_list

def invert_solution(solution):

    n=len(solution)
    inv_solution = [i for i in range(n)]
    for i, v in enumerate(solution):
        inv_solution[v]=i

    return inv_solution

def compute_mean_sig_D(solution, D):

    n=len(solution)
    meanD=0
    varD=0
    for i in range(n):
        meanD+=D[solution[i], solution[(i+1)%n]]
    for i in range(n):
        varD+=(D[solution[i], solution[(i+1)%n]] - meanD)**2
    sigD=np.sqrt(varD)
    return meanD, sigD

def tabu_search(solution, obj, D, T, dt_dive, dt_renew, tabu_length, short_tabu_length):

    n=len(solution)
    inv_solution = invert_solution(solution)
    best_solution=copy(solution)
    best_inv_solution=copy(solution)
    best_obj=obj
    meanD, sigD = compute_mean_sig_D(solution, D)
    nearby_list=create_nearby_list(D)
    diving=False
    T0=time()
    curr_time=time() - T0
    time_dive_switch=time() - T0
    time_renew_search=time() - T0
    tabu_list=list()
    short_tabu_list=list()
    prev_obj=999999999999999999

    while curr_time < T:
        curr_time=time() - T0
        
        if (curr_time - time_dive_switch) > dt_dive:
            diving=(not diving)
            time_dive_switch=curr_time
        if (curr_time - time_renew_search) > dt_renew:
            diving=True
            time_dive_switch=curr_time
            time_renew_search=curr_time
            shift=random.randint(0,n-1)
            i1=random.randint(0,n//4)
            i2=n-1-random.randint(0,n//4)
            solution, obj=replace_out_piece(best_solution, D, i1, i2, shift)
            prev_obj=obj
            inv_solution=invert_solution(solution)
            tabu_list=list()
            #print('After renew:', obj)

        #K_opt(solution, obj, inv_solution, D, meanD, sigD, nearby_list, diving, tabu_list, tabu_length)
        solution, inv_solution, obj, tabu_list, short_tabu_list=two_opt(solution, obj, inv_solution, D, meanD, sigD, nearby_list, diving, tabu_list, tabu_length, short_tabu_list, short_tabu_length)

        if obj < prev_obj:
            prev_obj=obj
            time_dive_switch=curr_time
            time_renew_search=curr_time

        if obj < best_obj:
            #if obj < best_obj-10**(-5):
                #print(best_obj)
            best_solution=copy(solution)
            best_inv_solution=copy(inv_solution)
            best_obj=obj
            
    return best_solution, best_obj

def create_nearby_list(D):
    # For each vertex we enlist the n nearest neighbors

    n=len(D)
    nearby_list=dict()
    for v in range(len(D)):
        nearby_list[v]=list()
        near=np.argsort(D[v,:])
        for i in range(n):
            nearby_list[v].append(near[i])

    return nearby_list

def find_sol(vehicle_tours, customers, vehicle_count, vehicle_capacity, D, perc, T_max, stop, T_tabu, dt_dive, dt_renew):

    T0=time()
    curr_time=time() - T0
    best_tours=copy(vehicle_tours)
    best_obj=compute_obj(vehicle_tours, D)
    prev_print=10
    while True:
        curr_time=time() - T0
        print(np.floor(100*curr_time/T_max), '%')
        if (curr_time > T_max or best_obj < stop):
            break
        vehicle_tours, obj=improve_initial_solution_tsp(customers, vehicle_tours, D, T_tabu, dt_dive, dt_renew)
        if obj < best_obj:
            best_obj=obj
            print(best_obj)
            best_tours=deepcopy(vehicle_tours)
        break
        vehicle_tours, obj = partial_init_solution(vehicle_tours, customers, vehicle_count, vehicle_capacity, D, perc)
        if obj < best_obj:
            best_obj=obj
            print(best_obj)
            best_tours=deepcopy(vehicle_tours)
    return best_tours, best_obj

def partial_init_solution(vehicle_tours, customers, vehicle_count, vehicle_capacity, D, perc):

    iterator_customers=list()
    vehicle_tours_init=[[n_c for n_c in vehicle_tours[v]] for v in range(vehicle_count)]
    for v in range(vehicle_count):
        for n in reversed(range(1, len(vehicle_tours_init[v]))):
            if random.uniform(0, 1) < perc:
                iterator_customers.append(vehicle_tours_init[v][n])
                vehicle_tours_init[v].pop(n)

    vehicle_loads_init=[0 for i in range(vehicle_count)]
    for v in range(vehicle_count):
        for n in range(1,len(vehicle_tours_init[v])):
            vehicle_loads_init[v]+=customers[vehicle_tours_init[v][n]].demand

    while True:
        vehicle_loads=[v_l for v_l in vehicle_loads_init]
        vehicle_tours=[[n_c for n_c in vehicle_tours_init[v]] for v in range(vehicle_count)]
        repeat=False
        random.shuffle(iterator_customers)
        for n_c in iterator_customers:
            c=customers[n_c]
            cost_add_to_v=[0 for v in range(vehicle_count)]
            all_bad=True
            for v in range(vehicle_count):
                cost_add_to_v[v]=99999999999999999
                if (c.demand + vehicle_loads[v]) <= vehicle_capacity:
                    all_bad=False
                    for n in range(len(vehicle_tours[v])):
                        cost1=D[vehicle_tours[v][n], n_c]
                        cost2=D[n_c,vehicle_tours[v][(n+1)%len(vehicle_tours[v])]]
                        cost_add_to_v[v]=min(cost_add_to_v[v], cost1+cost2)
            if all_bad:
                repeat=True
                break

            chosen_v=np.argmin(cost_add_to_v)
            chosen_place=0
            c1=vehicle_tours[chosen_v][0]
            c2=vehicle_tours[chosen_v][(0+1) % len(vehicle_tours[chosen_v])]
            cost_place=D[c1, n_c] + D[n_c, c2]

            for n in range(1, len(vehicle_tours[chosen_v])):
                c1=vehicle_tours[chosen_v][n]
                c2=vehicle_tours[chosen_v][(n+1) % len(vehicle_tours[chosen_v])]
                if D[c1, n_c] + D[n_c, c2] < cost_place:
                    cost_place=D[c1, n_c] + D[n_c, c2]
                    chosen_place=n

            vehicle_tours[chosen_v].insert(chosen_place+1, n_c)
            vehicle_loads[chosen_v]+=customers[n_c].demand
        if not repeat:
            break
    obj=compute_obj(vehicle_tours, D)
    return vehicle_tours, obj


def generate_initial_solution(customers, vehicle_count, vehicle_capacity, D, N, stop):

    best_obj=9999999999999
    for i in range(N):
        while True:
            vehicle_tours=[[0] for v in range(vehicle_count)]
            vehicle_loads=[0 for v in range(vehicle_count)]
            repeat=False
            iterator_customers = list(range(1,len(customers)))
            random.shuffle(iterator_customers)
            for n_c in iterator_customers:
                c=customers[n_c]
                cost_add_to_v=[0 for v in range(vehicle_count)]
                all_bad=True
                for v in range(vehicle_count):
                    cost_add_to_v[v]=99999999999999999
                    if (c.demand + vehicle_loads[v]) <= vehicle_capacity:
                        all_bad=False
                        for n in range(len(vehicle_tours[v])):
                            cost1=D[vehicle_tours[v][n], n_c]
                            cost2=D[n_c,vehicle_tours[v][(n+1)%len(vehicle_tours[v])]]
                            cost_add_to_v[v]=min(cost_add_to_v[v], cost1+cost2)
                if all_bad:
                    repeat=True
                    break

                chosen_v=np.argmin(cost_add_to_v)
                chosen_place=0
                c1=vehicle_tours[chosen_v][0]
                c2=vehicle_tours[chosen_v][(0+1) % len(vehicle_tours[chosen_v])]
                cost_place=D[c1, n_c] + D[n_c, c2]

                for n in range(1, len(vehicle_tours[chosen_v])):
                    c1=vehicle_tours[chosen_v][n]
                    c2=vehicle_tours[chosen_v][(n+1) % len(vehicle_tours[chosen_v])]
                    if D[c1, n_c] + D[n_c, c2] < cost_place:
                        cost_place=D[c1, n_c] + D[n_c, c2]
                        chosen_place=n

                vehicle_tours[chosen_v].insert(chosen_place+1, n_c)
                vehicle_loads[chosen_v]+=customers[n_c].demand
            if not repeat:
                break
        obj=compute_obj(vehicle_tours, D)
        if obj < best_obj:
            best_obj=obj
            print(best_obj)
            best_tours=deepcopy(vehicle_tours)
            if obj < stop:
                return best_tours
    return best_tours

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

def init_solver(D, part_solution):
    # part_solution is a part of the solution with
    # len(part_solution) < len(solution).

    n=len(D)
    if not part_solution:
        node_degree=[0 for i in range(n)]
        connected=np.zeros(np.shape(D), dtype = bool)
        connect_list=dict()
        for i in range(n):
            connect_list[i]=[i]
    else:
        node_degree=[0 for i in range(n)]
        node_degree[part_solution[0]]=1
        node_degree[part_solution[-1]]=1
        for i in range(1, len(part_solution)-1):
            node_degree[part_solution[i]]=2
        connected=np.zeros(np.shape(D), dtype = bool)
        for v1 in part_solution:
            for v2 in part_solution:
                if (v1 != v2):
                    connected[v1, v2]=True
        connect_list=dict()
        for i in range(n):
            connect_list[i]=[i]

        for v in part_solution:
            connect_list[v]=part_solution

    D_flat=np.ndarray.flatten(D)
    sort_idxs=np.argsort(D_flat)
    edges=dict()
    for i in range(n):
        edges[i]=list()
    if part_solution:
        for i, v in enumerate(part_solution):
            if i == 0:
                edges[v].append(part_solution[i+1])
            elif i == len(part_solution)-1:
                edges[v].append(part_solution[i-1])
            else:
                edges[v].append(part_solution[i-1])
                edges[v].append(part_solution[i+1])

    j=0
    if part_solution:
        mm=len(part_solution)-1
    else:
        mm=0

    for qqq in range(n-1-mm):
        while True:
            sort_idx=sort_idxs[j]
            double_idx=np.unravel_index(sort_idx, np.shape(D))
            i1=double_idx[0]
            i2=double_idx[1]
            if (i1 == i2):
                continue

            if (node_degree[i1] == 2) or (node_degree[i2] == 2):
                j+=1
                continue
            if connected[i1, i2]:
                j+=1
                continue
            node_degree[i1]+=1
            node_degree[i2]+=1
            edges[i1].append(i2)
            edges[i2].append(i1)
            connected[i1, i2] = True
            connected[i2, i1] = True
            connect_list[i1].append(i2)
            connect_list[i2].append(i1)
            for i3 in range(n):
                if connected[i2, i3]:
                    for i4 in connect_list[i1]:
                        if not connected[i3, i4]:
                            connected[i4, i3] = True
                            connected[i3, i4] = True
                            connect_list[i3].append(i4)
                            connect_list[i4].append(i3)

                if connected[i1, i3]:
                    for i4 in connect_list[i2]:
                        if not connected[i3, i4]:
                            connected[i4, i3] = True
                            connected[i3, i4] = True
                            connect_list[i3].append(i4)
                            connect_list[i4].append(i3)
            j+=1
            break

    last2=list()
    for i in range(n):
        if len(edges[i]) == 1:
            last2.append(i)
    edges[last2[0]].append(last2[1])
    edges[last2[1]].append(last2[0])

    in_sol=[False for v in range(n)]
    obj=0
    solution=list()
    
    v1=0
    in_sol[v1]=True
    solution.append(v1)

    for i in range(n-1):
        for v2 in edges[v1]:
            if not in_sol[v2]:
                in_sol[v2]=True
                solution.append(v2)
                obj+=D[v1, v2]
                v1=v2
                break
    obj+=D[solution[0], solution[-1]]

    return solution, obj


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
    perc=0.5
    T_max=60

    D=compute_D(customers)
    if len(customers) == 16:
        stop=280
        N=10000
    elif len(customers) == 26:
        stop=630
        N=20000
    elif len(customers) == 51:
        stop=540
        N=20000
    elif len(customers) == 101:
        stop=830
        N=20000
    elif len(customers) == 200:
        stop=1400
        N=200
    elif len(customers) == 431:
        stop=2392
        N=200
    else:
        stop=9999999999
        N=100
    vehicle_tours=generate_initial_solution(customers, vehicle_count, vehicle_capacity, D, N, stop)
    T_tabu=60
    dt_dive=10
    dt_renew=10
    vehicle_tours, obj=find_sol(vehicle_tours, customers, vehicle_count, vehicle_capacity, D, perc, T_max, stop, T_tabu, dt_dive, dt_renew)
    #vehicle_tours=improve_initial_solution_tsp(customers, vehicle_tours, D)
    plot_route(vehicle_tours, customers)
    for v in range(vehicle_count):
        veh_load=0
        for i in range(1,len(vehicle_tours[v])):
            veh_load+=customers[vehicle_tours[v][i]].demand
    # checks that the number of customers served is correct
    #assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj=compute_obj(vehicle_tours, D)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += ' '.join([str(c) for c in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData

def plot_route(routes, customers):
    x = [customers[i][2] for i in range( len(customers))]
    y = [customers[i][3] for i in range( len(customers))]
    plt.scatter(x, y)
    for route in routes:
        x = [customers[0][2]] + [customers[i][2] for i in route] + [customers[0][2]]
        y = [customers[0][3]] + [customers[i][3] for i in route] + [customers[0][3]]       
        plt.plot( x, y)
    plt.show()
    return


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        solve_it(input_data)
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

