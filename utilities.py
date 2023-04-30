
from turtle import back
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import networkx as nx
import random
import itertools
import os
from tqdm.notebook import trange, tqdm
from time import sleep
from time import time 
from dwave_qbsolv import QBSolv

import math
from scipy.optimize import minimize, basinhopping


# qiskit part 
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile, assemble
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.circuit.library import TwoLocal
from qiskit import IBMQ


from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import NELDER_MEAD
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from docplex.mp.model import Model

#from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)


IBMQ.load_account()
IBMQ.providers()  

provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

# simulator
#ibmq_casablanca_sim = QasmSimulator.from_backend(provider.get_backend('ibmq_casablanca')) 
ibmq_jakarta_sim    = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta')) 
ibmq_guadalupe_sim  = QasmSimulator.from_backend(provider.get_backend('ibmq_guadalupe'))
#ibmq_qasm_simulator = Aer.get_backend('qasm_simulator')#provider.get_backend('ibmq_qasm_simulator') #Aer.get_backend('qasm_simulator')

ibmq_qasm_simulator = QasmSimulator(
    method='statevector', #matrix_product_state
    max_parallel_experiments = 0,

    )


# real device
#ibmq_casablanca = provider.get_backend('ibmq_casablanca')       
ibm_lagos       = provider.get_backend('ibm_lagos')                   
ibmq_jakarta    = provider.get_backend('ibmq_jakarta')
ibmq_guadalupe  = provider.get_backend('ibmq_guadalupe')


def reconstruct_ham_from_weight(weight, N):

    reg = list(range(N))
    terms = []; weights = []
    for qubit in reg:
        terms.append([qubit, ])
        weights.append(0)
    for q1, q2 in itertools.combinations(reg, 2):
        terms.append([q1, q2])
        weights.append(0)

    empty_ham = {tuple(term):weight for term,weight in zip(terms,weights)}
    
    index = 0 
    for term in empty_ham.items():
        empty_ham[term[0]] = weight[index]
        index += 1

    return empty_ham  

def graph_from_hamiltonian(hamiltonian):

    G = nx.Graph()
    for term,weight in hamiltonian.items():
        if(len(term)==1):
            G.add_node(term[0], weight=weight)
            #G.add_edge(term[0], term[0], weight=weight) #if you want the qubo matrix
        elif(len(term)==2):
            G.add_edge(term[0], term[1], weight=weight)
    return G

def complete_ham_form(ham):

    G = graph_from_hamiltonian(ham)
    reg = list(np.sort(nx.nodes(G)))
    terms = []; weights = []
    for qubit in reg:
        terms.append([qubit, ])
        weights.append(0)
    for q1, q2 in itertools.combinations(reg, 2):
        terms.append([q1, q2])
        weights.append(0)
    empty_ham = {tuple(term):weight for term,weight in zip(terms,weights)}
    for term in empty_ham.items():
        for term_ in ham.items():
            if term_[0] == term[0]:
                empty_ham[term[0]] = ham[term_[0]]

    ham = empty_ham

    return ham


def hamiltonian_from_graph(G):

    # Node bias terms
    bias_nodes = [*nx.get_node_attributes(G, 'weight')]
    biases = [*nx.get_node_attributes(G, 'weight').values()]
    hamiltonian = {(term,):weight for term,weight in zip(bias_nodes,biases)}
    # Edge terms
    hamiltonian_edge = {edge[:-1]:edge[-1].get('weight',1) for edge in G.edges(data=True)}
    hamiltonian.update(hamiltonian_edge)

    return hamiltonian



def random_k_regular_graph(degree: int,
                           nodes,
                           seed: int = None,
                           weighted: bool = False,
                           biases: bool = False):

    np.random.seed(seed=seed)
    # create a random regular graph on the nodes
    G = nx.random_regular_graph(degree, len(nodes), seed)
    nx.relabel_nodes(G, {i: n for i, n in enumerate(nodes)})
    for edge in G.edges():
        if not weighted:
            G[edge[0]][edge[1]]['weight'] = 1
        else:
            G[edge[0]][edge[1]]['weight'] = np.random.rand()
    if biases:
        for node in G.nodes():
            G.nodes[node]['weight'] = np.random.rand()

    return G

def portfolio_ham(mu, sigma, q, budget, penalty):

    ham = {}
    for i in range(len(mu)):
        ham[(i, )] = mu[i]*0.5 + budget*penalty -\
            len(mu)*penalty*0.5 -q*0.5*(np.sum(sigma, axis = 0))[i]    

    for i, j in itertools.combinations(list(range(len(mu))), 2):
        ham[(i, j)] = (q*sigma[i][j] + penalty)*0.5
        
    return ham 

def qubitOp_to_ham(qubitOp):

    ham = {}
    for i in qubitOp.to_dict()['paulis']:
        label_test = i['label']
        weight     = i['coeff']['real']
        c = 0
        index_list = []
        for s in label_test:
            if s == 'Z':
                index_list.append(c)
            c+=1
        index_list = len(label_test) - 1 - np.array(index_list)
        #print(weight)
        #print(index_list)
        if len(index_list) == 1:
            ham[(index_list[0], )] = weight
        elif len(index_list) == 2:
            #print(index_list)
            ham[(index_list[1], index_list[0])] = weight
    ham = complete_ham_form(ham)
    return ham


def energy_finder(ham, test_config):

    config = ((np.array(test_config)*2)-1)*-1

    terms = list(ham.keys())
    weights = list(ham.values())

    energy = 0

    for i, term in enumerate(terms):
        if len(term) == 1:
            energy += config[term[0]]*weights[i]
        elif len(term) == 2:
            energy += config[term[0]]*config[term[1]]*weights[i]

    return energy 


def exact_sol(ham): #Exact diagonalization 

    terms = list(ham.keys())
    weights = list(ham.values())
    G = graph_from_hamiltonian(ham)
    register = list(np.sort(nx.nodes(G)))

    diag = np.zeros((2**len(register)))
    for i, term in enumerate(terms):
        out = np.real(weights[i])
        for qubit in register:
            if qubit in term:
                out = np.kron([1, -1], out)
            else:
                out = np.kron([1, 1], out)
                
        diag += out
    energy = np.min(diag)
    indices = []
    for idx in range(len(diag)):
        if diag[idx] == energy:
            indices.append(idx)
    config_strings = [np.binary_repr(index, len(register))[::-1] for index in indices]
    configs = [np.array([int(x) for x in config_str]) for config_str in config_strings]

    return configs, energy

def exact_sol_les(ham, N_les): #Exact diagonalization 

    terms = list(ham.keys())
    weights = list(ham.values())
    G = graph_from_hamiltonian(ham)
    register = list(np.sort(nx.nodes(G)))

    diag = np.zeros((2**len(register)))
    for i, term in enumerate(terms):
        out = np.real(weights[i])
        for qubit in register:
            if qubit in term:
                out = np.kron([1, -1], out)
            else:
                out = np.kron([1, 1], out)
                
        diag += out

    energy_list = np.sort(diag)[:N_les]
    config_list = []
    for energy in energy_list:
        indices = []
        for idx in range(len(diag)):
            if diag[idx] == energy:
                indices.append(idx)
        config_strings = [np.binary_repr(index, len(register))[::-1] for index in indices]
        configs = [np.array([int(x) for x in config_str]) for config_str in config_strings]
        config_list.append(configs)

    return config_list, energy_list



def Dwave_sol(ham): # classical tabu solver

    n_qubit_ = 0
    for i_, term_ in enumerate(list(ham.keys())):
        if len(term_) == 1:
            n_qubit_+=1 
    
    """
    h = {}
    J = {}
    for h_i in range(n_qubit_):
        h[(h_i)] = ham[(h_i, )]
    
    for i, j in itertools.combinations(list(range(n_qubit_)), 2):
        if (i,j) in ham:
            J[(i,j)] = ham[(i,j)]
    """

    # This is fine for fully connected graph
    index_couple_list = itertools.combinations(list(range(n_qubit_)), 2)
    h = dict(zip(list(range(n_qubit_)),list(ham.values())[:n_qubit_]))
    J = dict(zip(index_couple_list, list(ham.values())[n_qubit_:]))
    
    response = QBSolv().sample_ising(h, J)

    config = []
    for index, item in list(response.samples())[0].items():
        config.append(item)
        
    return [(np.array(config)*-1 +1)/2], list(response.data_vectors['energy'])[0]

def weight_to_qp(rand_weight, n_qubit):

    h_coeff      = rand_weight[:n_qubit] #[::-1] #np.full(n_qubit, 1) 
    J_spin_coeff = np.zeros((n_qubit, n_qubit))
    J_spin_coeff_ = rand_weight[n_qubit:] #[::-1] #np.full((len(rand_weight[n_qubit:])), 1)
    if np.sum(J_spin_coeff_) != 0:

        c = 0
        for i in range(n_qubit):
            for j in range(n_qubit):
                if i != j :
                    if J_spin_coeff[i][j] == 0 and J_spin_coeff[j][i] == 0:
                        J_spin_coeff[i][j] += J_spin_coeff_[c]
                        #print("c = ", c)
                        c += 1
        
        J = np.zeros((n_qubit, n_qubit))
        for i in range(n_qubit):
            for j in range(n_qubit):
                J[i][j] += J_spin_coeff[i][j]*2
                J[j][i] += J_spin_coeff[i][j]*2

        J_elim = np.sum(J, axis=0)*-1

    elif np.sum(J_spin_coeff_) == 0:
        J = np.zeros((n_qubit, n_qubit))
        J_elim = np.sum(J, axis=0)*-1


    h = h_coeff*-2
    mdl = Model()
    z = [mdl.binary_var() for i in range(n_qubit)]
    objective = \
        mdl.sum(J[i, j] * z[i] * z[j] for i in range(n_qubit)for j in range(n_qubit)) + \
            mdl.sum(J_elim[k]*z[k] for k in range(n_qubit)) + \
                mdl.sum(h[l]*z[l] for l in range(n_qubit))
    mdl.minimize(objective)
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    
    """
    # convert to Ising Hamiltonian
    qubitOp, offset = qp.to_ising()
    print('Ising Hamiltonian:')
    #print(qubitOp.print_details())
    print(qubitOp)
    """
    return qp

def weight_to_qubitOP(rand_weight, n_qubit):

    h_coeff      = rand_weight[:n_qubit] #[::-1] #np.full(n_qubit, 1) 
    J_spin_coeff = np.zeros((n_qubit, n_qubit))
    J_spin_coeff_ = rand_weight[n_qubit:] #[::-1] #np.full((len(rand_weight[n_qubit:])), 1)
    if np.sum(J_spin_coeff_) != 0:

        c = 0
        for i in range(n_qubit):
            for j in range(n_qubit):
                if i != j :
                    if J_spin_coeff[i][j] == 0 and J_spin_coeff[j][i] == 0:
                        J_spin_coeff[i][j] += J_spin_coeff_[c]
                        #print("c = ", c)
                        c += 1
        
        J = np.zeros((n_qubit, n_qubit))
        for i in range(n_qubit):
            for j in range(n_qubit):
                J[i][j] += J_spin_coeff[i][j]*2
                J[j][i] += J_spin_coeff[i][j]*2

        J_elim = np.sum(J, axis=0)*-1

    elif np.sum(J_spin_coeff_) == 0:
        J = np.zeros((n_qubit, n_qubit))
        J_elim = np.sum(J, axis=0)*-1

    h = h_coeff*-2
    mdl = Model()
    z = [mdl.binary_var() for i in range(n_qubit)]
    objective = \
        mdl.sum(J[i, j] * z[i] * z[j] for i in range(n_qubit)for j in range(n_qubit)) + \
            mdl.sum(J_elim[k]*z[k] for k in range(n_qubit)) + \
                mdl.sum(h[l]*z[l] for l in range(n_qubit))
    mdl.minimize(objective)
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    
    # convert to Ising Hamiltonian
    qubitOp, offset = qp.to_ising()
    
    return qubitOp
#QAOA

def QAOA_calculate(backend, ham, optimizer_maxiter = 5, shots = 1024, print_eval = True, error_mitigation = False):

    n_qubit_ = 0
    for i_, term_ in enumerate(list(ham.keys())):
        if len(term_) == 1:
            n_qubit_+=1 

    #index_couple_list = []
    index_couple_list = list(itertools.combinations(list(range(n_qubit_)), 2))

    h_reindex = {}
    #for i in range(n_qubit_):
    #    for j in range(n_qubit_):
    #        if j > i:
    #            index_couple_list.append((i, j))

    for index, item in enumerate(ham.items()):
        if len(item[0]) == 1: #bias term
            h_reindex[(index,)] = item[1]
        if len(item[0]) == 2: #coupling term
            h_reindex[index_couple_list[index-n_qubit_]] = item[1]
    
    weights = list(h_reindex.values())

    qp_ = weight_to_qp(np.array(weights), n_qubit=n_qubit_)
    
    shot=shots
    if error_mitigation == True:
        quantum_instance = QuantumInstance(backend, shot, measurement_error_mitigation_cls=CompleteMeasFitter,
                            cals_matrix_refresh_period=30) # modify this to do error mitigation
    elif error_mitigation == False:
        quantum_instance = QuantumInstance(backend, shot) # modify this to do error mitigation
    #optimizer = NELDER_MEAD(maxiter=optimizer_maxiter, maxfev=) # SPSA, L_BFGS_B, QNSPSA, NELDER_MEAD
    optimizer = NELDER_MEAD(maxfev=optimizer_maxiter)

    def store_intermediate_result(eval_count, parameters, mean, std):
        if print_eval == True:
            print("eval_count :", eval_count)

    qaoa_mes = QAOA(optimizer = optimizer, reps = 1, quantum_instance = quantum_instance, initial_point=[0., 1.],
                    callback=store_intermediate_result)

    qaoa = MinimumEigenOptimizer(qaoa_mes)
    result = qaoa.solve(qp_)

    return [np.array([int(i) for i in result.x])], energy_finder(h_reindex, list(result.x))




def LSSA___(ham,
        total_sample_time = 1,
        group_size = 7,
        approximate_ratio = True,
        visible = False,
        method = "Random"
        ):
    
    G = graph_from_hamiltonian(ham)
    n_qubit = len(np.sort(nx.nodes(G)))

    group_num = int(n_qubit/group_size)

    # randomly group_size qubits sub-system
    sol_count = list(np.zeros(n_qubit))
    
    if visible == False:
        range_ = range(total_sample_time)
    elif visible == True: 
        range_ = trange(total_sample_time)

    if method == "Random":      

        for sample_time in range_:

            group = random.sample(list(range(n_qubit)), int(n_qubit))

            group_list   = [group[int(group_size)*group_num_index:+int(group_size)*(group_num_index+1)] for group_num_index in range(group_num)]
            ham_sub_list = [{} for group_num_index in range(group_num)]

            for group_num_index in range(group_num):
                for term, weight in ham.items() : 
                    if len(term) == 1:
                        if term[0] in group_list[group_num_index] : 
                            ham_sub_list[group_num_index][term] = weight
                    elif len(term) == 2:
                        if term[0] in group_list[group_num_index] and term[1] in group_list[group_num_index] :
                            ham_sub_list[group_num_index][term] = weight
                            
                exact_sol_sub_ = exact_sol(ham_sub_list[group_num_index])  ##########

                for count_index, index in enumerate(np.sort(group_list[group_num_index])):
                    sol_count[index] += ((exact_sol_sub_[0][0]*2)-1)[count_index]*exact_sol_sub_[1]*(-1)

        repres = (np.sign(np.array(sol_count)) + 1 )/2
        repres = [int(i) for i in repres]

        if approximate_ratio == True: 
            return energy_finder(ham, repres)/exact_sol(ham)[1]

        elif approximate_ratio == False:
            return energy_finder(ham, repres)


# Define LSSA function 
"""
def LSSA(ham,
        total_sample_time = 1,
        group_size = 5,
        approximate_ratio = True,
        visible = False,
        method = "Random",
        count_method = "avg",
        plot_sol_count ="False",
        solver = "dwave-tabu",
        backend = ibmq_qasm_simulator,
        shots    = 8192,
        optimizer_maxiter = 5
        ):
    
    bias_list = []
    for i, term in enumerate(list(ham.keys())):
        if len(term) == 1:
            bias_list.append(term[0])
        elif len(term) != 1:
            break;
    n_qubit = max(bias_list)+1 
    #print(n_qubit)
    del bias_list

    group_num = math.ceil(n_qubit/group_size)

    # randomly group_size qubits sub-system
    sol_count = list(np.zeros(n_qubit))
    
    if visible == True: 
        range_ = trange(total_sample_time)
    elif visible == False:
        range_ = range(total_sample_time)

    #if method == "Random":        
    amplitude_list   = []
    config_list      = []
    group_index_list = []

    for sample_time in range_:
        #group = random.sample(list(range(n_qubit)), int(n_qubit)) # this will break down if a subgroup has only one variable
        group = random.sample(list(range(n_qubit)), int(n_qubit))+random.sample(list(range(n_qubit)), int(group_size*math.ceil(n_qubit/group_size) - n_qubit))
        group_count  = 0
        group_list   = []
        ham_sub_list = []

        for group_num_index in range(group_num):
            
            group_sub = group[int(group_size)*group_num_index:+int(group_size)*(group_num_index+1)]
            group_count+=1 
            group_list.append(group_sub)
            group_index_list.append(group_sub)
            ham_sub_list.append({})

        for group_num_index in range(group_num):
            if visible == True:
                print("group_num_index = ", group_num_index)

            for term, weight in ham.items() : 
                if len(term) == 1:
                    if term[0] in group_list[group_num_index] : 
                        ham_sub_list[group_num_index][term] = weight

                elif len(term) == 2:
                    if term[0] in group_list[group_num_index] and term[1] in group_list[group_num_index] :
                        ham_sub_list[group_num_index][term] = weight

            if solver == "dwave-tabu":
                exact_sol_sub_ = Dwave_sol(ham_sub_list[group_num_index]) 
            elif solver == "exact":
                exact_sol_sub_ = exact_sol(ham_sub_list[group_num_index])
            elif solver == "qaoa":
                exact_sol_sub_ = QAOA_calculate(
                    backend = backend,
                    ham = ham_sub_list[group_num_index],
                    optimizer_maxiter = optimizer_maxiter,
                    shots = shots,
                    print_eval = True
                )

            amplitude_list.append(exact_sol_sub_[1])
            config_list.append(exact_sol_sub_[0][0])

            for count_index, index in enumerate(np.sort(group_list[group_num_index])):
                if count_index < len(exact_sol_sub_[0][0]): # for subsystem size < group size (repeated index)
                    sol_count[index] += ((exact_sol_sub_[0][0]*2)-1)[count_index]*exact_sol_sub_[1]*(-1)   
                         
    if plot_sol_count == True:
        plt.bar(list(range(n_qubit)), sol_count, color=plt.cm.viridis(0.4))
        sol_count_avg = np.average(sol_count)
        sol_count_std = np.std(sol_count)
        print("sol_count_avg = ",sol_count_avg)
        print("sol_count_std = ", sol_count_std)
        extreme_value_count = 0
        for spin in sol_count:
            if spin > sol_count_avg + sol_count_std:
                #print(spin)
                extreme_value_count += 1
            if spin < sol_count_avg - sol_count_std:
                extreme_value_count += 1
        print( (extreme_value_count/n_qubit)*100, "% of spins are beyond 1 std" )
        plt.axhline(y = sol_count_avg, color = 'r', linestyle = '-', label = "avg")
        plt.fill_between(list(range(n_qubit)), sol_count_avg + sol_count_std, sol_count_avg - sol_count_std, color=plt.cm.viridis(0.4), alpha=0.2)
        plt.legend()
        plt.show()

    if count_method == "abs":
        repres = (np.sign(np.array(sol_count)) + 1 )/2
        repres = [int(i) for i in repres]
    
    if count_method == "avg":
        sol_count_avg = np.average(sol_count)
        sol_count = np.array(sol_count) - sol_count_avg
        repres = (np.sign(np.array(sol_count)) + 1 )/2

    if approximate_ratio == True: 
        if solver == "dwave-tabu" or solver == "qaoa":
            return energy_finder(ham, repres)/Dwave_sol(ham)[1], amplitude_list, config_list, group_index_list, repres
        elif solver == "exact":
            return energy_finder(ham, repres)/exact_sol(ham)[1], amplitude_list, config_list, group_index_list, repres

    elif approximate_ratio == False:
        return energy_finder(ham, repres), amplitude_list, config_list, group_index_list, repres
"""

def LSSA(ham,
        total_sample_time = 1,
        group_size = 5,
        approximate_ratio = True,
        visible = False,
        count_method = "avg",
        plot_sol_count ="False",
        solver = "dwave-tabu",
        backend = ibmq_qasm_simulator,
        shots    = 8192,
        optimizer_maxiter = 5,
        group_method = "random", # "random", "weak_weight",
        error_mitigation = False
        ):
    
    bias_list = []
    for i, term in enumerate(list(ham.keys())):
        if len(term) == 1:
            bias_list.append(term[0])
        elif len(term) != 1:
            break;
    n_qubit = max(bias_list)+1 
    #print(n_qubit)
    del bias_list

    group_num = math.ceil(n_qubit/group_size)

    # randomly group_size qubits sub-system
    sol_count = list(np.zeros(n_qubit))
    
    if visible == True: 
        range_ = trange(total_sample_time)
    elif visible == False:
        range_ = range(total_sample_time)

    #if method == "Random":        
    amplitude_list   = []
    config_list      = []
    group_index_list = []

    for sample_time in range_:
        
        if group_method == "random":
            #group = random.sample(list(range(n_qubit)), int(n_qubit))
            group = random.sample(list(range(n_qubit)), int(n_qubit))+random.sample(list(range(n_qubit)), int(group_size*math.ceil(n_qubit/group_size) - n_qubit))
            group_count  = 0
            group_list   = []
            ham_sub_list = []

            for group_num_index in range(group_num):
                
                group_sub = group[int(group_size)*group_num_index:+int(group_size)*(group_num_index+1)]
                group_count+=1 
                group_list.append(group_sub)
                group_index_list.append(group_sub)
                ham_sub_list.append({})

        for group_num_index in range(group_num):
            if visible == True:
                print("group_num_index = ", group_num_index)

            for term, weight in ham.items() : 
                if len(term) == 1:
                    if term[0] in group_list[group_num_index] : 
                        ham_sub_list[group_num_index][term] = weight

                elif len(term) == 2:
                    if term[0] in group_list[group_num_index] and term[1] in group_list[group_num_index] :
                        ham_sub_list[group_num_index][term] = weight

            if solver == "dwave-tabu":
                exact_sol_sub_ = Dwave_sol(ham_sub_list[group_num_index]) 
            elif solver == "exact":
                exact_sol_sub_ = exact_sol(ham_sub_list[group_num_index])
            elif solver == "qaoa":
                exact_sol_sub_ = QAOA_calculate(
                    backend = backend,
                    ham = ham_sub_list[group_num_index],
                    optimizer_maxiter = optimizer_maxiter,
                    shots = shots,
                    print_eval = True,
                    error_mitigation = error_mitigation
                )

            amplitude_list.append(exact_sol_sub_[1])
            config_list.append(exact_sol_sub_[0][0])

            for count_index, index in enumerate(np.sort(group_list[group_num_index])):
                if count_index < len(exact_sol_sub_[0][0]): # for subsystem size < group size (repeated index)
                    sol_count[index] += ((exact_sol_sub_[0][0]*2)-1)[count_index]*exact_sol_sub_[1]*(-1)   
                         

    if plot_sol_count == True:
        plt.bar(list(range(n_qubit)), sol_count, color=plt.cm.viridis(0.4))
        sol_count_avg = np.average(sol_count)
        sol_count_std = np.std(sol_count)
        print("sol_count_avg = ",sol_count_avg)
        print("sol_count_std = ", sol_count_std)
        extreme_value_count = 0
        for spin in sol_count:
            if spin > sol_count_avg + sol_count_std:
                #print(spin)
                extreme_value_count += 1
            if spin < sol_count_avg - sol_count_std:
                extreme_value_count += 1
        print( (extreme_value_count/n_qubit)*100, "% of spins are beyond 1 std" )
        plt.axhline(y = sol_count_avg, color = 'r', linestyle = '-', label = "avg")
        plt.fill_between(list(range(n_qubit)), sol_count_avg + sol_count_std, sol_count_avg - sol_count_std, color=plt.cm.viridis(0.4), alpha=0.2)
        plt.legend()
        plt.show()

    if count_method == "abs":
        repres = (np.sign(np.array(sol_count)) + 1 )/2
        repres = [int(i) for i in repres]
    
    if count_method == "avg":
        sol_count_avg = np.average(sol_count)
        sol_count = np.array(sol_count) - sol_count_avg
        repres = (np.sign(np.array(sol_count)) + 1 )/2

    if approximate_ratio == True: 
        if solver == "dwave-tabu" or solver == "qaoa":
            return energy_finder(ham, repres)/Dwave_sol(ham)[1], amplitude_list, config_list, group_index_list, repres
        elif solver == "exact":
            return energy_finder(ham, repres)/exact_sol(ham)[1], amplitude_list, config_list, group_index_list, repres

    elif approximate_ratio == False:
        return energy_finder(ham, repres), amplitude_list, config_list, group_index_list, repres


def given_energy_get_config(ham, energy): 
    # Exact diagonalization (Don't try too large)

    terms = list(ham.keys())
    weights = list(ham.values())
    G = graph_from_hamiltonian(ham)
    register = list(np.sort(nx.nodes(G)))

    diag = np.zeros((2**len(register)))
    for i, term in enumerate(terms):
        out = np.real(weights[i])
        for qubit in register:
            if qubit in term:
                out = np.kron([1, -1], out)
            else:
                out = np.kron([1, 1], out)
                
        diag += out

    indices = []
    for idx in range(len(diag)):
        if diag[idx] == energy:
            indices.append(idx)
    config_strings = [np.binary_repr(index, len(register))[::-1] for index in indices]
    configs = [np.array([int(x) for x in config_str]) for config_str in config_strings]

    return configs


def vqe_ansatz(
    qubits,circ,
    parameters,
    layers = 2,
    entangler = "full" # "full" or "linear"
    ):

    for layer in range(layers):

        for iz in range (0, len(qubits)):
            circ.ry(parameters[layer][iz], qubits[iz])
            circ.rz(parameters[layer][iz+len(qubits)], qubits[iz])

        circ.barrier()
        if entangler == "full":
            for i, j in itertools.combinations(list(range(len(qubits))), 2):
                circ.cz(qubits[i], qubits[j])

        elif entangler == "linear":
            for i in range(len(qubits)-1):
                circ.cz(qubits[i], qubits[i+1])

        circ.barrier()

    for iz in range (0, len(qubits)):
        circ.ry(parameters[layers][iz], qubits[iz])
        circ.rz(parameters[layers][iz+len(qubits)], qubits[iz])

    circ.measure_all()

# Define LSSA_AOQ function 

def LSSA_AOQ(
            ham,
            total_sample_time=1,
            group_size = 7,
            count_method="abs",
            plot_sol_count=False,
            print_eval_num = True,
            solver = "dwave-tabu",
            backend = ibmq_qasm_simulator, # only takes effect if solver = "qaoa"
            vqe_layer = 2,
            vqe_entangler = "full",
            shots = 8192,
            shots_sub = 8192,  # only takes effect if solver = "qaoa"
            max_iter_sub = 5,   # only takes effect if solver = "qaoa"
            group_method = "random",
            approximate_ratio = True,
            error_mitigation = False
            ):

    print("stage 1")
    result_ = LSSA(
                ham,
                total_sample_time=total_sample_time,
                group_size = group_size,
                approximate_ratio = False, # this should be False to do the energy minimization by amplitude optimization
                visible = False,
                count_method="abs",
                plot_sol_count=False,
                solver = solver,
                backend = backend,
                shots    = shots_sub,
                optimizer_maxiter = max_iter_sub,
                group_method = group_method, # "random", "weak_weight"
                error_mitigation = error_mitigation
                )
                
    n_qubit = len(result_[4]) # count the qubit number
    count_method = "abs" # "avg", "abs"

    def amplitude_optimize(x):
        """
        Input  : Amplitude of subgroups 
        Output : Corresponding energy of full system 
        """
        x = np.array(x)
        x = abs(x)
        sol_count = list(np.zeros(n_qubit))
        config_list = 2*np.array(result_[2], dtype=object) - 1
        group_index_list = result_[3]

        for ind_, group_list in enumerate(group_index_list):
            for count_index, index in enumerate(np.sort(group_list)):
                if count_index < len(config_list[ind_]): # for subsystem size < group size (repeated index)
                    sol_count[index] += config_list[ind_][count_index]*x[ind_]
        """
        if count_method == "avg":
            # average method
            sol_count_avg = np.average(sol_count)
            sol_count = np.array(sol_count) - sol_count_avg
            config_full = (np.sign(np.array(sol_count)) + 1 )/2
            config_full = [int(i) for i in config_full]
        """
        if count_method == "abs":
            config_full = (np.sign(np.array(sol_count)) + 1 )/2
            config_full = [int(i) for i in config_full]
            #print(config_full)

        return energy_finder(ham, config_full), config_full

    # Implements the entire cost function on the quantum circuit

    global eval_num
    global backend_
    global shots_

    shots_   = shots
    backend_ = backend
    eval_num = 0

    def calculate_cost_function(parameters):
        
        global eval_num
        global backend_
        global shots_ 

        parameters = np.reshape(parameters, (layers+1,n_qubit_ao*2))

        circ = QuantumCircuit(
            QuantumRegister(n_qubit_ao)
            )

        backend = backend_

        vqe_ansatz(
            qubits     = list(range(n_qubit_ao)),
            circ       = circ, 
            parameters = parameters,
            layers     = layers,
            entangler  = vqe_entangler
            )

        t_circ = transpile(circ, backend)
        job = backend.run(t_circ, shots = shots_) 

        eval_num += 1
        
        if print_eval_num == True:
            print("eval", eval_num)
        
        result = job.result()
        count_ = result.get_counts()

        #print("count_ (unmit) = ", count_)
        
        if error_mitigation == True and n_qubit_ao <= 8:

            # Generate the calibration circuits
            qr = qiskit.QuantumRegister(n_qubit_ao)
            qubit_list = list(range(n_qubit_ao))

            meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')

            print("state_labels = ", state_labels)
            print("result_last = ", result)
            # Execute the calibration circuits
            job_mit = qiskit.execute(meas_calibs, backend=backend, shots=8192)
            cal_results = job_mit.result()
            meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')

            # Results without mitigation
            # count_

            # Get the filter object
            meas_filter = meas_fitter.filter

            # Results with mitigation
            mitigated_results = meas_filter.apply(result)
            mitigated_counts = mitigated_results.get_counts(0)
            count_ = mitigated_counts
            #print("count_ = ", count_)
        


        probs  = {}
        space  = " ".ljust(n_qubit_ao+1, "0")

        #for output in [format(i, '0'+str(n_qubit_ao)+'b')+space for i in range(2**n_qubit_ao)]:
        for output in [format(i, '0'+str(n_qubit_ao)+'b') for i in range(2**n_qubit_ao)]:
        
            if output in count_:
                probs[output] = count_[output]/shots_
            else:
                probs[output] = 0
        
        prob = [p for basis, p in probs.items()]

        amplitude_optimize_result = amplitude_optimize(prob)
        cost_value = amplitude_optimize_result[0]
        energy_list.append(cost_value)
        config_list.append(amplitude_optimize_result[1])
        #print(cost_value)
        return cost_value

    energy_list = []
    config_list = [] 
    layers = vqe_layer

    #n_qubit_ao = int(np.log2(int((n_qubit/group_size))*total_sample_time))+1
    n_qubit_ao = math.ceil(np.log2(math.ceil((n_qubit/group_size))*total_sample_time))

    print("n_qubit_ao = ",n_qubit_ao)
    out = minimize(calculate_cost_function, x0=np.random.rand((layers+1) * (n_qubit_ao*2)), method="COBYLA", options={'maxiter':200},
                callback = None) 

    if approximate_ratio == True:

        if solver == "dwave-tabu" or solver == "qaoa":
            return result_[0], min(energy_list)/Dwave_sol(ham)[1], config_list[-1], energy_list
        elif solver == "exact":
            return result_[0], min(energy_list)/exact_sol(ham)[1], config_list[-1], energy_list

    elif approximate_ratio == False:

        if solver == "dwave-tabu" or solver == "qaoa":
            return result_[0], min(energy_list), config_list[-1], energy_list
        elif solver == "exact":
            return result_[0], min(energy_list), config_list[-1], energy_list

# Define LSSA_L2 function 

def LSSA_L2(
        ham,
        total_sample_time = 1,
        group_size = 10,
        approximate_ratio = True,
        visible = False,
        method = "Random",
        count_method = "avg",
        plot_sol_count ="False",
        solver = "LSSA",
        backend = ibmq_qasm_simulator,
        L2_group_size = 5
        ):

    print("LSSA_L2")
    bias_list = []
    for i, term in enumerate(list(ham.keys())):
        if len(term) == 1:
            bias_list.append(term[0])
        elif len(term) != 1:
            break;
    n_qubit = max(bias_list)+1 
    del bias_list
    group_num = math.ceil(n_qubit/group_size)
    # randomly group_size qubits sub-system
    sol_count = list(np.zeros(n_qubit))
    if visible == True: 
        range_ = trange(total_sample_time)
    elif visible == False:
        range_ = range(total_sample_time)
    amplitude_list   = []
    config_list      = []
    group_index_list = []
    for sample_time in range_:
        group = random.sample(list(range(n_qubit)), int(n_qubit))
        group_count  = 0
        group_list   = []
        ham_sub_list = []
        for group_num_index in range(group_num):
            group_sub = group[int(group_size)*group_num_index:+int(group_size)*(group_num_index+1)]
            group_count+=1 
            group_list.append(group_sub)
            group_index_list.append(group_sub)
            ham_sub_list.append({})
        for group_num_index in range(group_num):
            if visible == True:
                print("group_num_index = ", group_num_index)
            for term, weight in ham.items() : 
                if len(term) == 1:
                    if term[0] in group_list[group_num_index] : 
                        ham_sub_list[group_num_index][term] = weight
                elif len(term) == 2:
                    if term[0] in group_list[group_num_index] and term[1] in group_list[group_num_index] :
                        ham_sub_list[group_num_index][term] = weight
            
            #print("ham_sub_list[group_num_index] : ", ham_sub_list[group_num_index])
            ham_sub = ham_sub_list[group_num_index]
            n_qubit_ = 0
            for i_, term_ in enumerate(list(ham_sub.keys())):
                if len(term_) == 1:
                    n_qubit_+=1 

            index_couple_list = list(itertools.combinations(list(range(n_qubit_)), 2))
            ham_sub_ = dict(zip([(i,) for i in range(n_qubit_)],list(ham_sub.values())[:n_qubit_]))
            J = dict(zip(index_couple_list, list( ham_sub.values())[n_qubit_:]))
            ham_sub_.update(J)
            #print("ham_sub_ : ", ham_sub_)
            
            if solver == "LSSA":
                 exact_sol_sub_ = LSSA_AOQ(
                                ham = ham_sub_, # subsystem hamiltonian with size "group_size"
                                total_sample_time=1,
                                group_size = L2_group_size,
                                count_method="abs",
                                plot_sol_count=False,
                                print_eval_num = True,
                                solver = "dwave-tabu",
                                backend = ibmq_qasm_simulator,
                                vqe_layer = 2,
                                vqe_entangler = "full"
                 )
            amplitude_list.append(exact_sol_sub_[0])
            config_list.append(np.array(exact_sol_sub_[2]))
            for count_index, index in enumerate(np.sort(group_list[group_num_index])):
                sol_count[index] += ((np.array(exact_sol_sub_[2])*2)-1)[count_index]*exact_sol_sub_[0]*(-1)  
                 
    if plot_sol_count == True:
        plt.bar(list(range(n_qubit)), sol_count, color=plt.cm.viridis(0.4))
        sol_count_avg = np.average(sol_count)
        sol_count_std = np.std(sol_count)
        print("sol_count_avg = ",sol_count_avg)
        print("sol_count_std = ", sol_count_std)
        extreme_value_count = 0
        for spin in sol_count:
            if spin > sol_count_avg + sol_count_std:
                #print(spin)
                extreme_value_count += 1
            if spin < sol_count_avg - sol_count_std:
                extreme_value_count += 1
        print( (extreme_value_count/n_qubit)*100, "% of spins are beyond 1 std" )
        plt.axhline(y = sol_count_avg, color = 'r', linestyle = '-', label = "avg")
        plt.fill_between(list(range(n_qubit)), sol_count_avg + sol_count_std, sol_count_avg - sol_count_std, color=plt.cm.viridis(0.4), alpha=0.2)
        plt.legend()
        plt.show()
    if count_method == "abs":
        repres = (np.sign(np.array(sol_count)) + 1 )/2
        repres = [int(i) for i in repres]
    if count_method == "avg":
        sol_count_avg = np.average(sol_count)
        sol_count = np.array(sol_count) - sol_count_avg
        repres = (np.sign(np.array(sol_count)) + 1 )/2
    if approximate_ratio == True: 
        if solver == "dwave-tabu" or solver == "qaoa":
            return energy_finder(ham, repres)/Dwave_sol(ham)[1], amplitude_list, config_list, group_index_list, repres
        elif solver == "exact":
            return energy_finder(ham, repres)/exact_sol(ham)[1], amplitude_list, config_list, group_index_list, repres
    elif approximate_ratio == False:
        return energy_finder(ham, repres), amplitude_list, config_list, group_index_list, repres

def LSSA_AOQ_L2(
            ham,
            total_sample_time=1,
            group_size = 10,
            count_method="abs",
            plot_sol_count=False,
            print_eval_num = True,
            solver = "dwave-tabu",
            backend = ibmq_qasm_simulator,
            vqe_layer = 2,
            vqe_entangler = "full", 
            L2_group_size = 5,
            ):
            
    result_ = LSSA_L2(
                ham = ham,
                total_sample_time = total_sample_time,
                group_size = group_size,
                approximate_ratio = False,
                visible = False,
                count_method = "avg",
                plot_sol_count ="False",
                solver = "LSSA",
                backend = backend,
                L2_group_size = L2_group_size
                )
    #print(result_)

    n_qubit = len(result_[4]) # count the qubit number
    count_method = "abs" # "avg", "abs"

    def amplitude_optimize(x):
        """
        Input  : Amplitude of subgroups 
        Output : Corresponding energy of full system 
        """
        x = np.array(x)
        x = abs(x)
        sol_count = list(np.zeros(n_qubit))
        config_list = 2*np.array(result_[2], dtype=object) - 1
        group_index_list = result_[3]

        for ind_, group_list in enumerate(group_index_list):
            for count_index, index in enumerate(np.sort(group_list)):
                sol_count[index] += config_list[ind_][count_index]*x[ind_]
        """
        if count_method == "avg":
            # average method
            sol_count_avg = np.average(sol_count)
            sol_count = np.array(sol_count) - sol_count_avg
            config_full = (np.sign(np.array(sol_count)) + 1 )/2
            config_full = [int(i) for i in config_full]
        """
        if count_method == "abs":
            config_full = (np.sign(np.array(sol_count)) + 1 )/2
            config_full = [int(i) for i in config_full]
            #print(config_full)

        return energy_finder(ham, config_full), config_full

    # Implements the entire cost function on the quantum circuit

    global eval_num
    global backend_

    backend_ = backend
    eval_num = 0

    def calculate_cost_function(parameters):
        
        global eval_num
        global backend_

        parameters = np.reshape(parameters, (layers+1,n_qubit_ao*2))

        circ = QuantumCircuit(
            QuantumRegister(n_qubit_ao),
            ClassicalRegister(n_qubit_ao)
            )

        backend = backend_

        vqe_ansatz(
            qubits     = list(range(n_qubit_ao)),
            circ       = circ, 
            parameters = parameters,
            layers     = layers,
            entangler  = vqe_entangler
            )
        
        t_circ = transpile(circ, backend)
        job = backend.run(t_circ, shots = 8192) 

        eval_num += 1
        
        if print_eval_num == True:
            print("eval", eval_num)
        
        result = job.result()
        count_ = result.get_counts()


        probs  = {}
        space  = " ".ljust(n_qubit_ao+1, "0")

        for output in [format(i, '0'+str(n_qubit_ao)+'b')+space for i in range(2**n_qubit_ao)]:

            if output in count_:
                probs[output] = count_[output]/8192
            else:
                probs[output] = 0
        
        prob = [p for basis, p in probs.items()]

        amplitude_optimize_result = amplitude_optimize(prob)
        cost_value = amplitude_optimize_result[0]
        energy_list.append(cost_value)
        config_list.append(amplitude_optimize_result[1])
        #print(cost_value)
        return cost_value

    energy_list = []
    config_list = [] 
    layers = vqe_layer

    #n_qubit_ao = int(np.log2(int((n_qubit/group_size))*total_sample_time))+1
    n_qubit_ao = math.ceil(np.log2(int((n_qubit/group_size))*total_sample_time))

    print("n_qubit_ao = ",n_qubit_ao)
    out = minimize(calculate_cost_function, x0=np.random.rand((layers+1) * (n_qubit_ao*2)), method="COBYLA", options={'maxiter':200},
                callback = None)

    if solver == "dwave-tabu" or solver == "qaoa" or solver == "LSSA":
        return result_[0], min(energy_list)/Dwave_sol(ham)[1], config_list[-1], energy_list
    elif solver == "exact":
        return result_[0], min(energy_list)/exact_sol(ham)[1], config_list[-1], energy_list

