import os
from read_epanet import print_water_network_info, fetch_epanet_data
from operational_pyomo import PyomoOperationalModel
from operational_gurobi import GurobiOperationalModel
from operational_mosek import MosekOperationalModel

def write_results(epanet_model, method, problem_status, lower_bound, upper_bound, solution_time):
    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results.txt')
    with open(results_file, 'a') as f:
        f.write(f'{epanet_model} | {method} | status: {problem_status} | lower bound: {lower_bound} | upper bound: {upper_bound} | solution time: {solution_time}\n')

def operational_problem_algorithm2(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation):
    problem_status, pump_status, valve_status, lower_bound, upper_bound, solution_time = MosekOperationalModel(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
    write_results(epanet_model, 'algorithm2 mosek', problem_status, lower_bound, upper_bound, solution_time)

    if problem_status == 'feasible':
        problem_status, lower_bound, upper_bound, solution_time = PyomoOperationalModel('algorithm2', epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, pump_status, valve_status)
        write_results(epanet_model, 'algorithm2 ipopt', problem_status, lower_bound, upper_bound, solution_time)

    return problem_status, lower_bound, upper_bound, solution_time

if __name__ == '__main__':

    head_loss_equation = 'HW'

    water_networks = [
        ('Net1', 'epanet'),
        ('ATM-12_Steps-Day_1', 'tasseff'),
        ('ATM-12_Steps-Day_2', 'tasseff'),
        ('Anytown', 'epanet'),
        ('Simple_FSD-12_Steps-Day_2', 'tasseff')
    ]

    methods = ['algorithm2', 'ipopt', 'gurobi', 'baron']

    for epanet_model, model_folder in water_networks:
        water_network_path = f'data/{model_folder}/{epanet_model}.inp'

        junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep = fetch_epanet_data(water_network_path, head_loss_equation)

        results_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', epanet_model)
        os.makedirs(results_directory, exist_ok=True)

        for method in methods:
            if method == 'gurobi':
                problem_status, lower_bound, upper_bound, solution_time = GurobiOperationalModel(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
            elif method == 'mosek':
                problem_status, pump_status, valve_status, lower_bound, upper_bound, solution_time = MosekOperationalModel(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
            elif method == 'algorithm2':
                problem_status, lower_bound, upper_bound, solution_time = operational_problem_algorithm2(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
            else:
                problem_status, lower_bound, upper_bound, solution_time = PyomoOperationalModel(method, epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)

            if method != 'algorithm2':
                write_results(epanet_model, method, problem_status, lower_bound, upper_bound, solution_time)