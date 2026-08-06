import os
import time
from helpers import write_results
from bound_tightening import mosek_obbt
from operational_pyomo import PyomoOperationalModel
from operational_gurobi import GurobiOperationalModel
from two_phased import operational_problem_two_phased
from read_epanet import fetch_epanet_data, set_bounds, set_timesteps, preprocess_decisions

if __name__ == '__main__':
    head_loss_equation = 'HW'

    water_networks = [
        ('Net1', 'epanet'),
        ('vanzyl', 'epanet'),
        ('Simple_FSD-24_Steps-Day_1', 'tasseff'),
        ('Simple_FSD-24_Steps-Day_2', 'tasseff'),
        ('ATM-12_Steps-Day_1', 'tasseff'),
        ('ATM-12_Steps-Day_2', 'tasseff'),
        ('Poormond-12_Steps-Day_1', 'tasseff'),
        ('Poormond-12_Steps-Day_2', 'tasseff'),
        ('Richmond_skeleton', 'adnetworks'),
        ('Anytown', 'epanet'),
        ('Net3', 'epanet')
    ]

    methods = ['gurobi', 'two_phased']
    obbt_flag = True

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
    write_results(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), 'results.txt', f'Model | Method | Status | LB | UB | T.T. | Obbt T. | Hull Facet T.', mode = 'w')

    for epanet_model, model_folder in water_networks:
        water_network_path = f'data/{model_folder}/{epanet_model}.inp'
        junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep = fetch_epanet_data(water_network_path, head_loss_equation)

        results_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', epanet_model)
        os.makedirs(results_directory, exist_ok=True)

        set_bounds(junctions, tanks, pipes, pumps, valves, head_loss_equation, time_steps, hydraulic_timestep, results_directory)
        #time_steps = min(3, time_steps)
        #set_timesteps(junctions, tanks, pipes, pumps, valves, list(range(time_steps)))

        if obbt_flag:
            obbt_start = time.time()
            mosek_obbt(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, results_directory)
            obbt_time = time.time() - obbt_start
            preprocess_decisions(junctions, tanks, pipes, pumps, valves, time_steps, head_loss_equation, True, results_directory)

        for method in methods:
            if method == 'gurobi':
                problem_status, lower_bound, upper_bound, solution_time = GurobiOperationalModel('gurobi', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
            elif method == 'two_phased':
                operational_problem_two_phased(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
            else:
                problem_status, solution, lower_bound, upper_bound, solution_time = PyomoOperationalModel(method, method, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)

            if method != 'two_phased':
                write_results(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), 'results.txt', f'{epanet_model} | {method} | {problem_status} | {lower_bound:.2g} | {upper_bound:.2g} | {solution_time:.2g} | {obbt_time:.2g if obbt_flag else "-"} | -')