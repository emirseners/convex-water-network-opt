import os
import time
from src.helpers import write_results
from src.graph import fix_bridge_flows
from src.bound_tightening import mosek_obbt
from src.operational.pyomo import PyomoOperationalModel
from src.operational.gurobi import GurobiOperationalModel
from src.read_epanet import fetch_epanet_data, set_bounds, set_timesteps, preprocess_decisions
from src.algorithms import operational_problem_two_phased, operational_problem_two_phased_soc, operational_problem_soc_callback

if __name__ == '__main__':
    head_loss_equation = 'HW'

    water_networks = [
        ('Net1', 'epanet'),
        ('Simple_FSD-24_Steps-Day_1', 'tasseff'),
        ('Simple_FSD-24_Steps-Day_2', 'tasseff'),
        ('ATM-12_Steps-Day_1', 'tasseff'),
        ('ATM-12_Steps-Day_2', 'tasseff'),
        ('Poormond-12_Steps-Day_1', 'tasseff'),
        ('Poormond-12_Steps-Day_2', 'tasseff'),
        ('Richmond_skeleton', 'epanet'),
        ('vanzyl', 'epanet'),
        ('Anytown', 'epanet'),
        ('Net3', 'epanet')
    ]

    methods = ['gurobi', 'two_phased_mosek', 'two_phased_gurobi_soc', 'gurobi_callback']
    methods = ['gurobi', 'two_phased_gurobi_soc', 'gurobi_callback']
    obbt_flags = [True]
    run_prefix = f'gurobi_comparison_{7}'

    for obbt_flag in obbt_flags:
        run_name = f'{run_prefix}_obbt({obbt_flag})'
        results_folder = os.path.join('results', run_name)
        os.makedirs(results_folder, exist_ok=True)
        write_results(results_folder, 'results.txt', 'Model | Method | Status | LB | UB | Solve.T. | Obbt T. | Hull Facet T.', mode='w')

        for epanet_model, model_folder in water_networks:
            junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep = fetch_epanet_data(f'data/{model_folder}/{epanet_model}.inp', head_loss_equation)

            results_directory = os.path.join(results_folder, epanet_model)
            os.makedirs(results_directory, exist_ok=True)

            set_bounds(junctions, tanks, pipes, pumps, valves, head_loss_equation, time_steps, hydraulic_timestep, results_directory)
            fix_bridge_flows(junctions | tanks, pipes | pumps | valves)
            #time_steps = min(3, time_steps)
            #set_timesteps(junctions, tanks, pipes, pumps, valves, list(range(time_steps)))

            if obbt_flag:
                obbt_start = time.time()
                mosek_obbt(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, results_directory)
                obbt_time = time.time() - obbt_start
                preprocess_decisions(junctions, tanks, pipes, pumps, valves, time_steps, head_loss_equation, True, results_directory)

            for method in methods:
                if method == 'gurobi':
                    problem_status, solution, lower_bound, upper_bound, solution_time = GurobiOperationalModel('gurobi', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
                elif method == 'two_phased_mosek':
                    problem_status, solution, lower_bound, upper_bound, solution_time = operational_problem_two_phased(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
                elif method == 'two_phased_gurobi_soc':
                    problem_status, solution, lower_bound, upper_bound, solution_time = operational_problem_two_phased_soc(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
                elif method == 'gurobi_callback':
                    problem_status, solution, lower_bound, upper_bound, solution_time = operational_problem_soc_callback(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
                else:
                    problem_status, solution, lower_bound, upper_bound, solution_time = PyomoOperationalModel(method, method, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)

                write_results(results_folder, 'results.txt', f'{epanet_model} | {method} | {problem_status} | {f"{lower_bound:.2f}" if lower_bound is not None else "-"} | {f"{upper_bound:.2f}" if upper_bound is not None else "-"} | {solution_time:.2f} | {f"{obbt_time:.2f}" if obbt_flag else "-"} | -')