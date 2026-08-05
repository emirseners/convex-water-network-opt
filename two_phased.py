import os
import time
from helpers import write_results
from hull_facets import find_pipe_facets, find_pump_facets
from operational_pyomo import PyomoOperationalModel
from operational_gurobi import GurobiOperationalModel
from operational_mosek import MosekOperationalModel, SolveMosekOperationalModel, AddNoGoodCut

def operational_problem_two_phased(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, flow_direction = False, num_rounds = 5):
    hull_facet_start = time.time()
    find_pipe_facets(pipes, time_steps, head_loss_equation)
    find_pump_facets(pumps, time_steps)
    hull_facet_time = time.time() - hull_facet_start

    mosek_model = MosekOperationalModel('two_phased mosek', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)

    for r in range(num_rounds):

        mosek_status, mosek_solution, mosek_lower_bound, mosek_upper_bound, mosek_solution_time = SolveMosekOperationalModel(mosek_model, 'two_phased mosek', results_directory)
        write_results(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), 'results.txt', f'{epanet_model} | two_phased mosek | {mosek_status} | {mosek_lower_bound} | {mosek_upper_bound} | {mosek_solution_time} | - | {hull_facet_time if r == 0 else "-"}')

        if mosek_status != 'feasible':
            mosek_model.dispose()
            return

        GurobiOperationalModel('two_phased mosek check', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, mosek_solution)

        ipopt_status, ipopt_solution, ipopt_lower_bound, ipopt_upper_bound, ipopt_solution_time = PyomoOperationalModel('two_phased', 'two_phased ipopt flow direction', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, mosek_solution, flow_direction)
        write_results(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), 'results.txt', f'{epanet_model} | two_phased ipopt | {ipopt_status} | {ipopt_lower_bound} | {ipopt_upper_bound} | {ipopt_solution_time} | - | -')

        if len(pumps) + len(valves) == 0 or ipopt_status == 'feasible': break

        if ipopt_status != 'feasible':
            AddNoGoodCut(mosek_model, mosek_solution)

    if ipopt_status == 'feasible':
        GurobiOperationalModel('two_phased ipopt check', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, ipopt_solution)

    mosek_model.dispose()