from read_epanet import print_water_network_info, fetch_epanet_data
from operational_pyomo import PyomoOperationalModel
from operational_gurobi import GurobiOperationalModel
from operational_mosek import MosekOperationalModel
import numpy as np

def hazen_williams_headloss(length, roughness, diameter, flow):
    return ((10.67 * length) / ((roughness ** 1.852) * (diameter ** 4.8704))) * flow * (abs(flow) ** 0.852)

def darcy_weisbach_headloss(length, roughness, diameter, flow):
    f = (2 * np.log10((3.7 * diameter) / roughness)) ** (-2)
    return ((8 * f * length) / ((np.pi ** 2) * 9.80665 * (diameter ** 5))) * flow * abs(flow)

if __name__ == '__main__':

    head_loss_equation = 'HW'

    solver = 'mosek'

    epanet_model, model_folder = 'Net1', 'epanet'

    #epanet_model, model_folder = 'ATM-12_Steps-Day_1', 'tasseff'

    water_network_path = f'data/{model_folder}/{epanet_model}.inp'

    junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep = fetch_epanet_data(water_network_path, head_loss_equation)

    if solver == 'gurobi':
        GurobiOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
    elif solver == 'baron':
        PyomoOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
    elif solver == 'mosek':
        MosekOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)