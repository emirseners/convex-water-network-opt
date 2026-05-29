from read_epanet import print_water_network_info, fetch_epanet_data
from pyomo_operational import PyomoOperationalModel
from mosek_operational import MosekOperationalModel

if __name__ == '__main__':

    head_loss_equation = 'HW'
    
    epanet_model = 'Net1'

    #water_network_path = 'data/tasseff/ATM/ATM-12_Steps-Day_1.inp'

    water_network_path = f'data/epanet/{epanet_model}.inp'

    junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep = fetch_epanet_data(water_network_path, head_loss_equation)

    PyomoOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
    #MosekOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)