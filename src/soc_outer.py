



def pipe_soc_outer_app(pipes, time_steps, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    for pipe in pipes.values():
        pipe.soc_outer_approximation = {}
        for t in range(1, time_steps+1):
            if pipe.flow_lower_bound[t-1] == pipe.flow_upper_bound[t-1]: continue
            if pipe.flow_upper_bound[t-1] > 0.0:
                pipe.soc_outer_approximation[t,'pos'] = {'exponent': 1 / exponent, 'x_coef': 1, 'y_coef': 1, 'z_coef': pipe.loss_coefficient ** (1 / exponent), 'x_bounds': [min(0.0, pipe.start_node.head_lower_bound[t-1] - pipe.end_node.head_upper_bound[t-1]), max(0.0, pipe.start_node.head_upper_bound[t-1] - pipe.end_node.head_lower_bound[t-1])], 'y_bounds': [0.0, 1.0], 'z_bounds': [0.0, max(0.0, pipe.flow_upper_bound[t-1])]}

            if pipe.flow_lower_bound[t-1] < 0.0:
                pipe.soc_outer_approximation[t,'neg'] = {'exponent': 1 / exponent, 'x_coef': 1, 'y_coef': 1, 'z_coef': pipe.loss_coefficient ** (1 / exponent), 'x_bounds': [min(0.0, pipe.end_node.head_lower_bound[t-1] - pipe.start_node.head_upper_bound[t-1]), max(0.0, pipe.end_node.head_upper_bound[t-1] - pipe.start_node.head_lower_bound[t-1])], 'y_bounds': [0.0, 1.0], 'z_bounds': [min(0.0, pipe.flow_lower_bound[t-1]), 0.0]}

def pump_soc_outer_app(pumps, time_steps):
    for pump in pumps.values():
        pump.soc_outer_approximation = {}
        for t in range(1, time_steps+1):
            if pump.decisions[t-1] == 0 or pump.flow_lower_bound[t-1] == pump.flow_upper_bound[t-1] or pump.curve_c == 1: continue
            pump.soc_outer_approximation[t] = {'exponent': 1 / pump.curve_c, 'x_coef': 1, 'y_coef': 1, 'z_coef': (-pump.curve_b) ** (1 / pump.curve_c), 'x_bounds': [min(0.0, pump.start_node.head_lower_bound[t-1] - pump.end_node.head_upper_bound[t-1] + pump.curve_a), max(0.0, pump.start_node.head_upper_bound[t-1] - pump.end_node.head_lower_bound[t-1] + pump.curve_a)], 'y_bounds': [0.0, 1.0], 'z_bounds': [min(0.0, pump.flow_lower_bound[t-1]), max(0.0, pump.flow_upper_bound[t-1])]}