import os
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

def write_results(results_directory, txt_name, line, mode = 'a'):
    with open(os.path.join(results_directory, txt_name), mode) as f:
        f.write(line + '\n')

def write_solution(solution, results_directory, output_file = 'solution.xlsx'):
    if solution is None: return
    to_write = [['decision_variable', 'value']]

    for variable, components in solution.items():
        for label, values in components.items():
            for t, value in enumerate(values):
                to_write.append([f'{variable}({label}, {t})', value])

    os.makedirs(results_directory, exist_ok = True)
    output_file = os.path.join(results_directory, os.path.basename(output_file))
    pd.DataFrame(to_write).to_excel(output_file, index = False, header = False)

def write_bounds(junctions, tanks, pipes, pumps, valves, time_steps, results_directory, sheet_name = 'bounds', output_file = 'decision_variable_bounds.xlsx'):
    to_write = [['decision_variable_key', 'lower_bound', 'upper_bound']]

    for label, node in (junctions | tanks).items():
        for t in range(time_steps):
            to_write.append([f'head({label}, {t})', node.head_lower_bound[t], node.head_upper_bound[t]])
        if label in tanks.keys():
            to_write.append([f'head({label}, {time_steps})', node.head_lower_bound[time_steps], node.head_upper_bound[time_steps]])

    for label, arc in (pipes | pumps | valves).items():
        for t in range(time_steps):
            to_write.append([f'flow({label}, {t})', arc.flow_lower_bound[t], arc.flow_upper_bound[t]])

    for variable, arcs in [('pump_status', pumps), ('valve_status', valves)]:
        for label, arc in arcs.items():
            for t in range(time_steps):
                to_write.append([f'{variable}({label}, {t})', arc.decisions[t]])

    os.makedirs(results_directory, exist_ok = True)
    output_file = os.path.join(results_directory, os.path.basename(output_file))
    with pd.ExcelWriter(output_file, engine = 'openpyxl', mode = 'a' if os.path.exists(output_file) else 'w') as writer:
        pd.DataFrame(to_write).to_excel(writer, sheet_name = sheet_name, index = False, header = False)

def round_bound(value, rounding, decimal='0.001'):
    rounding = ROUND_FLOOR if rounding == 'floor' else ROUND_CEILING

    if isinstance(value, np.ndarray):
        rounded = np.vectorize(lambda element: float(element) if not np.isfinite(element) else Decimal(str(float(element))).quantize(Decimal(decimal), rounding=rounding), otypes=[float])(value)
        return np.where(rounded == 0, 0.0, rounded)

    rounded = Decimal(str(float(value))).quantize(Decimal(decimal), rounding=rounding)
    return 0.0 if rounded == 0 else float(rounded)
