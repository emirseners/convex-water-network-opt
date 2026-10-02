import os
import json
import pandas as pd
import numpy as np
from fractions import Fraction
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
        if label in tanks:
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

def valid_inequality_key(terms, sense, rhs):
    coefficients = {}

    for variable, label, period, coefficient in terms:
        key = (variable, label, int(period))
        coefficients[key] = coefficients.get(key, Fraction(0)) + Fraction(float(coefficient))

    terms = sorted((key, coefficient) for key, coefficient in coefficients.items() if coefficient != 0)
    rhs = Fraction(float(rhs))
    sign = 1

    if sense == 'le':
        sense = 'ge'
        sign = -1
    elif sense == 'eq' and terms and terms[0][1] < 0:
        sign = -1

    scale = abs(terms[0][1]) if terms else Fraction(1)

    return (sense, tuple((key, sign * coefficient / scale) for key, coefficient in terms), sign * rhs / scale)

def write_valid_inequalities(inequalities, results_directory, output_file='valid_inequalities.json'):
    to_write = [
        {'terms': [(variable, label, period, float(coefficient)) for (variable, label, period), coefficient in terms],
         'sense': sense, 'rhs': float(rhs)} for sense, terms, rhs in sorted(inequalities)]

    os.makedirs(results_directory, exist_ok=True)
    output_file = os.path.join(results_directory, os.path.basename(output_file))
    content = json.dumps(to_write, indent=4, allow_nan=False)

    with open(output_file, 'w') as f:
        f.write(content + '\n')