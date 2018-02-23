#!/bin/python
import yaml
import sys
import os
import numpy as np


def saw(val, param_min, param_max):
    param_range = param_max - param_min
    if param_range == 0.0:
        return param_min
    else:
        val = np.mod(np.abs(val), param_range * 2)
        if val > param_range:
            val = param_range * 2 - val
        val += param_min
        return val


if len(sys.argv) >= 3:
    param_file = yaml.load(open(os.path.abspath(sys.argv[2])).read())["Parameters"]
    pso_file = open(sys.argv[1]).read().split("\n")
    pOut = pso_file[2].split(', ')
    if len(param_file) == (len(pOut) - 1):
        out_file = open(os.path.abspath(sys.argv[3]), "w")
        out_file.write("#" + pso_file[0] + "\n")
        for i in range(len(pOut) - 1):
            out_file.write(param_file[i]["name"] + ": " + str(
                round(saw(float(pOut[i]), param_file[i]["min"], param_file[i]["max"]), 5)) + "\n")
    else:
        print "ERROR: Number of parameters do not match!"
else:
    print "Failed. Usage: python evalPSOOutput.py $pso_best_params.dat $param.yml $output_parameters.yml"
