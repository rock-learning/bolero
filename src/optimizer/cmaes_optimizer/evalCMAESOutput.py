#!/bin/python3
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
    cmaes = open(sys.argv[1]).read().split('\n')
    xbeststart = [l[0:9] for l in cmaes].index("xbestever") + 1
    xbestend = [l[0:6] for l in cmaes].index("xbest ")
    i = xbeststart
    pOut = []
    while i < xbestend:
        pOut = pOut + cmaes[i].split()
        i += 1
    if len(param_file) == len(pOut):
        out_file = open(os.path.abspath(sys.argv[3]), "w")
        out_file.write(cmaes[[l[0:5] for l in cmaes].index("# ---")] + "\n#" + cmaes[xbeststart - 1] + "\n")
        for i in range(len(pOut)):
            out_file.write(param_file[i]["name"] + ": " + str(
                round(saw(float(pOut[i]), param_file[i]["min"], param_file[i]["max"]), 5)) + "\n")
    else:
        print("ERROR: Number of parameters do not match!")
else:
    print("Failed. Usage: python evalCMAESOutput.py $allcmaes.dat $param.yml $output_parameters.yml")
