#!/bin/python
import yaml
import sys
import os
import numpy as np


def saw(val, min, max):
    range = max - min
    if range == 0.0:
      return min
    else:
      val = np.mod(np.abs(val), range*2)
      if(val) > range:
        val = range*2 - val
      val += min
      return val

if len(sys.argv)>=3:
  paramf = yaml.load(open(os.path.abspath(sys.argv[2])).read())["Parameters"]
  psof = open(sys.argv[1]).read().split("\n")
  pOut = psof[2].split(', ')
  if len(paramf)==(len(pOut)-1):
    out = open(os.path.abspath(sys.argv[3]),"w")
    out.write("#"+psof[0])
    for i in range(len(pOut)-1):
      out.write(paramf[i]["name"]+": "+str(round(saw(float(pOut[i]),paramf[i]["min"],paramf[i]["max"]),5))+"\n")
  else:
    print "ERROR: Number of parameters do not match!"
else:
  print "Failed. Usage: python evalPSOOutput.py $pso_best_params.dat $param.yml $output_parameters.yml"
