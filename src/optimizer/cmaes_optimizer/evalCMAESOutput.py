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
  cmaesf = open(sys.argv[1])
  cmaes = cmaesf.read().split('\n')
  xbeststart = [l[0:9] for l in cmaes].index("xbestever")+1
  xbestend = [l[0:6] for l in cmaes].index("xbest ")
  i = xbeststart
  pOut = []
  while i<xbestend:
    pOut= pOut + cmaes[i].split()
    i+=1
  if len(paramf)==len(pOut):
    out = open(os.path.abspath(sys.argv[3]),"w")
    out.write(cmaes[[l[0:5] for l in cmaes].index("# ---")]+"\n#"+cmaes[xbeststart-1])
    for i in range(len(pOut)):
      out.write(paramf[i]["name"]+": "+str(round(saw(float(pOut[i]),paramf[i]["min"],paramf[i]["max"]),5))+"\n")
  else:
    print "ERROR: Number of parameters do not match!"
else:
  print "Failed. Usage: python evalCMAESOutput.py $allcmaes.dat $param.yml $output_parameters.yml"
