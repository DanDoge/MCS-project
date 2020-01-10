
import os
import time
import zlib
import numpy as np
import sys
import cPickle as pickle
import json
import sys


import subprocess

processes = set()
max_processes = 8

datasets = ["prot", "yach"]
#datasets = ['boston']
for dataset in datasets:
	M = 50
	I = 2000


	command_list = []
	'''
	command_list.append('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_reg_sep.py -d ' + dataset + ' -m ' + str(M) + ' -i ' + str(I))
	command_list.append('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_reg_sep.py -d ' + dataset + ' -hi 1 ' + ' -m ' + str(M) + ' -i ' + str(I))
	command_list.append('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_reg_sep.py -d ' + dataset + ' -hi 2 ' + ' -m ' + str(M) + ' -i ' + str(I))
	command_list.append('OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_reg_sep.py -d ' + dataset + ' -hi 3 ' + ' -m ' + str(M) + ' -i ' + str(I))
	'''
	#command_list.append('python run_reg_data.py -d ' + dataset + ' -m ' + str(M) + ' -i ' + str(I))
	command_list.append('python run_reg_data.py -d ' + dataset + ' -hi 1 ' + ' -m ' + str(M) + ' -i ' + str(I))
	#command_list.append('python run_reg_data.py -d ' + dataset + ' -hi 2 ' + ' -m ' + str(M) + ' -i ' + str(I))
	#command_list.append('python run_reg_data.py -d ' + dataset + ' -hi 3 ' + ' -m ' + str(M) + ' -i ' + str(I))

	for i, command in enumerate(command_list):
		print 'running, ', command
		name = dataset + '_' + str(i)
		processes.add(subprocess.Popen(command, shell=True))
		'''
		if len(processes) >= max_processes:
			os.wait()
			processes.difference_update([
				p for p in processes if p.poll() is not None])
		'''
