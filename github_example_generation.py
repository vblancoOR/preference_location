# -*- coding: utf-8 -*-
"""
This file contains an example application to generate the parameters to generate the parameters involved 
in the location with preferences paper
Victor Blanco, Ricardo Gázquez and Marina Leal.
"""

import github_gen_parameters_library as gen_lib


NN = [10,20,50,100,500,1000]
SED = [2036,2732,3264,6744,7891,9205]
FUNCTIONS   = ["L","D","CES","CD","GL"]
NORMS = [1,2,3]
  
for nn in NN:        
    for sed in SED:
        for ff in FUNCTIONS:
            for nrs in NORMS:
                A, w, R, Norms, Norms2 = gen_lib.read_data(nn, sed, nrs)
                F = gen_lib.gen_parameters(A, R, nrs, ff, sed)