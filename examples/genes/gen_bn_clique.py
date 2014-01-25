############################################################
#  This examples uses the same genetic network as before
#  but now we find marginals by belief propagation
#  we create a clique tree, calibrate and query
#  amazingly fast
############################################################
import numpy as np
import sys
sys.path.append( '../../.' ) # logysoft module address

import Factor
import Rvar
import FactorOperations
import UndGraph
import BFS_Paths
import IndMarkovNet
import CliqueTree

v = {}
v[0] = Rvar.Rvar('rene_g', 3)
v[1] = Rvar.Rvar('rene_p', 2)
v[2] = Rvar.Rvar('jason_g', 3)
v[3] = Rvar.Rvar('jason_p', 2)
v[4] = Rvar.Rvar('eva_g', 3)
v[5] = Rvar.Rvar('eva_p', 2)
v[6] = Rvar.Rvar('sandra_g', 3)
v[7] = Rvar.Rvar('sandra_p', 2)
v[8] = Rvar.Rvar('aaron_g', 3)
v[9] = Rvar.Rvar('aaron_p', 2)
v[10] = Rvar.Rvar('benito_g', 3)
v[11] = Rvar.Rvar('benito_p', 2)
v[12] = Rvar.Rvar('james_g', 3)
v[13] = Rvar.Rvar('james_p', 2)
v[14] = Rvar.Rvar('ira_g', 3)
v[15] = Rvar.Rvar('ira_p', 2)
v[16] = Rvar.Rvar('robin_g', 3)
v[17] = Rvar.Rvar('robin_p', 2)

f = [None]*18
# Robin
f[0] = Factor.Factor([v[17], v[16]])
f[0].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
f[1] = Factor.Factor([v[16]])
f[1].fill_values([0.01, 0.18, 0.81])

# Ira
f[2] = Factor.Factor([v[15], v[14]])
f[2].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
f[3] = Factor.Factor([v[14]])
f[3].fill_values([0.01, 0.18, 0.81])

# James
f[4] = Factor.Factor([v[13], v[12]])
f[4].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
f[5] = Factor.Factor([v[12], v[14], v[16]])
f[5].fill_values([1., 0., 0., 0.5, 0.5, 0.0, 0.0, 1.0, 0., 0.5, 0.5, 0., 0.25, 0.5, 0.25, 0., 0.5, 0.5, 0., 1., 0., 0., 0.5, 0.5, 0., 0., 1.])

# Benito
f[6] = Factor.Factor([v[11], v[10]])
f[6].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
f[7] = Factor.Factor([v[10], v[12], v[0]])
f[7].fill_values([1., 0., 0., 0.5, 0.5, 0.0, 0.0, 1.0, 0., 0.5, 0.5, 0., 0.25, 0.5, 0.25, 0., 0.5, 0.5, 0., 1., 0., 0., 0.5, 0.5, 0., 0., 1.])

# Aaron
f[8] = Factor.Factor([v[9], v[8]])
f[8].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
f[9] = Factor.Factor([v[8]])
f[9].fill_values([0.01, 0.18, 0.81])

# Sandra
f[10] = Factor.Factor([v[7], v[6]])
f[10].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
f[11] = Factor.Factor([v[6], v[8], v[4]])
f[11].fill_values([1., 0., 0., 0.5, 0.5, 0.0, 0.0, 1.0, 0., 0.5, 0.5, 0., 0.25, 0.5, 0.25, 0., 0.5, 0.5, 0., 1., 0., 0., 0.5, 0.5, 0., 0., 1.])

# Eva
f[12] = Factor.Factor([v[5], v[4]])
f[12].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
f[13] = Factor.Factor([v[4], v[14], v[16]])
f[13].fill_values([1., 0., 0., 0.5, 0.5, 0.0, 0.0, 1.0, 0., 0.5, 0.5, 0., 0.25, 0.5, 0.25, 0., 0.5, 0.5, 0., 1., 0., 0., 0.5, 0.5, 0., 0., 1.])

# Jason
f[14] = Factor.Factor([v[3], v[2]])
f[14].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
f[15] = Factor.Factor([v[2], v[12], v[0]])
f[15].fill_values([1., 0., 0., 0.5, 0.5, 0.0, 0.0, 1.0, 0., 0.5, 0.5, 0., 0.25, 0.5, 0.25, 0., 0.5, 0.5, 0., 1., 0., 0., 0.5, 0.5, 0., 0., 1.])

# Rene
f[16] = Factor.Factor([v[1], v[0]])
f[16].fill_values([0.8, 0.2, 0.6, 0.4, 0.1, 0.9])
f[17] = Factor.Factor([v[0]])
f[17].fill_values([0.01, 0.18, 0.81])


cc = CliqueTree.CliqueTree(f)
cc.calibrate()
print cc

# Alternative 1. Observe and recalibrate
# Change (reduce) the initial factors and calibrate
# we observe that rene has g=0 and james has g=1 // FactorOperations.observe(f[17], {rene_g:0, james_g:1})
#                                                                # without obervations => benito phen => 0.197
#f[17] = FactorOperations.observe(f[17], {f[17].variables[0]:0})  # only this => benito phen => 0.62
#f[5] = FactorOperations.observe(f[5], {f[5].variables[0]:1})     # if also this => benito phen => 0.7 (and 0.8 if val is 0)
#f[2] = FactorOperations.observe(f[2], {f[2].variables[0]:0})     # if also this => benito phen => 0.7

# Alternative 2. Calibrate, reduce a single node with observed var, and recalibrate
#benito_g, james_g, rene_g = cc.factors[10].variables
#cc.factors[10] = FactorOperations.observe(cc.factors[10], {rene_g:0, james_g:1})
#cc.calibrate()

# Alternative 3. aka 'HOLY FUCK!'
# Reduce a single node with observed var, compute messages till node with
# target var and compute new belief to question

#print cc.factors[10].variables
#benito_g, james_g, rene_g = cc.factors[10].variables

#belief_10 = FactorOperations.observe(cc.factors[10], {rene_g:0, james_g:1})
#print cc.adj[2]
#msg_10_1 = FactorOperations.multiply(belief_10, cc.delta[11][2], True)
#msg_10_1 = FactorOperations.marginalize(msg_10_1, msg_10_1.variables[1])
#msg_10_1 = FactorOperations.marginalize(msg_10_1, msg_10_1.variables[1])
#print '---->', msg_10_1.variables

#belief_1 = FactorOperations.multiply(msg_10_1, cc.factors[1], True)

#print belief_1.variables
#sol = FactorOperations.marginalize(belief_1, belief_1.variables[0])

#print 'benito prob of having illnes is now :', sol.values[0]

# COMPUTE ALL EXACT MARGINALS (of showing the sickness for all)
import copy

# for fun lets reduce some evidence
cc.factors[3] = FactorOperations.observe(cc.factors[3], {v[15]:0}) # Ira shows pheno
cc.factors[6] = FactorOperations.observe(cc.factors[6], {v[0]:0})  # rene has gen FF
cc.factors[4] = FactorOperations.observe(cc.factors[4], {v[12]:1}) # James has gen Ff
cc.calibrate()

phenos_nodes = [0,1,2,3,4,5,6,7,8]
probs = {}
for i in phenos_nodes:
	belief = cc.beta[i]
	genes = [v for v in belief.variables if not v.id.endswith("_p")]
	f = copy.copy(belief)
	f = FactorOperations.marginalize(f, genes[0])
	probs[f.variables[0].id] = f.values[0]
print probs

