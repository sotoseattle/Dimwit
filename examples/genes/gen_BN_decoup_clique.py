############################################################
#  This examples uses the same genetic network as before
#  but now we find marginals by belief propagation
#  we create a clique tree, calibrate and query
#  amazingly fast
############################################################
import copy
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
from gen_BN_decoup import *

def geneticNetwork(pedigree, alleleFreqs, alphaList):
    F = []
    v = []
    varList = {}
    PHENOTYPE_CARD = 2 # present or absent
    numAlleles = len(alleleFreqs)
    # first we create all variables
    count = 0
    for name in pedigree:
        varList[name] = {}
        dad_gn, mom_gn = pedigree[name]
        v1 = Rvar.Rvar(name+'_g1', numAlleles)
        varList[name]['var_geno1'] = v1
        v.append(v1)
        count +=1

        v2 = Rvar.Rvar(name+'_g2', numAlleles)
        varList[name]['var_geno2'] = v2
        v.append(v2)
        count +=1
        v3 = Rvar.Rvar(name+'_p', PHENOTYPE_CARD)
        varList[name]['var_pheno'] = v3
        v.append(v3)
        count +=1

    # and now the factors
    for name in pedigree:
        kid_gn1 = varList[name]['var_geno1']
        kid_gn2 = varList[name]['var_geno2']
        kid_ph  = varList[name]['var_pheno']
        dad_gn, mom_gn = pedigree[name]
        if dad_gn and mom_gn:
            f1 = Factor_genotype_given_parents_genes(numAlleles, kid_gn1, varList[dad_gn]['var_geno1'], varList[dad_gn]['var_geno2'])
            f2 = Factor_genotype_given_parents_genes(numAlleles, kid_gn2, varList[mom_gn]['var_geno1'], varList[mom_gn]['var_geno2'])
        else:
            f1 = Factor_genotype_given_allele_freqs(alleleFreqs, kid_gn1)
            f2 = Factor_genotype_given_allele_freqs(alleleFreqs, kid_gn2)
        f3 = Factor_phenotype_given_genotype_Not_Mendelian(alphaList, numAlleles, kid_gn1, kid_gn2, kid_ph)

        varList[name]['factor_geno1'] = f1
        varList[name]['factor_geno2'] = f2
        varList[name]['factor_pheno'] = f3

        F.append(f1)
        F.append(f2)
        F.append(f3)
    return [F,v]



[F, v] = geneticNetwork(family_tree, frequency_of_alleles_in_general_population, probability_of_trait_based_on_genotype)
print F
cc = CliqueTree.CliqueTree(F)

for i,e in enumerate(v):
    print i, e
# for fun lets reduce some evidence
cc.factors[2] = FactorOperations.observe(cc.factors[2], {v[17]:0}) # Ira shows pheno
cc.factors[5] = FactorOperations.observe(cc.factors[5], {v[6]:0})  # rene has gen1 F
cc.factors[5] = FactorOperations.observe(cc.factors[5], {v[7]:1})  # rene has gen2 f
cc.factors[1] = FactorOperations.observe(cc.factors[1], {v[5]:0}) # Eva shows pheno

cc.calibrate()
print cc

phenos_nodes = [0,1,2,3,4,5,6]
probs = {}
for i in phenos_nodes:
    belief = cc.beta[i]
    genes = [v1 for v1 in belief.variables if not v1.id.endswith("_p")]
    f = copy.copy(belief)
    print genes
    for g in genes:
        f = FactorOperations.marginalize(f, g)
    probs[f.variables[0].id] = f.values[0]
print probs

### CHECK CALIBRATION (IMPRESSIVE!)
# Check that the exact marginal over a var in adjacent nodes (beliefs) is the same
for vari in v:
    if not vari.id.endswith('_p'):
        print vari.id
        for beta in cc.beta:
            if vari in beta.variables:
                f = copy.copy(beta)
                f.values = FactorOperations.normalize(f.values)
                for g in (set(beta.variables) - set([vari])):
                    f = FactorOperations.marginalize(f, g)
                print f.values
