import numpy as np
import itertools

import sys
sys.path.append( '../../.' ) # logysoft module address
import Factor
import Rvar
import FactorOperations


### HELPER FUNCTIONS ###

def numGenos(numAlleles):
    '''how many genotypes result based on the number of alleles'''
    return numAlleles*(numAlleles+1)/2

def genes2Alleles(numAlleles):
    '''mapping from the number of alleles to the index position in a sequential array
       [0,0] => 0, [0,1] => 1, [1,0] => 1, [1,1] =>2'''
    sol = {}
    count = 0
    for i in range(numAlleles):
        for j in range(i, numAlleles):
            sol[(i,j)] = count
            count +=1
    return sol

### FACTOR BUIDLING FUNCTIONS ###

def Factor_phenotype_given_genotype_Not_Mendelian(alphaList, genoVar, phenoVar):
    ff = Factor.Factor([phenoVar, genoVar]) # written like in cond prob order
    order, dic = ff.idx2ass(ff.values.flatten())
    arr = ff.values.copy().flatten()
    for i in range(len(alphaList)):
        v = alphaList[i]
        k = order.index((0,i))  # trait present
        arr[k] = v
        arr[k+1] = 1.-v    # trait absent
    ff.fill_values(arr)
    return ff

def Factor_genotype_given_allele_freqs(alleleFreqs, genoVar):
    ff = Factor.Factor([genoVar])
    
    numAlleles = len(alleleFreqs)    
    numGenotypes = genoVar.totCard()
    chocho = genes2Alleles(numAlleles)
    for row in range(numAlleles):
        for col in range(numAlleles):
            key = chocho[tuple(sorted((row, col)))]
            ff.values[key] += alleleFreqs[row]*alleleFreqs[col]
    return ff

def Factor_genotype_given_parents_genes(numAlleles, genKid, genDad, genMom):
    ff = Factor.Factor([genKid, genDad, genMom])
    order, dic = ff.idx2ass(ff.values.flatten())
    chocho = genes2Alleles(numAlleles)
    kid = chocho
    dad = chocho.copy()
    mom = chocho.copy()
    arr = ff.values.flatten()
    for dad_gen in dad:
        for mom_gen in mom:
            for kid_gen in kid:
                hits = 0.
                for chuchu in itertools.product(dad_gen, mom_gen):
                    if sorted(chuchu)==list(kid_gen):
                        hits += 1
                hits = hits/4.
                arr[order.index((kid[kid_gen], dad[dad_gen], mom[mom_gen]))] = hits
    ff.fill_values(arr)
    return ff

### GENETIC NETWORK ###

def geneticNetwork(pedigree, alleleFreqs, alphaList):
    PHENOTYPE_CARD = 2 # present or absent
    numAlleles = len(alleleFreqs)
    numG = numGenos(numAlleles)
    
    varList = {}
    
    # first we create all variables
    count = 0
    for name in pedigree:
        varList[name] = {}
        varList[name]['var_geno'] = Rvar.Rvar(count, numG)
        count +=1
        varList[name]['var_pheno'] = Rvar.Rvar(count, PHENOTYPE_CARD)
        count +=1
        
    # and now the factors
    for name in pedigree:
        kid_gn = varList[name]['var_geno']
        kid_ph = varList[name]['var_pheno']
        
        dad_gn, mom_gn = pedigree[name]
        if dad_gn and mom_gn:
            dad_gn = varList[dad_gn]['var_geno']
            mom_gn = varList[mom_gn]['var_geno']
            ff = Factor_genotype_given_parents_genes(numAlleles, kid_gn, dad_gn, mom_gn)
        else:
            ff = Factor_genotype_given_allele_freqs(alleleFreqs, kid_gn)
        varList[name]['factor_geno'] = ff
        varList[name]['factor_pheno'] = Factor_phenotype_given_genotype_Not_Mendelian(alphaList, kid_gn, kid_ph)
    
    return varList

# this is fugly
def modify_Factor_by_evidence(name, node, ass):
    factor = GN[name]['factor_'+node]
    randvar = GN[name]['var_'+node]
    GN[name]['factor_'+node] = FactorOperations.observe(factor, {randvar:ass})

def build_joint_cpd():
    a = None
    for k in GN.keys():
        b = FactorOperations.multiply(GN[k]['factor_geno'], GN[k]['factor_pheno'])
        if a==None:
            a = b
        else:
            a = FactorOperations.multiply(a, b)
    return a

################################################################
################################################################
################################################################


## THE INPUT DATA ##

family_tree = {}
family_tree['Ira'] = [None, None]
family_tree['Robin'] = [None, None]
family_tree['Aaron'] = [None, None]
family_tree['Rene'] = [None, None]
family_tree['James'] = ['Ira', 'Robin']
family_tree['Eva']   = ['Ira', 'Robin']
family_tree['Sandra'] = ['Aaron', 'Eva']
family_tree['Jason'] = ['James', 'Rene']
family_tree['Benito'] = ['James', 'Rene']

frequency_of_alleles_in_general_population = [0.1, 0.9]
probability_of_trait_based_on_genotype = [0.8, 0.6, 0.1]

TRAIT_PRESENT, TRAIT_ABSENT = [0,1]
FF, Ff, ff = [0,1,2]

## BUILDING THE NETWORK ##

GN = geneticNetwork(family_tree, frequency_of_alleles_in_general_population, probability_of_trait_based_on_genotype)

# Evidence conditioning

#modify_Factor_by_evidence('Ira',   'pheno', TRAIT_PRESENT)
modify_Factor_by_evidence('James', 'geno', Ff)
modify_Factor_by_evidence('Rene',  'geno', FF)

# lets try first the whole kahuna CPD and compute the prob of developing CF

a =build_joint_cpd()

# marginalizing
target = GN['Benito']['var_pheno']

lista = [x for x in a.variables if x!=target]

for v in lista:
    a = FactorOperations.marginalize(a, v)
    #print a.variables, a.values.size

print 'probability of Benito showing ailment', 100.*a.values[0], '%'






