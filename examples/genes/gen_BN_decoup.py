import numpy as np
import itertools

import sys
sys.path.append( '../../.' ) # logysoft module address
import Factor
import Rvar
import FactorOperations


### HELPER FUNCTIONS ###
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

def alleles2Genes(numAlleles):
    '''reverse mapping from the number of alleles to the index position in a sequential array
       0 => [0,0], 1 => [0,1], 2 => [1,0], 3 => [1,1]'''
    sol = {}
    count = 0
    for i in range(numAlleles):
        for j in range(numAlleles):
            if i<j:
                sol[count] = (i,j)
            else:
                sol[count] = (j, i)
            count +=1
    return sol

### FACTOR BUIDLING FUNCTIONS ###

def Factor_genotype_given_allele_freqs(alleleFreqs, genoVar):
    f = Factor.Factor([genoVar])
    f.fill_values(alleleFreqs)
    return f

def Factor_genotype_given_parents_genes(numAlleles, genKid, gen_parent_1, gen_parent_2):
    ff = Factor.Factor([genKid, gen_parent_1, gen_parent_2])
    arr = ff.values.flatten()
    alleleProbsCount = 0;
    for i in range(numAlleles):
        for j in range(numAlleles):
            for k in range(numAlleles):
                if j == k:
                    if i == k:
                        arr[alleleProbsCount] = 1.
                    else:
                        arr[alleleProbsCount] = .5
                elif i == k:
                    arr[alleleProbsCount] = .5
                alleleProbsCount += 1
    ff.fill_values(arr)
    return ff

def Factor_phenotype_given_genotype_Not_Mendelian(alphaList, numAlleles, genoVar_1, genoVar_2, phenoVar):
    ff = Factor.Factor([phenoVar, genoVar_1, genoVar_2]) # written like in cond prob order    
    
    chocho = alleles2Genes(numAlleles)
    chochin = genes2Alleles(numAlleles)
    cols_CPD = genoVar_1.totCard() * genoVar_2.totCard()
    arr = ff.values.flatten()

    for i in range(len(chocho)):
        k = chocho[i] 
        v = alphaList[chochin[k]]
        arr[2*i] = v        # trait present
        arr[2*i+1] = 1-v    # trait absent
    ff.fill_values(arr)
    return ff


### GENETIC NETWORK ###

def geneticNetwork(pedigree, alleleFreqs, alphaList):
    PHENOTYPE_CARD = 2 # present or absent
    numAlleles = len(alleleFreqs)

    varList = {}

    # first we create all variables
    count = 0
    for name in pedigree:
        varList[name] = {}
        dad_gn, mom_gn = pedigree[name]
        varList[name]['var_geno1'] = Rvar.Rvar(count, numAlleles)
        count +=1
        varList[name]['var_geno2'] = Rvar.Rvar(count, numAlleles)
        count +=1
        varList[name]['var_pheno'] = Rvar.Rvar(count, PHENOTYPE_CARD)
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
        varList[name]['factor_geno1'] = f1
        varList[name]['factor_geno2'] = f2

        varList[name]['factor_pheno'] = Factor_phenotype_given_genotype_Not_Mendelian(alphaList, numAlleles, kid_gn1, kid_gn2, kid_ph)

    return varList

# this is fugly
def modify_Factor_by_evidence(name, node, ass):
    factor = GN[name]['factor_'+node]
    randvar = GN[name]['var_'+node]
    GN[name]['factor_'+node] = FactorOperations.observe(factor, {randvar:ass})

def build_joint_cpd():
    a = None
    for k in GN.keys():
        b = FactorOperations.multiply(GN[k]['factor_geno1'], GN[k]['factor_geno2'])
        b = FactorOperations.multiply(b, GN[k]['factor_pheno'])
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
#family_tree['Aaron'] = [None, None]
family_tree['Rene'] = [None, None]
family_tree['James'] = ['Ira', 'Robin']
family_tree['Eva']   = ['Ira', 'Robin']
#family_tree['Sandra'] = ['Aaron', 'Eva']
family_tree['Jason'] = ['James', 'Rene']
family_tree['Benito'] = ['James', 'Rene']

frequency_of_alleles_in_general_population = [0.1, 0.7, 0.2]
probability_of_trait_based_on_genotype = [0.8, 0.6, 0.1, 0.5, 0.05, 0.01]

TRAIT_PRESENT, TRAIT_ABSENT = [0,1]
F,f,n = [0,1,2]

## BUILDING THE NETWORK ##
# I commented to crash it and stop the long delay while I tinker with the clique version
#GN = geneticNetwork(family_tree, frequency_of_alleles_in_general_population, probability_of_trait_based_on_genotype)



# Evidence conditioning

#modify_Factor_by_evidence('Ira',  'pheno', TRAIT_PRESENT)
#modify_Factor_by_evidence('Rene', 'geno1', F)
#modify_Factor_by_evidence('Rene', 'geno2', f)
#modify_Factor_by_evidence('Eva',  'pheno', TRAIT_PRESENT)

# lets try first the whole kahuna CPD and compute the prob of developing CF


#a = build_joint_cpd()

## marginalizing
#target = GN['Benito']['var_pheno']
#lista = [x for x in a.variables if x!=target]
##cpd = a.marginalize_by(lista)
#for v in lista:
#    a = FactorOperations.marginalize(a, v)
#    print a.variables, a.values.size


#print 'probability of Benito showing ailment', 100.*a.values[0], '%'




