# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:30:12 2017

@author: dan
"""
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps
import random as rnd
from datetime import datetime, date, time

amu = 1.66054e-27
alpha = 1/137.035999173
hbar = 1.0545718e-34
c = 3e8
m = 9.1093e-31
pi = 3.14159265359
e = 1.60217646e-19

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def fermi_function(Z, T, rN):
    E = T+m*c**2
    p = np.sqrt((E/c)**2 -(m*c)**2)
    S = np.sqrt(1-alpha**2*Z**2)
    rho = rN/hbar
    eta = (-1* alpha * Z * E) / (p*c)
    ff = ((2*(1+S))/sps.gamma(1+2*S)**2)*(2*p*rho)**(2*S-2)* np.exp(pi * eta) * np.abs(sps.gamma(S+(1j*eta)))**2
    return ff 

def N(Z, T, CL,Q, rN):
    E = T+(m*c**2)
    p = np.sqrt((E/c)**2 -(m*c)**2)
    ff = fermi_function(Z,T, rN)
    no = CL*ff*p*E*(Q-T)**2
    return no

def getRN():
    R = np.float64(rnd.getrandbits(256)/np.float64(2.0**256.0))
    return R 

def find_in_range(array,value):
    l = len(array)
    index = 0 
    while index < l: 
        if index==0 :
            if value > 0 and value < array[0] :
                return 0 
            index += 1 
        else:
            if value > array[index-1] and value < array[index]:
                return index
            index += 1 
    return -1

def ion_walk(E, ranges, IE, minE):   #minE and IE in eV 
    #['H','He','H2']
    #Ionization_Energy = [13.6, 25.75, 15.75] 
    #fractions = [0.99, 0.009999999999, 0.001]
    no_cols = np.zeros(len(ranges)) 
    while E> minE :
        i = find_in_range(ranges,getRN())
        E = E - IE[i]
        no_cols[i] += 1 
    return no_cols,E

def CalculateP(Target):
    #Lets get the data out of datframse and put them into arrays, this should be a lot quicker to deal with 
    E = Target['Energy'].as_matrix()
    P_Ps = (Target['Ps']/Target['Total']).as_matrix()           #P(n) = Qn/QT n=process
    P_Dir = (Target['Direct']/Target['Total']).as_matrix()
    P_Ann = (Target['Ann']/Target['Total']).as_matrix()
    P_Ex = ((Target['Excitation ']/1.1483*1.5)/Target['Total']).as_matrix()
    P_El = (Target['Elastic']/Target['Total']).as_matrix()
    E_Matrix = np.array(E)     #Energy 
    P_Matrix = np.cumsum((np.array([P_Ann,P_Ps,P_Dir,P_Ex,P_El])),0)    #Ranges for probabilities i.e. if we had P=[0.25,0.25,0.25,0.25] --> [0.25,0.5,0.75,1]
    EL_Matrix = np.array([-1.0, -1.0, Target['Threshold'][1], Target['Threshold'][2],0.0])  #Energy Loss 
    return E_Matrix, P_Matrix, EL_Matrix

descriptor = ['Annihilation', 'Positronium Formation', 'Direct Ionization', 'Target Excitation', 'Elastic Scattering']

def Collision(E,EM,PM,EL):
    i = find_nearest(EM, E)
    P = PM[0:5,i]
    n = find_in_range(P,getRN())
    Eprime = EL[n]
    return n, Eprime

def ReadInputDeck(filename):
    inputdeck = pd.read_csv(filename)
    atoms = inputdeck['atoms'].as_matrix()
    fraction = inputdeck['fraction'].as_matrix()
    energy_loss = inputdeck['energy loss'].as_matrix()
    nop = inputdeck['no of particles']
    ler_threshold = inputdeck['LER threshold']
    return atoms, fraction, energy_loss, nop, ler_threshold

#Lets read in the cross-section data from the CSV files and put them into dataframes
H = pd.read_csv("HydrogenXS.csv")
He= pd.read_csv("HeliumXS.csv")
H2 = pd.read_csv("MolecularHydrogenXS.csv")

#Calculate the 'elastic' scattering cross-section 
H['Elastic'] = H['Total'] - H['Ps'] - H['Direct'] - H['Ann'] - H['Excitation '] 
He['Elastic'] = He['Total'] - He['Ps'] - He['Direct'] - He['Ann'] - He['Excitation '] 
H2['Elastic'] = H2['Total'] - H2['Ps'] - H2['Direct'] - H2['Ann'] - H2['Excitation '] 

EM_H, PM_H, EL_H = CalculateP(H)
EM_He, PM_He, EL_He = CalculateP(He)
EM_H2, PM_H2, EL_H2 = CalculateP(H2)

#Vairables for the 44Sc energy distribution 
Q = 3653.3 #keV 
EMax = 1402300 #keV 
Z = 21 
r = 150e-12
T = np.linspace(0,EMax,np.abs(EMax/13.6)) * 1.602e-19 #keV
Cl = 1
num = np.nan_to_num(N(Z,T,Cl,EMax*e,r)) #Get the number distribution 
num = num / np.sum(num) #Normalize the area to 1 to get the probability
pdeq=np.cumsum(num) #Integrate to find the PDEq 

start = datetime.now()
######################################################################################
no_of_particles = 10000 # number of particles to run set to 0 to run for a time limit##
######################################################################################

particle_no = 0  #present particle 

######################################################################################
dte = date(2017,7,10)    #Set the date and time you want the run to finish here########
tme = time(11,30,00)     ##############################################################
######################################################################################
fdt = datetime.combine(dte,tme)

if no_of_particles == 0 :
    runtime = (fdt - dt.datetime.now()).total_seconds()
    no_of_particles = np.abs(10**(1.0016*np.log10(runtime)+0.707))
    print("estimated number of particles to finish at {0} is {1}".format(fdt,no_of_particles))

runtime = 10**(0.9984*np.log10(no_of_particles)-0.7059)
pft = dt.datetime.now()+dt.timedelta(seconds = runtime)

annihilation_counter = 0
positronium_counter = 0 
lowEres_counter = 0
lowEspread =[];
highEspread =[];
medEspread =[];

print("==============================================================================================")
print("============                         SIMULATION START                             ============")
print("==============================================================================================")
print("Simulation will finish after --> {0} and run approximately {1} particles".format(pft, no_of_particles))

while particle_no < no_of_particles: 
#while fdt > datetime.now():
    N = getRN() #roll the dice! 
    E = T[find_nearest(pdeq,N)]/e
    highEspread.append(E)
    atoms = ['Hydrogen', 'Helium', 'H2']
    fractions = [0.99,0.009,0.001] #Fractions H, He, H2 
    ranges = np.cumsum(fractions)  #[0.25,0.25,0.5]->[0.25,0.5,1]
    energy_loss = [13.6,25.75,15.75] #IE in eV
    stop_energy = 1000 #eV energy to stop the walk 
    
    nc, E = ion_walk(E, ranges ,energy_loss,stop_energy)     #Lets assume ionization will get you down to 1000eV 
    medEspread.append(E)
    #Now lets look at the cross-sections 
    while E>6.8 :   #Ep[H] = 6.8eV, EPs[He] = 18.95, EPs[H2] = 8.95eV
        i = find_in_range(ranges,getRN())   #in this case i will return an index 0,1,2...,n 
        
        if i==0: #H
            n, el = Collision(E,EM_H,PM_H,EL_H)
            if n == 0 :
                #print("Particle {0} ended it's path via {1} on an {2} atom at an energy of {3:.1f}eV".format(particle_no,descriptor[n],atoms[i],E))
                annihilation_counter += 1
                particle_no += 1    #Increment the particle counter 
                break
            if n == 1:
                #print("Particle {0} ended it's path via {1} on an {2} atom at an energy of {3:.1f}eV".format(particle_no,descriptor[n],atoms[i],E))
                positronium_counter += 1
                particle_no += 1    #Increment the particle counter 
                break
            else:
                if E > el:    #Only subtract the process if the energy is above the threshold
                    E = E - el
                         
        elif i==1: #He
            n, el = Collision(E,EM_He,PM_He,EL_He)
            if n == 0 :
                #print("Particle {0} ended it's path via {1} on an {2} atom at an energy of {3:.1f}eV".format(particle_no,descriptor[n],atoms[i],E))
                annihilation_counter += 1
                particle_no += 1    #Increment the particle counter 
                break
            if n == 1:
                #print("Particle {0} ended it's path via {1} on an {2} atom at an energy of {3:.1f}eV".format(particle_no,descriptor[n],atoms[i],E))
                positronium_counter += 1
                particle_no += 1    #Increment the particle counter 
                break
            else:
                if E > el:    #Only subtract the process if the energy is above the threshold
                    E = E - el
        
        elif i==2: #H2
            n, el = Collision(E,EM_H2,PM_H2,EL_H2)
            if n == 0 :
                #print("Particle {0} ended it's path via {1} on an {2} atom at an energy of {3:.1f}eV".format(particle_no,descriptor[n],atoms[i],E))
                annihilation_counter += 1
                particle_no += 1    #Increment the particle counter 
                break
            if n == 1:
                #print("Particle {0} ended it's path via {1} on an {2} atom at an energy of {3:.1f}eV".format(particle_no,descriptor[n],atoms[i],E))
                positronium_counter += 1
                particle_no += 1    #Increment the particle counter 
                break
            else:
                if E > el:    #Only subtract the process if the energy is above the threshold
                    E = E - el
        else:
            print("Error!?")
    if n> 1 :
        #print("Particle number {0} ended it's path falling below 6.8eV with an energy of {1}".format(particle_no,E))
        lowEspread.append(E)
        lowEres_counter += 1 
        particle_no += 1

#Now lets calculate the useful quantities from the simulation, plot and save them 
no_of_particles = particle_no
Ps_fraction = (positronium_counter/no_of_particles)*100.000
Ann_fraction = (annihilation_counter/no_of_particles)*100.000
LER_fraction = (lowEres_counter/no_of_particles)*100.000
stop = datetime.now()
deltat = (stop - start).total_seconds()

print("==============================================================================================")
print("============                            RESULTS                                   ============")
print("==============================================================================================")
print("after {0} particles {1:.2f}% formed Ps, {2:.2f}% annihilated and {3:.2f}% entered the low energy reservoir".format(no_of_particles,Ps_fraction,Ann_fraction,LER_fraction))
print("process completed in {0:.2f} seconds".format(deltat))


her_bins = np.linspace(0,1400000,1000)
plt.figure(num= 1, figsize=(10,10))
her_counts, her_bins, X = plt.hist(highEspread,bins = her_bins)
plt.title("Initial Energy Distribution")
plt.xlabel('Energy(eV)')
plt.ylabel('Number')
plt.savefig('OUT\\PIE.png')
plt.show()

ler_bins = np.linspace(0,6.8,100)
plt.figure(num= 1, figsize=(10,10))
ler_counts, ler_bins, X = plt.hist(lowEspread,bins = ler_bins)
plt.title("Low Energy Reservoir energy distribution")
plt.xlabel('Energy(eV)')
plt.ylabel('Number')
plt.savefig('OUT\\LERED.png')
plt.show()

plt.figure(num= 2, figsize=(10,10))
lables = 'Annihilation', 'Positronium', 'Low Energy Reservoir'
explode = (0.1,0,0)
sizes = [Ann_fraction, Ps_fraction, LER_fraction]
plt.pie(sizes, labels=lables, autopct='%1.1f%%', shadow = True, explode = explode, startangle=90)
plt.axis('equal')
plt.title("Positron Final State Distribution")
plt.savefig('OUT\\PFSD.png')
plt.show()


N = no_of_particles 
dN = np.sqrt(N)

NPs = positronium_counter
dNPs = np.sqrt(NPs) 

NAnn = annihilation_counter
dNAnn = np.sqrt(NAnn)

NLER = lowEres_counter
dNLER = np.sqrt(NLER) 

print("The abundances were:")
for xi in range(len(atoms)):
    print("{0} f={1} Ei={2}eV".format(atoms[xi],fractions[xi],energy_loss[xi]))

print("N = {0} ± {1}".format(N,dN))
print("N_Ps = {0} ± {1}".format(NPs,dNPs))
print("N_Ann = {0} ± {1}".format(NAnn,dNAnn))
print("N_LER = {0} ± {1}".format(NLER,dNLER))
print("==============================================================================================")
print("Energy (eV)\tN\tdN")

for xj in range(len(ler_counts))  : 
    print("{0}\t{1}\t{2}".format(ler_bins[xj],ler_counts[xj],np.sqrt(ler_counts[xj])))


ndt = dt.datetime.now()

filenametxt = "OUT\\OUTd{0}{1}{2}t{3}{4}{5}.txt".format(ndt.year,ndt.day,ndt.month,ndt.hour,ndt.minute,ndt.second)

file = open(filenametxt,"w")
file.write("==============================================================================================\n")
file.write("============                            RESULTS                                   ============\n")
file.write("==============================================================================================\n")
file.write("after {0} particles {1:.2f}% formed Ps, {2:.2f}% annihilated and {3:.2f}% entered the low energy reservoir\n".format(no_of_particles,Ps_fraction,Ann_fraction,LER_fraction))
file.write("process completed in {0:.2f} seconds\n".format(deltat))
file.write("==============================================================================================\n")
file.write("The abundances were:\n")
for xi in range(len(atoms)):
    file.write("{0} f={1} Ei={2}eV\n".format(atoms[xi],fractions[xi],energy_loss[xi]))
file.write("==============================================================================================\n")
file.write("N\t{0}\t{1}\n".format(N,dN))
file.write("N_Ps\t{0}\t{1}\n".format(NPs,dNPs))
file.write("N_Ann\t{0}\t{1}\n".format(NAnn,dNAnn))
file.write("N_LER\t{0}\t{1}\n".format(NLER,dNLER))
file.write("==============================================================================================\n")
file.write("LOW ENERGY\n")
file.write("==============================================================================================\n")
file.write("Energy (eV)\tN\tdN\n")
for xj in range(len(ler_counts)): 
    file.write("{0}\t{1}\t{2}\n".format(ler_bins[xj],ler_counts[xj],np.sqrt(ler_counts[xj])))
file.write("==============================================================================================\n")
file.write("HIGH ENERGY\n")
file.write("==============================================================================================\n")
file.write("Energy (eV)\tN\tdN\n")
for xk in range(len(her_counts)): 
    file.write("{0}\t{1}\t{2}\n".format(her_bins[xk],her_counts[xk],np.sqrt(her_counts[xk])))
file.close()
