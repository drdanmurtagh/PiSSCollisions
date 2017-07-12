# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:10:05 2017

@author: dan
"""

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
import scipy.constants as constants

amu = 1.66054e-27
alpha = constants.alpha
hbar = constants.hbar
c = constants.speed_of_light
k = constants.k
pi = constants.pi
m = constants.electron_mass
e = constants.electron_volt
epsilon0 = constants.epsilon_0

##################################################################################################
# Non-physics related functions
#
##################################################################################################

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

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

def getRN():
    R = np.float64(rnd.getrandbits(256)/np.float64(2.0**256.0))
    return R 


##################################################################################################
# electron scattering related functions
#
##################################################################################################
def rndMB_elec(T):
    Emax = 10*(k*T/e)   #set the energy range to consider 10kT is a bit arbuitary but seems to work for all T 
    energy_eV = np.linspace(0,Emax,1000)    
    v = np.sqrt(2*energy_eV*e/m)    #convert energy to velocity 
    f = np.sqrt((m/(2*pi*k*T))**3)*4*pi*v**2*np.exp((-m*v**2)/(2*k*T))  #MB distribution function 
    f = f/np.sum(f)
    pdeq = np.cumsum(f)
    E = energy_eV[find_nearest(pdeq,getRN())]
    return E 

def AnglePDEQ(energy_eV):       #generates a PDEq from the positron-electron angularly differential cross-section 
    E = energy_eV * e           #Let's make sure we stick to eV when passing arguments 
    theta = np.logspace(-4,np.log10(3.14),180)  #n.b. infinite at theta = 0 
    dcs = (e**2/(4*pi*epsilon0))**2 * 1/(16*E**2*np.sin(theta/2)**4)
    norm = np.sum(dcs)
    pdeq = np.cumsum(dcs/norm)
    return theta, pdeq          #return the list of angles and the PDE

def velocity(E):
    return np.sqrt(2*E*e/m)     #I think it's obvious what this does ! 

def rndCollision(E,T):
    E_electron = rndMB_elec(T)
    v_electron = velocity(E_electron)
    v_positron = velocity(E)
    angles, a_PDEQ = AnglePDEQ(E)
    angle = angles[find_nearest(a_PDEQ,getRN())]
    if(v_positron > v_electron):                    #bit of a fudge, if the positron has more energy transfer to the electron
        Q = (m/2) * v_positron**2 * np.sin(angle/2)
    else:                                           #else transfer from the electorn to the positron
        Q = -(m/2) * v_electron**2 * np.sin(angle/2)
    return Q/e  #Let's stick to those electron volts! 

def CalculateP_Electrons():
    E = Electron['Energy'].as_matrix()
    P_Ann = (Electron['Ann']/Electron['Total']).as_matrix()
    P_RR =  (Electron['RR']/Electron['Total']).as_matrix()
    P_ET = (Electron['ET']/Electron['Total']).as_matrix()
    E_Matrix = np.array(E)
    P_Matrix = np.cumsum((np.array([P_Ann,P_RR,P_ET])),0)    #Ranges for probabilities i.e. if we had P=[0.25,0.25,0.25,0.25] --> [0.25,0.5,0.75,1]
    return E_Matrix, P_Matrix 

def CalculateDistance_Electrons(Density_cm3):
    rho = Density_cm3 * (100*100*100)
    D_Ann = (1/(Electron['Ann']*1e-20*rho)).as_matrix()
    D_RR = (1/(Electron['RR']*1e-20*rho)).as_matrix()
    D_ET = (1/(Electron['ET']*1e-20*rho)).as_matrix()
    D_Matrix = np.array([D_Ann,D_RR,D_ET])
    return D_Matrix

Electron_Descriptor = ['Direct Annihilation', 'Radiative Recombination', 'Energy Transfer']
    
##################################################################################################
# Atomic scattering physics related functions
#
##################################################################################################

def CalculateP(Target):
    #Lets get the data out of datframse and put them into arrays, this should be a lot quicker to deal with 
    E = Target['Energy'].as_matrix()
    P_Ps = (Target['Ps']/Target['Total']).as_matrix()           #P(n) = Qn/QT n=process
    P_Dir = (Target['Direct']/Target['Total']).as_matrix()
    P_Ann = (Target['Ann']/Target['Total']).as_matrix()
    P_Ex = ((Target['Excitation '])/Target['Total']).as_matrix()
    P_El = (Target['Elastic']/Target['Total']).as_matrix()
    E_Matrix = np.array(E)     #Energy 
    P_Matrix = np.cumsum((np.array([P_Ann,P_Ps,P_Dir,P_Ex,P_El])),0)    #Ranges for probabilities i.e. if we had P=[0.25,0.25,0.25,0.25] --> [0.25,0.5,0.75,1]
    EL_Matrix = np.array([-1.0, -1.0, Target['Threshold'][1], Target['Threshold'][2],0.0])  #Energy Loss 
    return E_Matrix, P_Matrix, EL_Matrix

descriptor = ['Annihilation', 'Positronium Formation', 'Direct Ionization', 'Target Excitation', 'Elastic Scattering', 'Electron Scattering']

def CalculateDistance(Density_cm3, Target):
    rho = Density_cm3 * (100*100*100)
    D_Ps = (1/(Target['Ps']*1e-20*rho)).as_matrix()
    D_Dir = (1/(Target['Direct']*1e-20*rho)).as_matrix()
    D_Ann = (1/(Target['Ann']*1e-20*rho)).as_matrix()
    D_Ex = (1/(Target['Excitation ']*1e-20*rho)).as_matrix()
    D_El = (1/(Target['Elastic']*1e-20*rho)).as_matrix()
    D_Matrix = np.array([D_Ann,D_Ps,D_Dir,D_Ex,D_El])
    return D_Matrix


def Collision(E,EM,PM,EL):
    i = find_nearest(EM, E)
    P = PM[0:5,i]
    n = find_in_range(P,getRN())
    Eprime = EL[n]
    return n, Eprime

##################################################################################################
# Main code
#
############################################################################################

#Lets read in the cross-section data from the CSV files and put them into dataframes
H = pd.read_csv("HydrogenXS.csv")
He= pd.read_csv("HeliumXS.csv")
H2 = pd.read_csv("MolecularHydrogenXS.csv")
Electron = pd.read_csv("Electron.csv")

#Calculate the 'elastic' scattering cross-section 
H['Elastic'] = H['Total'] - H['Ps'] - H['Direct'] - H['Ann'] - H['Excitation '] 
He['Elastic'] = He['Total'] - He['Ps'] - He['Direct'] - He['Ann'] - He['Excitation '] 
H2['Elastic'] = H2['Total'] - H2['Ps'] - H2['Direct'] - H2['Ann'] - H2['Excitation '] 

EM_H, PM_H, EL_H = CalculateP(H)
EM_He, PM_He, EL_He = CalculateP(He)
EM_H2, PM_H2, EL_H2 = CalculateP(H2)
EM_E, PM_E = CalculateP_Electrons()

T = np.linspace(0,6.8,1000) * 1.602e-19 #Energy range 0-6.8eV 
num =  0.3*T #Get the number distribution from the fit  
num = num / np.sum(num) #Normalize the area to 1 to get the probability
pdeq=np.cumsum(num) #Integrate to find the PDEq 

start = datetime.now()
######################################################################################
no_of_particles = 100 # number of particles to run set to 0 to run for a time limit##
######################################################################################

particle_no = 0  #present particle 

######################################################################################
dte = date(2017,7,10)    #Set the date and time you want the run to finish here########
tme = time(11,30,00)     ##############################################################
######################################################################################
fdt = datetime.combine(dte,tme)

if no_of_particles == 0 :
    runtime = (fdt - dt.datetime.now()).total_seconds()
    no_of_particles = np.abs((runtime-11.72)/4.66)
    print("estimated number of particles to finish at {0} is {1}".format(fdt,no_of_particles))

runtime = (4.66*no_of_particles)+11.72
pft = dt.datetime.now()+dt.timedelta(seconds = runtime)

annihilation_counter = 0
positronium_counter = 0 
lowEres_counter = 0

lowEspread =[];
highEspread =[];
distanceRange =[];

print("==============================================================================================")
print("============                         SIMULATION START                             ============")
print("==============================================================================================")
print("Simulation will finish after --> {0} and run approximately {1} particles".format(pft, no_of_particles))

#These vairable define the region of space 
region_temp = 100
ionization_fraction = 0.001
atoms = ['Hydrogen', 'Helium', 'H2', 'electron']
fractions = [0.99,0.009,0.001] #Fractions H, He, H2 
ranges = np.cumsum(fractions)  #[0.25,0.25,0.5]->[0.25,0.5,1]
energy_loss = [13.6,25.75,15.75] #IE in eV

stop_energy = 1000 #eV energy to stop the walk 

DM_H = CalculateDistance(fractions[0],H)
DM_He = CalculateDistance(fractions[1],He)
DM_H2 = CalculateDistance(fractions[2],H2)
DM_E = CalculateDistance_Electrons(ionization_fraction)

while particle_no < no_of_particles: 
#while fdt > datetime.now():
    distance = 0    # This is the distance counter for each particle 
    
    neutrals = 0    #Counter to keep track of the probabilities
    charged = 0 
    total_col=0
    H_count = 0
    He_count = 0 
    H2_count = 0
    
    E = T[find_nearest(pdeq,getRN())]/e
    highEspread.append(E)    
    #Now lets look at the cross-sections 
    while E>0 :   #run until the positron annihilates on something
        total_col +=1
        i = find_in_range(ranges,getRN())   #in this case i will return an index 0,1,2...,n  
        if i==0: #H
            H_count +=1
            electron_check = getRN() #lets see if this is an electron based on the ionization_fraction fi=ne/(nH+ne)
            if electron_check <= ionization_fraction:   #The particle is an electron
                i = 3 #This is just for informaiton, this interaction happened with an electron     
                index = find_nearest(EM_E, E)
                P = PM_E[0:3,index]
                n = find_in_range(P,getRN())
                distance += DM_E[n,find_nearest(EM_E,E)]
                if n == 0 :
                    #print("Particle {0} ended it's path via {1} with an {2} at an energy of {3:.1f}eV".format(particle_no,electron_descriptor[n],"electron",E))
                    annihilation_counter += 1 
                    particle_no += 1
                    break
                if n == 1: 
                    #print("Particle {0} ended it's path via {1} with an {2} at an energy of {3:.1f}eV".format(particle_no,electron_descriptor[n],"electron",E))
                    positronium_counter += 1 
                    particle_no += 1 
                else : 
                    deltaE = rndCollision(E,region_temp)
                    E = E - deltaE
                    #print("I hit an electron and lost {0}eV".format(deltaE))
                    charged += 1
                    n=5 #set n to a new value to represent electron collisions
            else:   #it's a neutral collision 
                neutrals += 1             
                n, el = Collision(E,EM_H,PM_H,EL_H)
                distance += DM_H[n,find_nearest(EM_H,E)]
                if n == 0:
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
            He_count += 1 
            n, el = Collision(E,EM_He,PM_He,EL_He)
            distance += DM_He[n,find_nearest(EM_He,E)]
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
            H2_count += 1 
            n, el = Collision(E,EM_H2,PM_H2,EL_H2)
            distance += DM_H2[n,find_nearest(EM_H2,E)]
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
    lowEspread.append(E)
    distanceRange.append(distance)
    print("#{0} tc:{1}\t e-:{2}\t H:{3}\t He:{4}\t H2:{5}\t {6} on {7} d:{8:.1f}Pc".format(particle_no,total_col,charged,H_count, He_count, H2_count,descriptor[n], atoms[i], (distance/3.086e16)))
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

her_bins = np.linspace(0,6.8,100)
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
plt.title("Final Energy distribution")
plt.xlabel('Energy(eV)')
plt.ylabel('Number')
plt.savefig('OUT\\LERED.png')
plt.show()

distanceRangePc = np.array(distanceRange)/3.086e16/1000
distance_bins = np.linspace(0,100,100)
plt.figure(num= 1, figsize=(10,10))
distance_counts, distance_bins, X = plt.hist(distanceRangePc,bins = distance_bins)
plt.title("Distance distribution")
plt.xlabel('Distance (kPc)')
plt.ylabel('Number')
plt.savefig('OUT\\PIE.png')
plt.show()

ndt = dt.datetime.now()

filenametxt = "OUT\\OUTd{0}{1}{2}t{3}{4}{5}.txt".format(ndt.year,ndt.day,ndt.month,ndt.hour,ndt.minute,ndt.second)
file = open(filenametxt,"w")
file.write("==============================================================================================\n")
file.write("============                            RESULTS                                   ============\n")
file.write("==============================================================================================\n")
file.write("==============================================================================================\n")
file.write("INITIAL ENERGY DISTRIBUTION\n")
file.write("==============================================================================================\n")
file.write("Energy (eV)\tN\tdN\n")
for xj in range(len(ler_counts)): 
    file.write("{0}\t{1}\t{2}\n".format(ler_bins[xj],ler_counts[xj],np.sqrt(ler_counts[xj])))
file.write("==============================================================================================\n")
file.write("FINAL ENERGY DISTRIBUTION\n")
file.write("==============================================================================================\n")
file.write("Energy (eV)\tN\tdN\n")
for xk in range(len(her_counts)): 
    file.write("{0}\t{1}\t{2}\n".format(her_bins[xk],her_counts[xk],np.sqrt(her_counts[xk])))
file.write("==============================================================================================\n")
file.write("DISTANCE DISTRIBUTION\n")
file.write("==============================================================================================\n")
file.write("Energy (eV)\tN\tdN\n")
for xi in range(len(distance_counts)): 
    file.write("{0}\t{1}\t{2}\n".format(distance_bins[xi],distance_counts[xi],np.sqrt(distance_counts[xi])))
file.close()
