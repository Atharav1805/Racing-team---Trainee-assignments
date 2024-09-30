import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm 
#You are tasked with coding a scenario where a food delivery robot moves from Aroma Dhaba (origin) to Hostel 3 in a 1-dimensional motion setting. 
#The robot has sensors onboard that detect “Hostel” upon reaching any of the hostels (Hostel 1, 2, or 3, locations of which are known) and also
#senses the control command (speed). These sensors have some level of noise, and you need to use Kalman filters to estimate the robot's state accurately on the map. 
#The goal is to ensure successful parcel delivery by localizing the robot's position accurately.

landmarks=[25,45,60] #position of the hostels

#initialising the matrices  
A=1 #depicts what would have happened if the control commands were zero
B=0.1 #depicts time dt
C=1 #to transform a predicted position to the measurement of that position


#Problem says: You are able to take a measurement at only 3 instants (on reaching a hostel)
#So correct the predicted values only at 3 instances
#for all other instances set z=None and do not correct the predicted position/mean
#For the code I only have predicted values and no means to know when my bot reaches ony hostel
#SO I make the following assumption:
#My vel is in range(5,10) so avg vel=7.5 with each step I am moving 0.75 m 
#so I assume I physically reach hostel 1 at 25/0.75 th iteration and which is somewhat good as the predicted mean is also close to it
#So I take the measurement z only at 33rd, 60th, 80th iteration 


'''
HAVE ARBITRARILY USED THE VALUES OF FOLLOWING PARAMETERS(TO MATCH THE PLOTS GIVEN IN PROBLEM):
1. R
2. Q
3. Variance of the measurement values to plot the measured wala plot
'''


#defining lists for plots
predmean,predvar,obsmean,obsvar,corrmean,corrvar=list(),list(),list(),list(),list(),list()


def Kalman_filter(mean,var,u,z):
    R=random.uniform(2,3) #Noise #ARBITRARILY SET 
    Q=random.uniform(0,0.25) #Noise #ARBITRARILY SET 
    
    #predicted means
    mean_pred=A*mean+B*u
    var_pred=A*var*A+Q
    
    if(mean<60):
        if(z!=None):
            #calculating Kalman Gain K
            K=(var_pred*C)/(C*var_pred*C+R) 
            #correcting the means
            mean_final=mean_pred+K*(z-C*mean_pred)
            var_final=(1-K*C)*var_pred
            
            predmean.append(mean_pred)
            predvar.append(var_pred)
            obsmean.append(z)
            obsvar.append(2)#ARBITRARILY SET TO MATCH PLOTS IN PS 
            corrmean.append(mean_final)
            corrvar.append(var_final)

        else:
            mean_final=mean_pred
            var_final=var_pred
    else:
        mean_final=60
        var_final=0
        print("Reached destination")
    # print(f"Current mean is {mean_final}")
    return mean_final,var_final




#initialising the values
mean=0
var=0
for i in range(100):
    u=random.uniform(5,10)
    # print(f"Velocity used is {u}")
    if(i==33):
        z=landmarks[0]
        print(f"Physically at {z} predicted at {mean}")
    elif(i==60):
        z=landmarks[1]
        print(f"Physically at {z} predicted at {mean}")
        # print(f"Reached 2nd landmark at iteration {i}")
    elif(i==80):
        z=landmarks[2]
        print(f"Physically at {z} predicted at {mean}")
        # print(f"Reached 3rd landmark at iteration {i}")
    else:
        z=None
    mean,var=Kalman_filter(mean,var,u,z)
    print(mean,var)

# fig,axs=plt.subplots(3,1,figsize=(32,30))

# Plot between 0 and 100 with 0.01 steps. 
x_axis = np.arange(0, 100, 0.01)

for i in range(3):
    plt.plot(x_axis,norm.pdf(x_axis,predmean[i],predvar[i]),c='green',label="Predicted")
    plt.plot(x_axis,norm.pdf(x_axis,obsmean[i],obsvar[i]),c='blue',label="Measured")
    plt.plot(x_axis,norm.pdf(x_axis,corrmean[i],corrvar[i]),c='red',label="Corrected")
    plt.xticks(list(np.linspace(0,100,21)),fontsize=7)
    plt.legend()
    plt.show()
    plt.waitforbuttonpress()
