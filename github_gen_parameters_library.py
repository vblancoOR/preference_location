"""
This file contains the functions to generate the parameters involved 
in the location with preferences paper by the authors:
Victor Blanco, Ricardo Gázquez and Marina Leal.

The paper can be found in open access in Arxiv.
"""

import numpy as np
import math as mth


def gen_parameters(A,R,Norms2,ff,sed):
    """
    It generates the parameters for each region with a given seed.
    
    Parameters
    ----------
    A      : List of points with region's centers.
    R      : List of radii.
    Norms2 : List of norms within the region.
    ff     : string with the preference function within the region.    
    sed    : int with the random seed.
    """
    
    np.random.seed(sed)
    n = len(A)
    N = range(n)
    L = range(2)
    
    if ff == "L":
        gamma0 = np.random.randint(1,11)
        gamma = []
        for i in N:
            g = []
            for l in L:                
                g.append(round((-1)**(np.random.randint(0,2))*(np.random.randint(1,11)),2))                
            gamma.append(g)
        F = ["L",gamma,gamma0]
    elif ff == "D":
        BB = read_centers(n,sed)        
        lv = []
        for i in N:
            kk = len(BB[i])
            ll = []
            for i in range(kk):
                ll.append(1/kk)
            lv.append(ll)
            
        F = ["D",lv,BB]
        
    elif ff in ["CES","CD","GL"]:
        delta = {}
        CS = []
        beta = []
        
        #### We fix the number of factors equals to 2.    
        nfactors = 2
        Nfactors = range(nfactors)

        for i in N:                     
            beta.append([0.5,0.5])
                
            CCS = []
            for j in Nfactors:
                cs = circle_section(A[i], R[i], Norms2[i],sed,False) 
                CCS.append(cs)
            CS.append(CCS)                   
            
            for j in Nfactors:
                for l in range(CS[i][j][1]):
                    #### Random value
                    delta[i,j,l] = 2*round(np.random.rand(),4)
        
        if ff == "CES":
            tau = []
            for i in N:                                    
                tau.append(0.5)
                
            F =  ["CES",beta,CS,delta,tau]
        elif ff == "CD":
            F = ["CD",beta,CS,delta]
        elif ff == "GL":
            F = ["GL",beta,CS,delta]
            
    return F

def read_centers(n,sed):
    """
    Function to read the Centers given by the Distance preference function.
    
    """
    
    file = "BlobCenters-%d-sed%d.txt"%(n,sed)
        
    data = np.genfromtxt(file,  delimiter=",", unpack=False)
    n = len(data[:,0])
    BB = []
    
    for i in range(n):
        BB2 = []
        for j in range(len(data[i])):
            if j%2 == 0 and data[i][j] != 1000:                
                BB2.append([data[i][j],data[i][j+1]])
        BB.append(BB2)
    
    return BB

def read_data(n,sed,nrs):
    
    file = "BlobData-%d-sem%d.txt"%(n,sed)
    
    data = np.genfromtxt(file,  delimiter=",", unpack=False, usecols=range(6))
    n = len(data[:,0])
    A = np.array([[data[j,0], data[j,1]] for j in range(n)])
    w = np.array([data[j,2] for j in range(n)])
    R = np.array([data[j,3] for j in range(n)])
    if nrs == 1:
        Norms1 = np.array([1 for j in range(n)])
        Norms2 = np.array([1 for j in range(n)])
    elif nrs == 2:
        Norms1 = np.array([2 for j in range(n)])
        Norms2 = np.array([2 for j in range(n)])
    else:
        Norms1 = np.array([int(data[j,4]) for j in range(n)])
        Norms2 = np.array([int(data[j,5]) for j in range(n)])
    
    return A,w,R,Norms1,Norms2

def circle_section(S,R,Norms,sed = 0):
    """
    This function generates sections within a circumference RANDOMLY.  
    Given the fixed number of 2 lines, two random points are generated within 
    the circumference to define the lines, resulting in random sections.  
        
    Several assumptions:  
      - The lines intersect at most in one point. That is, no overlapping lines 
        are allowed. However, parallel lines are possible.  
      - There are 8 possible distinct regions. It is not possible to uniquely 
        determine the sections. Duplicate points may appear in different regions.  
      - All randomization is uniform.  
        
    Parameters  
    ----------  
    S : np.array  
        Center of the circumference.  
    R : float  
        Radius of the circumference.  
    Norms : int  
        Norm used to generate the circumference.  
    
    Returns  
    -------  
    Returns a number and 9 lists.  
    nlines : Number of lines generating the sections.  
    Points : Points defining the lines for the sections.  
    Reg1 = [Reg1_x, Reg1_y] Coordinates x and y of the points in region 1.  
    
    There are 4 total regions, corresponding to the possible intersections of 2 lines within the circumference.  
    """
   
    np.random.seed(sed)
    #### This is to generate the points inside the circumference.    
    nn = 150    
    XZ,YZ = meshgrid_lp_ball(S,R,Norms,nn)
    
    #### We generate the sections.    
    nlines, Points, MM, NN = gen_lines(S, R, Norms, sed)
        
    Reg1_x,Reg2_x,Reg3_x,Reg4_x = [],[],[],[]
    Reg1_y,Reg2_y,Reg3_y,Reg4_y = [],[],[],[]
    for j in range(nn):
        for k in range(nn):
            if nlines == 2:
                #### Para 2 rectas.
                Z0 = YZ[j,k]-MM[0]*XZ[j,k]
                Z1 = YZ[j,k]-MM[1]*XZ[j,k]
                
                ### Comprobamos para cada hiperplano si es <= que el termino indep.
                if Z0 <= NN[0] and Z1 <= NN[1]:
                    Reg1_x.append(XZ[j,k])
                    Reg1_y.append(YZ[j,k])
                if Z0 <= NN[0] and Z1 >= NN[1]:
                    Reg2_x.append(XZ[j,k])
                    Reg2_y.append(YZ[j,k])
                if Z0 >= NN[0] and Z1 <= NN[1]:
                    Reg3_x.append(XZ[j,k])
                    Reg3_y.append(YZ[j,k])
                if Z0 >= NN[0] and Z1 >= NN[1]:
                    Reg4_x.append(XZ[j,k])
                    Reg4_y.append(YZ[j,k])
            
    Reg1, Reg2, Reg3, Reg4 = [Reg1_x,Reg1_y],[Reg2_x,Reg2_y],[Reg3_x,Reg3_y],[Reg4_x,Reg4_y]
    nsections = count_sections(Reg1, Reg2, Reg3, Reg4)
    
    return nlines,nsections,Points,MM,NN,Reg1, Reg2, Reg3, Reg4

def meshgrid_lp_ball(S, radius, norm, num_points=500):
    """
    This function generates a mesh of points inside a ball of norm Lp and radius

    Parameters
    ----------
    S : List
        Center of the ball
    radius : float
        Radius of the Lp-ball.
    p : int
        Norm Lp
    num_points : int, optional
        Number of points inside of the ball. The default is 500.

    """
    theta= np.arange(0.0,2.0*mth.pi,2.0*mth.pi/num_points)        
    rr= np.arange(0.0,radius, radius/num_points)
    A, B = np.meshgrid(rr, theta)
    if norm == 2:
        X=S[0]+A*(np.cos(B)**(2/norm))
        Y=S[1]+A*(np.sin(B)**(2/norm))    
    else:
        kk1 = [[mth.copysign(1, np.cos(B[i][j])) for j in range(len(B[i]))] for i in range(len(B))]
        kk2 = [[mth.copysign(1, np.sin(B[i][j])) for j in range(len(B[i]))] for i in range(len(B))]
        X=S[0]+A*kk1*(abs(np.cos(B))**(2/norm))
        Y=S[1]+A*kk2*(abs(np.sin(B))**(2/norm))
    return np.array(X), np.array(Y)

def gen_lines(S,R,Norms,sed):
    """
    This function defines the sections of a circumference.

    Parameters  
    ----------  
    S : np.array  
        Center of the circumference.  
    R : float  
        Radius of the circumference.  
    
    Returns  
    -------  
    nlines : int  
        Number of lines generating the sections.  
    Points : list  
        Points that define the lines.  
    MM : list  
        List of the M coefficients for the lines in the form:  
            y = M*x + N  
    NN : list  
        List of the N coefficients for the lines in the form:  
            y = M*x + N  
    """
    np.random.seed(sed)

    nlines = 2                
    #### Two random points to generate a line.
    Points = []        
    kk = {}
    for j in range(nlines):
        k1a = 0
        k2a = 0
        while k1a == k2a:
            kk[j] = np.random.randint(0,361)
            if j != 0:
                while abs(kk[j]-kk[j-1]) <= 45:                    
                    kk[j] = np.random.randint(0,361)
            th1 = mth.radians(kk[j])
            th2 = mth.radians(kk[j]+180)
            k1a,k2a = round(S[0] + R*mth.copysign(1,mth.cos(th1))*(abs(mth.cos(th1))**(2/Norms)),4), round(S[0] + R*mth.copysign(1,mth.cos(th2))*(abs(mth.cos(th2))**(2/Norms)),4)
            k1b,k2b = round(S[1] + R*mth.copysign(1,mth.sin(th1))*(abs(mth.sin(th1))**(2/Norms)),4), round(S[1] + R*mth.copysign(1,mth.sin(th2))*(abs(mth.sin(th2))**(2/Norms)),4)
        
        Points.append([[k1a,k1b],[k2a,k2b]])
    
    #### Lines
    MM,NN = [],[]
    for j in range(len(Points)):        
        m,n = fun_lines(Points[j][0], Points[j][1])
        MM.append(round(m,3))
        NN.append(round(n,3))
        
    return nlines,Points,MM,NN

def count_sections(Reg1,Reg2,Reg3,Reg4):
    nsections = 0
    if not np.array_equal(Reg1,[[],[]]):
        nsections+=1
    if not np.array_equal(Reg2,[[],[]]):
        nsections+=1
    if not np.array_equal(Reg3,[[],[]]):                        
        nsections+=1
    if not np.array_equal(Reg4,[[],[]]):
        nsections+=1    
    return nsections

def fun_lines(point1,point2):

    m = (point1[1]-point2[1])/(point1[0]-point2[0])
    n = point1[1]-m*point1[0]

    return m,n
    