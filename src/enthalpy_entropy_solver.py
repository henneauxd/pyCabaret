import numpy as np 
import _mutationpp as mpp
import rebuilding_setup as setup
import scipy
from scipy.optimize import minimize
from scipy import misc

class enthalpy_entropy_solver:
    """
    Class to create a solver for the enthalpy, entropy conservation equations.
  
    """

    def __init__(self,resmin,h,s,mix,state,name,options):
        self.resmin = resmin
        self.h = h
        self.s = s
        self.mix = mix
        self.state = state
        self.name = name
        self.options = options

    def set_resmin(self,resmin):
        """
        Function to set a new residual.

        Parameters
        ----------
        resmin : float
            Residual.

        Output
        ----------   
        self.resmin: float
            Atribute for the residual.    
        """
        self.resmin = resmin

    def func_minimize(self,var,T,p,resini,robust_choice): 
        """
        Function that computes the minimization metric of the total enthalpy-entropy equations system.

        Parameters
        ----------
        var : 1D array of size 2.
            Proportional variables for temperature and pressure.
        T : float
            Temperature.
        p: float
            Pressure.
        resini: float
            Initial residual.
        robust_choice: string
            String reflecting if the optimization problem should be solved using Newton-flavoured methods or gradient-free.

        Output
        ----------   
        metric: float or 1D array of shape 3
            Float or vector with the resulting metric.  
        """
        if robust_choice == "No":
            for i in range(len(var)):
                if var[i]<0.0:
                    return [1.0e+16]*len(var)
        else:
            for i in range(len(var)):
                if var[i]<0.0:
                    return 1.0e+16

        real = [var[0]*T,var[1]*p]

        setup.mixture_states(self.mix)[self.state].equilibrate(real[0],real[1])
        vel = 0.0
        if self.v0 !=0.:
            vel = setup.mixture_states(self.mix)[self.state].equilibriumSoundSpeed() * self.v0
        
        h_0 = setup.mixture_states(self.mix)[self.state].mixtureHMass() + (0.5*(vel**2))
        s_0 = setup.mixture_states(self.mix)[self.state].mixtureSMass()

        residual = [(h_0-self.h)/self.h, (s_0-self.s)/self.s]

        if robust_choice == "No":
            metric = [np.linalg.norm(residual[i]) for i in range(len(residual))]
        else:
            metric = np.linalg.norm(residual)/resini
        return metric

    def jacobian(self,var,T,p,resini):
        """
        Function that computes the Jacobian matrix.

        Parameters
        ----------
        var : 1D array of size 2.
            Proportional variables for temperature and pressure.
        T : float
            Temperature.
        p: float
            Pressure.
        resini: float
            Initial residual.

        Output
        ----------   
        jacob: ndarray or matrix of shape (2,2)
            Jacobian matrix.    
        """
        jacob = scipy.optimize.approx_fprime(var,self.func_minimize,1.0e-10,T,p,resini)
        return jacob

    def solution(self,T,p,v_0=0.):
        """
        Function that computes the solution of the total enthalpy-entropy equations system.

        Parameters
        ----------
        T : float
            Temperature.
        p: float
            Pressure.
        v_0: float
            Velocity. Set to 0 as default.

        Output
        ----------   
        1D array of shape 3
            Vector with the resulting T,p and v.  
        """

        ## Initial conditions ##
        var =[self.options["temperature"],self.options["pressure"]] #[10.,100.]
        self.v0=v_0
        resini = 1.0
        resini = self.func_minimize(var,T,p,resini,self.options["robust"])

        bnds = ((1.0, None), (1.0, None)) 
        options={'maxiter': None}
        if self.options["robust"] == "Yes":
            result = scipy.optimize.minimize(self.func_minimize,var,args=(T,p,resini,self.options["robust"]),method='Powell',tol=self.resmin,bounds=bnds,options=options)

            if result.success == False:
                print("Warning: convergence not guaranteed for"+' '+self.name)

        else:
            result = scipy.optimize.root(self.func_minimize,var,args=(T,p,resini,self.options["robust"]),tol=self.resmin)

            if result.success == False:
                print("Warning: convergence not guaranteed for"+' '+self.name)

        vel = 0.0
        if self.v0 !=0.:
            setup.mixture_states(self.mix)[self.state].equilibrate(result.x[0]*T,result.x[1]*p)
            vel = setup.mixture_states(self.mix)[self.state].equilibriumSoundSpeed() * self.v0

        return result.x[0]*T,result.x[1]*p,vel