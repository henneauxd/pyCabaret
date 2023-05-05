
import scipy
from scipy.optimize import minimize
import rebuilding_setup as setup
from reservoir import reservoir
from forward import forward
import math as math
import numpy as np


class LHTSInitializer:
    def __init__(self, gamma, Cp, M_inf, p_stag, H_stag):
        self.gamma = gamma
        self.Cp = Cp
        self.M_inf = M_inf
        self.p_stag = p_stag
        self.H_stag = H_stag
        self.R = ((gamma-1)/gamma) * Cp

    def staticPressureRatio(self):
        return (2.0*self.gamma*self.M_inf**2 - (self.gamma-1))/(self.gamma+1)
    
    def staticTemperatureRatio(self):
        num = (2.0*self.gamma*self.M_inf**2 - (self.gamma-1)) * ((self.gamma-1)*self.M_inf**2 + 2)
        den = ((self.gamma+1)**2)*(self.M_inf**2)
        return num/den
    
    def postShockMach(self):
        M_post_2 = ((self.gamma-1)*self.M_inf**2 + 2.0) / (2.0*self.gamma*self.M_inf**2 - (self.gamma-1))
        return math.sqrt(M_post_2)
    
    def totalOverStaticPressure(self,Mach):
        term = 1.0 + 0.5*(self.gamma-1)*Mach**2
        expo = (self.gamma/(self.gamma-1))
        return math.pow(term,expo)
    
    def totalOverStaticTemperature(self,Mach):
        term = 1.0 + 0.5*(self.gamma-1)*Mach**2
        return term
    
    def totalTemperature(self):
        return self.H_stag / self.Cp
    
    def soundSpeed(self,T):
        return math.sqrt(self.gamma*self.R*T)

    def getFreeStreamPressure(self):
        M_post = self.postShockMach()
        PtOverP_post = self.totalOverStaticPressure(M_post)
        Ppost_over_Ppre = self.staticPressureRatio()
        return self.p_stag / (PtOverP_post * Ppost_over_Ppre)
    
    def getFreeStreamTemperature(self):
        M_post = self.postShockMach()
        TtOverT_post = self.totalOverStaticTemperature(M_post)
        Tpost_over_Tpre = self.staticTemperatureRatio()
        Tt = self.totalTemperature()
        return Tt / (TtOverT_post * Tpost_over_Tpre)
    
    def getFreeStreamVelocity(self):
        T_inf = self.getFreeStreamTemperature()
        c_inf = self.soundSpeed(T_inf)
        return self.M_inf * c_inf
    
    def getPostShockStaticPressure(self):
        P_pre = self.getFreeStreamPressure()
        Ppost_over_Ppre = self.staticPressureRatio()
        return P_pre * Ppost_over_Ppre
    
    def getPostShockStaticTemperature(self):
        T_pre = self.getFreeStreamTemperature()
        Tpost_over_Tpre = self.staticTemperatureRatio()
        return T_pre * Tpost_over_Tpre
    
    def getPostShockVelocity(self):
        M_post = self.postShockMach()
        T_post = self.getPostShockStaticTemperature()
        c_post = self.soundSpeed(T_post)
        return M_post * c_post
    


class LHTSSolver:
    def __init__(self, lhts_beta, lhts_h, lhts_p, fixed_Mach, ic_freestream, ic_post, ic_temp_stag, mix, options, print_info, T_w,pr,L,resmin, T_stag = 0.0):
        self.lhts_beta = lhts_beta
        self.lhts_h = lhts_h
        self.lhts_p = lhts_p
        self.fixed_Mach = fixed_Mach
        self.ic_freestream = ic_freestream
        self.ic_post = ic_post
        self.ic_temp_stag = ic_temp_stag
        self.mixs = setup.mixture_states(mix)
        self.options = options
        self.print_info = print_info
        self.T_w = T_w
        self.pr = pr
        self.L = L
        self.resmin = resmin
        self.T_stag = T_stag
            
    def prepost_shock_system(self,vars,method_choice):
        p_2 = vars[0]
        T_2 = vars[1]
        u_2 = vars[2]
        p_inf = vars[3]
        T_inf = vars[4]

        self.mixs["free_stream"].equilibrate(p_inf,T_inf)
        self.mixs["post_shock"].equilibrate(p_2,T_2)

        c_inf = self.mixs["free_stream"].equilibriumSoundSpeed()    
        u_inf = self.fixed_Mach * c_inf

        rho_2 = self.mixs["post_shock"].density()
        rho_inf = self.mixs["free_stream"].density()
        h_2 = self.mixs["post_shock"].mixtureHMass()
        h_inf = self.mixs["free_stream"].mixtureHMass()
        s_2 = self.mixs["post_shock"].mixtureSMass()
        # s_inf = self.mixs["free_stream"].mixtureSMass()

        s_e = self.mixs["total"].mixtureSMass()

        eqs = [0.0] * 5

        # Rankine-Hugoniot across the shock
        eqs[0] = rho_inf*u_inf - rho_2*u_2
        eqs[1] = rho_inf*u_inf**2 + p_inf - rho_2*u_2**2 - p_2
        eqs[2] = h_inf + 0.5*u_inf**2 - h_2 - 0.5*u_2**2

        # Conservation of total enthalpy and entropy along the stagnation line from the post-shock to the stagnation point
        eqs[3] = self.lhts_h - h_2 - 0.5*u_2**2
        eqs[4] = s_e - s_2

        # print("-> eqs[0] = %2.16f"%eqs[0])
        # print("-> eqs[1] = %2.16f"%eqs[1])
        # print("-> eqs[2] = %2.16f"%eqs[2])
        # print("-> eqs[3] = %2.16f"%eqs[3])
        # print("-> eqs[4] = %2.16f"%eqs[4])
        # print("")

        if method_choice == "Root":
            res_norm = [np.linalg.norm(eqs[i]) for i in range(5)]
        else:
            res_norm = np.linalg.norm(eqs)

        return res_norm
    
    def solve_prepost_shock_system(self,toler):
        opt = self.options["options"]["total"]
        ic_list = [self.ic_post,self.ic_freestream]
        ic = [item for sublist in ic_list for item in sublist] 
        # resini = self.prepost_shock_system(ic,0.0,opt["robust"])

        bnds = ((1.0, None), (1.0, None)) 
        options={'maxiter': self.options["maxiter"]}

        if self.options["method"] == "Root":
            result = scipy.optimize.root(self.prepost_shock_system,ic,args=(self.options["method"]),method='broyden2',tol=self.options["residual"],options=options)
        elif self.options["method"] == "Hybrid":
            result = scipy.optimize.minimize(self.prepost_shock_system,ic,args=(self.options["method"]),method="L-BFGS-B",tol=1.0e-03,options=options)
            ic = result.x
            result = scipy.optimize.root(self.prepost_shock_system,ic,args=(self.options["method"]),tol=self.options["residual"])
        else:
            result = scipy.optimize.minimize(self.prepost_shock_system,ic,args=(self.options["method"]),method=self.options["method"],tol=self.options["residual"],options=options)

        print(result.message)
        print("Residual value = ", result.fun)

        p_inf = result.x[3]
        T_inf = result.x[4]
        self.mixs["free_stream"].equilibrate(p_inf,T_inf)
        c_inf = self.mixs["free_stream"].equilibriumSoundSpeed()    
        u_inf = self.fixed_Mach * c_inf

        return T_inf,p_inf,u_inf
    
    def solve_reservoir_system(self,T_inf,p_inf,u_inf,toler):
        h_inf = self.mixs["free_stream"].mixtureHMass() + (0.5*u_inf**2)
        s_inf = self.mixs["free_stream"].mixtureSMass()

        T_0,p_0,v_0 = reservoir(T_inf,p_inf,h_inf,s_inf,toler,self.mixs["reservoir"],"reservoir",self.options["options"]["reservoir"])
        return T_0,p_0
    
    def temp_stagnation_system(self,T_e):
        p_e = self.lhts_p
        self.mixs["total"].equilibrate(p_e,T_e)
        eq = [0.0] * 1
        eq[0] = self.lhts_h - self.mixs["total"].mixtureHMass()

        # print("-> p_e = %2.16f"%p_e)
        # print("-> T_e = %2.16f"%T_e)
        # print("-> H_e = %2.16f"%self.lhts_h)
        # print("-> H_e_mpp = %2.16f"%self.mixs["total"].mixtureHMass())
        # print("-> eq[0] = %2.16f"%eq[0])
        # print("")


        return eq
    
    def solve_temp_stagnation_system(self,toler):
        opt = self.options["options"]["total"]
        options={'maxiter': self.options["maxiter"]}
        if opt["robust"] == "Yes":
            result = scipy.optimize.minimize(self.temp_stagnation_system,self.ic_temp_stag,method='Nelder-Mead',tol=toler,options=options)

            if result.success == False:
                print("Warning: convergence not guaranteed for robust solve_temp_stagnation_system")

        else:
            result = scipy.optimize.root(self.temp_stagnation_system,self.ic_temp_stag,tol=toler,options=options)

        if result.success == False:
            print("Warning: convergence not guaranteed for normal solve_temp_stagnation_system")

        T_e = result.x[0]
        return T_e
    
    def get_effective_radius(self,rho_e,p_inf):
        r_eff = (1./self.lhts_beta) * math.sqrt((2.0*(self.lhts_p-p_inf))/rho_e)
        return r_eff
    
    def solve_all(self):
        print("*** LHTS MODE RUNNING ***")
        print("Inputs:")
        print("-> Beta = %2.16f"%self.lhts_beta)
        print("-> P_stag = %2.16f"%self.lhts_p)
        print("-> H_stag = %2.16f"%self.lhts_h)
        print("-> Mach infty = %2.6f"%self.fixed_Mach)
        print("Initial guess:")
        print("-> p_inf = %2.16f"%self.ic_freestream[0])
        print("-> T_inf = %2.16f"%self.ic_freestream[1])
        print("-> p_2 = %2.16f"%self.ic_post[0])
        print("-> T_2 = %2.16f"%self.ic_post[1])
        print("-> u_2 = %2.16f"%self.ic_post[2])
        print("-> T_stag = %2.16f"%self.ic_temp_stag[0])
        print("")

        print("*** Finding stagnation temperature ***")
        if self.T_stag != 0.0:
            T_e = self.T_stag
        else:
            tol_temp = 1.0E-8
            T_e = self.solve_temp_stagnation_system(tol_temp)
        p_e = self.lhts_p
        self.mixs["total"].equilibrate(p_e,T_e)
        rho_e = self.mixs["total"].density()

        print("*** Solving for pre- and post-shock conditions ***")
        tol_prepost = 1.0E-8
        T_inf,p_inf,u_inf = self.solve_prepost_shock_system(tol_prepost)

        print("*** Solving for reservoir conditions ***")
        tol_res = 1.0E-8
        T_0,p_0 = self.solve_reservoir_system(T_inf,p_inf,u_inf,tol_res)
        

        r_eff = 0.0#self.get_effective_radius(rho_e,p_inf)

        print("")
        print("--- Computed freestream conditions ---")
        print("-> p_inf = %2.16f"%p_inf)
        print("-> T_inf = %2.16f"%T_inf)
        print("-> u_inf = %2.16f"%u_inf)
        print("--- Computed reservoir conditions ---")
        print("-> p_0 = %2.16f"%p_0)
        print("-> T_0 = %2.16f"%T_0)
        print("--- Computed effective radius ---")
        print("-> R_eff = %2.16f"%r_eff)

        preshock_state = [T_inf,p_inf,self.fixed_Mach]
        A_t = 0.0 # only used for mass flow

        measurements = forward(preshock_state,self.resmin,A_t,r_eff,self.T_w,self.pr,self.L,self.mixs,self.print_info,self.options["options"])
        print("")
        print("--- Check with the Cabaret forward mode ---")
        print("-> P_stag = %2.16f"%measurements["Stagnation_pressure"])
        print("-> H_stag = %2.16f"%measurements["Stagnation_pressure"])
        print("-> p_0 = %2.16f"%measurements["Reservoir_pressure"])
        print("-> T_0 = %2.16f"%measurements["Reservoir_temperature"])







