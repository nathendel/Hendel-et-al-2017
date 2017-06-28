from __future__ import division, print_function
import numpy as np
# import matplotlib.pyplot as plt
import scipy.stats as st
import time


class Cell:
    cells = []

    def __init__(self, t=30000, L=0, N=200, trans_speed=2, k_on=0, k_off=0,
             avalanche_on=True, thresh=30, num_release=5,
             L_mod=True, build_size=.00125, decay_size=.01, D=1.75, ava_power=2.85,
             retro=False, L_hog=0,ss=True, t_step=.01):  # L is length, N is number of particles

        self.t = t
        self.L = L
        self.k_on = k_on
        self.k_off = k_off
        self.N = N
        self.motors = [Motor(self) for i in range(N)]
        self.in_flagellum50 = self.N
        self.avalanche_on = avalanche_on
        self.thresh = thresh
        self.num_release = num_release
        self.recruited = 0
        self.L_mod = L_mod
        self.build_size = build_size
        self.ava_power = ava_power
        self.ava = []
        self.avaT = np.zeros(t)
        self.D = D #from Alex Chien and Ahmet Yildiz
        self.L_hog = L_hog #hand of god -- change length manually
        self.L_trace = np.zeros(t)
        self.flux = np.zeros(t)
        self.base = np.zeros(t)
        self.N_diffuse = np.zeros(t)
        self.retro=retro
        self.current_time=0
        self.ss=ss
        self.t_step = t_step #s
        self.trans_speed = trans_speed #2um, from Alex Chien and Ahmet Yildiz
        self.rms_disp = (2*D*1.75*self.t_step)**.5 #um
        self.decay_sizeMS = .01*decay_size #m/s
        self.decay = decay_size #m/s
        self.decay_size = t_step * decay_size #meters in one time step
        self.build_size = build_size
        self.L_predict = (2*self.D*(self.N-self.thresh)*self.build_size/self.decay_sizeMS)**.5

        if t:
            self.sim(self.t)

            if self.ss: #if True, simulate until length reaches steady state
                while not self.is_steadystate(): #check if steady state is achieved
                    self.extend(int(500/self.t_step))

            #define the steady state length as the average length in the last 3000 time steps
            self.L=np.mean(self.L_trace[-3000:])

            #determine when length reached steady state
            self.time2ss = np.argmax(self.L_trace>self.L)*self.t_step

            #calculate distribution of diffusing motors
            density=[]
            for i in range(self.current_time-10000,self.current_time):
                density+=self.distr(i)
            self.kd = st.gaussian_kde(density)
            self.diff_distr=self.kd.evaluate(np.linspace(0,self.L,100))

    def count_in_flagellum(self):
        '''
        method that returns the number of motors that are in the flagellum,
        either in in_flagellum transport or diffusion (not in the base)
        '''
        return sum([p.is_in_flagellum for p in self.motors])

    def extend(self,extend_time):
        '''
        method that extends the length of all arrays that have an entry
        at each time step, then simulates by extend_time more time steps
        '''
        self.L_trace = np.concatenate((self.L_trace,np.zeros(extend_time-1)))
        self.flux = np.concatenate((self.flux,np.zeros(extend_time-1)))
        self.base = np.concatenate((self.base,np.zeros(extend_time-1)))
        self.N_diffuse = np.concatenate((self.N_diffuse,np.zeros(extend_time-1)))
        self.avaT = np.concatenate((self.avaT,np.zeros(extend_time-1)))
        for p in self.motors:
            p.track = np.concatenate((p.track,np.zeros(extend_time-1)))
            p.in_flagellum_track = np.concatenate((p.in_flagellum_track,np.zeros(extend_time-1)))
            p.boundtrack = np.concatenate((p.boundtrack,np.zeros(extend_time-1)))

        self.sim(self.current_time+extend_time,start=self.current_time)

    def is_steadystate(self,fit_range=1000, eps=5e-6):
        '''
        method that looks at the last fit_range time steps, fits the length
        at those time steps to a linear regression, and if the slope of the
        linear regression is less than eps, then return True, indicating
        that the length is in steady state.
        '''
        fit_range = int(fit_range/self.t_step)
        if len(self.L_trace) < fit_range:
            return False
        slope,intercept = np.polyfit(range(fit_range),self.L_trace[-1*fit_range:],1)
        return abs(slope)<eps

    def distr(self,time=None):
        '''
        method that returns a list of positions of diffusing motors
        '''
        if time == None:
            time = self.current_time
        return [p.track[time] for p in self.motors if (p.in_flagellum_track[time] and not p.boundtrack[time])]

    def sim(self,t,start=0):
        '''
        Simulates intraflagellar transport. First checks if the user wants a
        special case (manual length adjustment (self.L_hog), no avalanche,
        no length changing.) If no special conditions are true,
        '''

        for i in range(start,t):
            self.current_time=i
            if self.L_hog and i==np.floor(t/2):
                self.L *= self.L_hog

            if self.avalanche_on:
                self.avalanche()

            if self.L_mod:
                if self.L >= self.decay_size:
                    self.L -= self.decay_size

                # new may 27
                elif self.L < self.decay_size:
                    self.L = 0

                self.L_trace[i] = self.L

            for p in self.motors:

                if p.is_in_flagellum:
                    if p.isbound:
                        p.active_trans()
                    else:
                        p.diffuse()

                p.isbound = p.binding()
                p.track[i] = p.pos
                p.in_flagellum_track[i] = p.is_in_flagellum
                p.boundtrack[i] = p.isbound

            self.flux[i] = sum([1 for j in self.motors if (j.pos < 1 and j.isbound and j.is_in_flagellum)])
            self.base[i]= sum([1 for j in self.motors if not j.is_in_flagellum])
            self.N_diffuse[i] = sum([j.is_in_flagellum and not j.isbound for j in self.motors])
            # self.track_in_flagellum[i] = self.count_in_flagellum()

    def avalanche(self):
        '''
        method that counts the number of motors in the base, compares it to a
        threshold for avalanching, determine a random weibull-distributed
        number of motors to inject. Injection happens by changing the in_flagellum
        status of that number of motors from False to True.
        '''
        # distr = floor(1/np.random.power(self.ava_power))

        # num_in_base = self.N - self.count_in_flagellum()
        in_base = [p for p in self.motors if not p.is_in_flagellum]
        num_in_base = len(in_base)

        if num_in_base > self.thresh:
            distr = int((num_in_base-self.thresh+10) * np.random.weibull(2.85) + 1)
            release = min(distr, num_in_base)
            self.avaT[self.current_time]=release

            for i in range(release):  # commented out to try power law
                #             for i in range(1/np.random.power(3))
                in_base[i].is_in_flagellum = True
                in_base[i].isbound = True

                if self.L_mod:
                    in_base[i].built = False
        else:
            self.avaT[self.current_time]=0

    def __repr__(self):
        string = 'Cell of length %s populated by %d motors' % (self.L, self.N)
        return string


class Motor:
    instances = []

    def __init__(self, cell, is_in_flagellum=False, isbound=True):
        self.pos = 0
        self.is_in_flagellum = is_in_flagellum
        self.isbound = isbound
        Motor.instances.append(self)
        self.cell = cell
        self.track = np.zeros(self.cell.t)
        self.in_flagellum_track = np.zeros(self.cell.t)
        self.boundtrack = np.zeros(self.cell.t)
        self.built = False

    def diffuse(self):
        '''
        Method that simulates diffusion.
        '''

        if self.cell.L_mod: #if length is allowed to change. Default=True
            if self.pos > self.cell.L:  # Length decay can push motors out of the flagellum, this pushes them back in
                self.pos = self.cell.L

            if self.pos == self.cell.L:
                if not self.isbound:
                    self.pos -= self.cell.rms_disp #it can only go back in the direction of the base, otherwise it bounces off the tip
            else:
                if self.cell.retro: #attempt to program retrograde transport, default is False
                    self.pos -= self.cell.trans_speed
                else: #default
                    r=np.random.rand() #decide if motor diffuses left or right
                    if r<.5:
                        self.pos -= self.cell.rms_disp
                    else:
                        self.pos += self.cell.rms_disp
                if self.pos < 0: #in case self.pos-rms_disp puts it below zero
                    self.pos = 0
                elif self.pos > self.cell.L:
                    self.pos = self.cell.L

            if self.pos <= 0:
                self.is_in_flagellum = False  # keep this for later, using avalanche model


        #Special case in which the length cannot change.
        elif not self.cell.L_mod:
            if self.pos == self.cell.L:
                self.binding()
                if not self.isbound:
                    self.pos -= self.cell.rms_disp


            elif self.pos == 0:
                self.is_in_flagellum = False
            else:
                r=np.random.rand()
                if r<.5:
                    self.pos -= self.cell.rms_disp
                else:
                    self.pos += self.cell.rms_disp


    def active_trans(self):
        '''
        Method that simulates active transport. Motors increase their
        position by cell.trans_speed. In the case of motor binding,
        this also checks if they already deposited their cargo when they
        reach the tip.
        '''
        if self.pos < self.cell.L:
            self.pos += self.cell.trans_speed
            self.pos = min(self.pos, self.cell.L)
        #         if self.pos == self.cell.L:
        if self.pos >= self.cell.L:
            if not self.built:
                self.cell.L += self.cell.build_size
                self.built = True
            self.isbound = False
        #         self.track.append(self.pos)

    def binding(self):
        '''
        Method that determines if motors should bind or unbind. Diffusing motors
        can bind to the flagellum if not self.isbound and start active transport,
        and active transport motors can unbind and start diffusing.
        '''
        roll = np.random.rand()
        if not self.isbound:
            if roll < self.cell.k_on:  # probability of binding to the IFT particle, stalling diffusion
                return True
            else:
                return False
            #                 print('bound!')
        else:  # if self.isbound == True
            if roll < self.cell.k_off:
                return False
            else:
                return True
            #                 print('unbound!')

            #         return self.isbound #return the updated bound state


    def __repr__(self):
        string = 'Motor at position %s' % self.pos
        return string


#
# #
# if __name__ == '__main__':
#     a=Cell()
#     print(a.L)
#     st=time.time()
#     b=Cell(t=5000,N=400)
#     while not b.is_steadystate():
#         # print('not ss')
#         b.extend(5000)
#     print(time.time()-st)
#
#     st2=time.time()
#     c=Cell(t=b.current_time, N=400)
#     print(time.time()-st2)

    # a.L_plot()
# 	# print(a.L)

## run profiler: python -m cProfile -s cumtime cell2.py


'''
notes on default params:
N=200 based on 10 transport complexes from Marshall and Rosenbaum 2001 Appendix multiplied by each injection event
sends 1-30 IFT particles from Ludington 2013 p.3926

decay_rate is from Marshall and Rosenbaum 2001 "intraflagellar transport balances continuous turnover.... appendix "Prediction of flagellar regeneration kinetics" section

'''
