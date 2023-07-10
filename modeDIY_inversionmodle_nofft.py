# vim: fileencoding=utf-8

#
#import from officail package
#
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import copy
import glob
import os
from scipy import signal
from scipy import fftpack
from scipy import interpolate
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.special import comb
from matplotlib.colors import Normalize
from matplotlib import ticker, cm
import warnings
warnings.filterwarnings('ignore')

cc = 299792458
ff = 46.5*1e6

def vortex_crrelation(scale, sigma, lag, mu=0):
    return scale*np.exp( -(lag-mu)**2/(2*sigma**2) )

def window_correlation(lag, width_pass):
    xcf_window = np.empty_like(lag, dtype=float)
    for i in range(lag.shape[0]):
        if(width_pass < np.abs(lag[i])):
            xcf_window[i] = 0
        elif(0<lag[i]):
            xcf_window[i] = -lag[i]/width_pass + 1
        else:
            xcf_window[i] = lag[i]/width_pass + 1
    return xcf_window


class signal_3Dvolume_simu():
    
    # def __init__(self, signal_dirpath, radar_range, data_leng=2**10, obervation_mode=None, ant_mu=["F2", "F3", "F4","A2","A4","F5"]):
    def __init__(self, signal_dirpath, ant_mu, radar_range, data_leng, snr_db, obervation_mode=None):
       #
        #load files
        #
        # print(signal_dirpath,'inputsigpath_Functionpro')
        files = glob.glob(signal_dirpath)
       # for file in files:
           # print(file)
        n_subarray = len(ant_mu)
        #
        #decode signal
        #
        
        # data = np.loadtxt(files[0], delimiter=" ", max_rows=data_leng)
        # data = np.loadtxt(files[0], delimiter = ',', max_rows=data_leng)
        # data = np.loadtxt(files[0])
        # n_subarray = int((data.shape[1]-1)/2)
        # if data.shape[1] == n_subarray*2+1:
        # if data.shape[1] == 15:
        #     time = data[:,0]
        #     dt = time[1] - time[0]
        #     self.sig = np.zeros((n_subarray,data_leng), dtype=complex)
        #     #
        #     #decode signal
        #     #
        #     for file in files:
        #         # data = np.loadtxt(file,  delimiter = ',', max_rows=data_leng)
        #         data = np.loadtxt(file, max_rows=data_leng)
        #         for i in range(n_subarray):
        #             self.sig[i,:] = self.sig[i,:] + data[:,i+1] + 1j*data[:,i+1+n_subarray]
        # else:
            #####for mode7 start
            # self.sig = np.zeros((7,data_leng), dtype=complex)
            # #
            # #decode signal
            # #
            # for file in files:
            #     # data = np.loadtxt(file,  delimiter = ',', max_rows=data_leng)
            #     # print(file,'filename_funcpro')
            #     data = np.loadtxt(file)
            #     new_data = np.zeros((int(len(data)/5), 15))
            #     for i in range(int(len(data)/5)):
            #         new_data[i] = np.concatenate((data[5*i], data[5*i+1], data[5*i+2], data[5*i+3], data[5*i+4]))
            #     time = new_data[:,0]
            #     dt = time[1] - time[0]
            #     # for k in range(n_subarray):
            #     for k in range(7):
            #         self.sig[k,:] = self.sig[k,:] + new_data[:,k+1] + 1j*new_data[:,k+1+n_subarray]
            ######mode7 end
            
            ######mode3 start
        self.sig = np.zeros((3,data_leng),dtype = complex)
            #
            #decode signal
            #
        for datfile in files:
            with open(datfile, 'r') as file:
                lines = file.readlines()

            data = []
            for line in lines:
                line = line.strip()
                if line:
                    elements = line.split()
                    data.extend(elements)

            data = np.array(data, dtype=float)

           
            num_columns = 7
            output_rows = len(data) // num_columns
            new_data = data[:output_rows * num_columns].reshape((output_rows, num_columns))
            time = new_data[:,0]
            dt = time[1] - time[0]
            for k in range(3):
                self.sig[k,:] = self.sig[k,:] + new_data[:,k+1] + 1j*new_data[:,k+1+n_subarray]
                
                
            
        #add noise(gaussian white noise)
        #power: signal power
        #npower: noise power
        #X: signal serial
        for i in range(self.sig.shape[0]):
            noise_r = np.random.randn(self.sig.shape[0],self.sig.shape[1])
            noise_i = np.random.randn(self.sig.shape[0],self.sig.shape[1])
            pre_power = abs(self.sig)
            snr = 10**(snr_db/10)
            power = np.mean(np.square(pre_power))
            npower = power / snr
            noise_r = noise_r * np.sqrt(npower)
            noise_i = 1j * noise_i * np.sqrt(npower)
            sig_noise = self.sig + noise_r + noise_i
            self.sig_noise = sig_noise


        self.time = time
        self.dt = dt
        self.fs = 1.0/self.dt
        self.cc = cc
        self.ff = ff
        self.lamda = self.cc/self.ff
        self.radar_range = radar_range
        self.data_leng = data_leng
        self.n_subarray = n_subarray
        self.snr_db = snr_db

        #get subarray number and name
        self.arrayname = np.array(ant_mu)
        self.arraynum = len(self.arrayname)

        if self.n_subarray == self.arraynum:
            print("Format matched!")
        else:
            print("Format doesn't match. Please check the input arrays or the dataset.")


    
    def get_simulatoin_signal_correlation_function(self, subarray1, subarray2, flag_dvl0=False):

        # if (self.data_leng % incoherent == 0):
        #     sample_len = int(self.data_leng / incoherent)
        # else:
        #     message = "Error!! mod( data_leng:{}, incoherent:{} ) is not 0!!".format(self.data_leng, incoherent)
        #     raise Exception(message)
        sample_len = self.data_leng
        frq = fftpack.fftfreq(n=2*sample_len-1, d=self.dt)
        fft_xcf = np.zeros_like(frq, dtype=complex)
        time_len = sample_len * self.dt
        if self.snr_db == 0:
            if ( type(subarray1) is not type(subarray2) ):
                message = 'type mismatch input subarrays in signal1:{} & in signal2:{}'.format(type(subarray1), type(subarray2))
                raise Exception(message)
            if ( type(subarray1) and type(subarray2) ) is tuple: 
                signal1 = np.sum(self.sig[subarray1, :], axis=0)
                signal2 = np.sum(self.sig[subarray2, :], axis=0)
                print('signal1: subarrays:{} are combined & signal2: subarrays:{} are combined'.format(subarray1, subarray2))
            elif ( type(subarray1) and type(subarray2) ) is (int or np.int64 or np.int32): 
                signal1 = self.sig[subarray1, :]
                signal2 = self.sig[subarray2, :]
                # print('signal1: subarrays:{} & signal2: subarrays:{}'.format(subarray1, subarray2))
            else:
                message = 'input subarrays in signal1:{} & in signal2:{}'.format(type(subarray1), type(subarray2))
                raise Exception(message)
        else:
            print('generate noisy signals')
            signal1 = self.sig_noise[subarray1, :]
            signal2 = self.sig_noise[subarray2, :]

        # for i in range(incoherent):
        #     s = sample_len*i
        #     e = sample_len*(i+1)
        #     #signal
        #     sig1_tmp = signal1[s:e]
        #     sig2_tmp = signal2[s:e]
            
        #     xcf_tmp = signal.correlate(in1=sig1_tmp, in2=sig2_tmp)
         
        #     xcf_tmp = fftpack.fftshift(xcf_tmp)
        #     fft_xcf = fft_xcf + fftpack.fft(xcf_tmp)
        
        xcf = signal.correlate(in1=signal1, in2=signal2)
        # xcf = fftpack.fftshift(xcf)
        fft_xcf = fftpack.fft(xcf)
        # fft_xcf = fftpack.fftshift(fft_xcf)
        
        
        
        if (flag_dvl0==False):
            # frq = np.delete(frq, 0)
            # fft_xcf = np.delete(fft_xcf, 0)
            fft_xcf[0] = 0
        
        xcf_d0 = fftpack.ifft(fft_xcf)
        xcf_d0_select = xcf_d0[3800:4199]
        lag_select = np.arange(-199,200)
        lag_time_select = lag_select*self.dt
        lag_time_select = np.array(lag_time_select)
        # print(lag_time_select.shape)
        # print(lag_time_select[399])
        
        # xcf_select = xcf[3970:4220]
        # xcf_select = fftpack.fftshift(xcf_select)
        # lag_select = signal.correlation_lags(126,126)
        # fft_xcf_select = fftpack.fft(xcf_select)
        lag_idx_all = signal.correlation_lags(in1_len=signal1.size, in2_len=signal1.size)
        lag_time_all = lag_idx_all * self.dt
        # lag_idx_incoherent = signal.correlation_lags(in1_len=sig1_tmp.size, in2_len=sig1_tmp.size)
        # lag_time_incoherent = lag_idx_incoherent * self.dt
       
        if ( type(subarray1) and type(subarray2) ) is tuple:
            label_mu = 'xcf({}),xcf({})'.format(self.arrayname[list(subarray1)], self.arrayname[list(subarray2)])
        elif ( type(subarray1) and type(subarray2) ) is (int or np.int64 or np.int32):
            label_mu = 'xcf({}),xcf({})'.format(self.arrayname[subarray1], self.arrayname[subarray2])
        out = {'fft_xcf':fft_xcf, 'xcf_d0_part':xcf_d0_select,'doppler_freq':frq, 'xcf':xcf, 'lag_time_all':lag_time_all,'name_list':label_mu,
               'lag_time_select1':lag_time_select}
            #    'mu_label':label_mu, 'range':self.radar_range, \
            #    'obsmode':self.obsmode, 'sampling_time_len':time_len, 'incoherent':incoherent}
        return out


    def get_unit_simulation_signal_correlation_function(self, flag_dvl0=False):
        # if (self.data_leng % incoherent_u == 0):
        #     sample_len = int(self.data_leng / incoherent_u)
        # else:
        #     message = "Error!! mod( data_leng:{}, incoherent:{} ) is not 0!!".format(self.data_leng, incoherent_u)
        #     raise Exception(message)
        sample_len = self.data_leng
        time_len = sample_len * self.dt                     
        fft_xcf_unit = []       
        xcf_unit = []
        namelist_u = []  
        # fft_xcf_select_com = []

        # for s1 in range(self.n_subarray-1):
        #     for s2 in range(self.n_subarray-1,0,-1):
        #         if s2 <= s1:
        #             break
        #         else:
        #             out = self.get_simulatoin_signal_correlation_function(\
        #                                                 subarray1=s1, \
        #                                                 subarray2=s2, \
        #                                                 incoherent=incoherent_u, flag_dvl0=flag_dvl0)
        #             fft_xcf_unit.append(out.get("fft_xcf"))        
        #             xcf_unit.append(out.get("xcf"))
        #             namelist_u.append(out.get('name_list'))
                    
                    # label_mu.append(out.get('label_mu'))
                    # print(s1,s2)
                    # print(namelist_u)
        #test 3----4 subarrays:F2,F3,F4(0,1,2)
        testcomb = [[0,1],[0,2],[1,2],[0,3],[2,3]]
        for i,j in testcomb:
            out = self.get_simulatoin_signal_correlation_function(subarray1=i,subarray2=j,flag_dvl0=flag_dvl0)
            fft_xcf_unit.append(out.get("fft_xcf"))
            xcf_unit.append(out.get("xcf_d0_part"))
            namelist_u.append(out.get('name_list'))
            # fft_xcf_select_com.append(out.get('fft_xcf_select'))
            
        
        

        fft_xcf_unit = np.atleast_2d(np.array(fft_xcf_unit))    
        xcf_unit = np.atleast_2d(np.array(xcf_unit))
        # print('wai_xcfunit',xcf_unit.shape)
        # fft_xcf_select_com = np.atleast_2d(np.array(fft_xcf_select_com))
        frq = out.get("doppler_freq")           
        lag_time_all = out.get("lag_time_all")   
        # print('lagtimeall',lag_time_all) 
        lag_time_select2 = out.get('lag_time_select1')   
        # print('lag_time_sel',lag_time_select2)      
        # lag_time_incoherent = out.get("lag_time_incoherent")
        
        namelist_u = tuple(namelist_u)
        outunit = {'fft_xcf_unit':fft_xcf_unit, 'doppler_freq':frq, \
               'lag_time_all':lag_time_all, \
               'xcf_unit':xcf_unit, 'Com_nmlist':namelist_u, 'range':self.radar_range, \
               'sampling_time_len':time_len,
               'lag_time_select2':lag_time_select2}
        return outunit

class beam_filed():
    def __init__(self, hfile, path_append, ant_mu):
        self.cc = cc
        self.ff = ff
        self.lamda = self.cc/self.ff
        try:
            f = open(hfile)
        except IndexError:
            print ('cannot open input file')
        else:
            header_data = f.read().split()
            f.close()
        self.DFILE = path_append + header_data[1]
        self.NG = int(header_data[3])
        self.NX = int(header_data[5])
        self.NY = int(header_data[7])
        self.XMIN = float(header_data[9])
        self.XMAX = float(header_data[11])
        self.YMIN = float(header_data[13])
        self.YMAX = float(header_data[15])
        self.RANGE = float(header_data[17])
        self.windvector_deg = float(header_data[19])
        self.psi_deg = float(header_data[21])
        self.mainrobe_zenith_deg = float(header_data[23])
        self.mainrobe_azimuth_deg= float(header_data[25])
        self.dx = (self.XMAX - self.XMIN)/self.NX
        #
        #decode binary data
        #
        head = ("head","<i")
        tail = ("tail","<i")
        num1 = self.NX*self.NY
        num2 = self.NX*self.NY*self.NG
        data_type = np.dtype([head, ("E",[("real", "<"+str(num2)+"d"),("imag", "<"+str(num2)+"d")]), tail])
        f = open(self.DFILE, mode='rb')
        data = np.fromfile(f, dtype=data_type)
        f.close()
        #
        #read reflextion data
        #
        reflex = data["E"]["real"] + 1j*data["E"]["imag"]
        self.reflex = reflex.reshape(self.NY,self.NX,self.NG)
        # self.ant_mu = np.array(ant_mu)
        # self.obsmode = "TX,RX: 3D scattering simulation"
        #get subarray number and name
        self.arrayname = np.array(ant_mu)
        self.arraynum = len(self.arrayname)
        #
        #check
        #
        # print("\n==========================")
        # print("DFILE:{}".format(self.DFILE))
        # print("NG:{}".format(self.NG))
        # print("NX:{}".format(self.NX))
        # print("NY:{}".format(self.NY))
        # print("xmin:{}[m] xmax:{}[m]".format(self.XMIN, self.XMAX))
        # print("ymin:{}[m] ymax:{}[m]".format(self.YMIN, self.YMAX))
        # print("range:{}[m]".format(self.RANGE))
        # print("space lag:{:.2f}[m]".format(self.dx))
        # print("reflex reshape:{}".format(self.reflex.shape))
        # print("==========================")


    
    def get_beamfield_correlation(self, subarray1, subarray2):
        if ( type(subarray1) is not type(subarray2) ):
            message = 'type mismatch input subarrays in signal1:{} & in signal2:{}'.format(type(subarray1), type(subarray2))
            raise Exception(message)
        if ( type(subarray1) and type(subarray2) ) is tuple:
            signal1 = np.sum(self.reflex[:,:,subarray1], axis=2)
            signal2 = np.sum(self.reflex[:,:,subarray2], axis=2)
            print('signal1: subarrays:{} are combined & signal2: subarrays:{} are combined'.format(subarray1, subarray2))
            # print(signal1.shape)
        elif ( type(subarray1) and type(subarray2) ) is (int or np.int64 or np.int32):
            signal1 = self.reflex[:,:,subarray1]
            signal2 = self.reflex[:,:,subarray2]
            # print('signal1: subarrays:{} & signal2: subarrays:{}'.format(subarray1, subarray2))
        else:
            message = 'input subarrays in signal1:{} & in signal2:{}'.format(type(subarray1), type(subarray2))
            raise Exception(message)
        if ( type(subarray1) and type(subarray2) ) is tuple:
            label_mu = 'xcf({}),xcf({})'.format(self.arrayname[list(subarray1)], self.arrayname[list(subarray2)])
        elif ( type(subarray1) and type(subarray2) ) is (int or np.int64 or np.int32):
            label_mu = 'xcf({}),xcf({})'.format(self.arrayname[subarray1], self.arrayname[subarray2])
    
        #
        #calculate correlation function
        #
        lag_space = signal.correlation_lags(in1_len=self.NX, in2_len=self.NX) * self.dx
        xcf = np.zeros(lag_space.size, dtype=complex)
        for idx_y in range(self.NY):
            xcf = xcf + signal.correlate(in1=signal1[idx_y,:], in2=signal2[idx_y,:])
        
        header = {'datafile':self.DFILE,'subarray_number':self.NG,'xgrid_number':self.NX,\
            'ygrid_number':self.NY,'xlim': [self.XMIN, self.XMAX], 'ylim':[self.YMIN, self.YMAX],\
                'radar_range': self.RANGE, 'dx_space_lag': self.dx,'label_mu': label_mu,'xcf_beam':xcf,'lag_space':lag_space}
        return header


    def get_unit_beamfield_correlation(self):
        unit_len = comb(self.arraynum,2)                       
        # print('unit_len{}'.format(unit_len))
        xcf_unit = []
        label_mu = []   
        # for s1 in range(self.arraynum-1):
        #     for s2 in range(self.arraynum-1,0,-1):
        #         if s2 <= s1:
        #             break
        #         else:
        #             out = self.get_beamfield_correlation(subarray1=s1,subarray2=s2)
        #             xcf_unit.append(out.get("xcf_beam"))        
        #             label_mu.append(out.get('label_mu'))
                    # print(label_mu)
        
        testcom = [[0,1],[0,2],[1,2],[0,3],[2,3]]
        for i,j in testcom:
            out = self.get_beamfield_correlation(subarray1=i,subarray2=j)
            xcf_unit.append(out.get("xcf_beam"))        
            label_mu.append(out.get('label_mu'))

      

        xcf_unit = np.atleast_2d(np.array(xcf_unit))
        # header = out.get('header')
        lag_space = out.get('lag_space')
        label_mu = tuple(label_mu)
        outunit = { \
            'xcf_beam_unit':xcf_unit, 'lag_space':lag_space, \
            'mu_label':label_mu}
        return outunit


    #
    #plot two way beam pattern
    #
    def plot_two_way_beam_pattern(self, xlim=None, ylim=None):
        if (xlim is None):
            xlim = [self.XMIN, self.XMAX]
        if (ylim is None):
            ylim = [self.YMIN, self.YMAX]
        xx = np.linspace(self.XMIN, self.XMAX, self.NX)
        yy = np.linspace(self.YMIN, self.YMAX, self.NY)
        X,Y = np.meshgrid(xx,yy)
        power = np.abs(np.sum(self.reflex, axis=2))**2
        logpower = 10*np.log10(power/np.max(power))
        #
        #2way beam pattern
        #
        fig, ax = plt.subplots(1,figsize=(6,6))
        ax.set_ylabel('Y [m]')
        ax.set_xlabel('X [m]')
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        f = ax.pcolormesh(X, Y, logpower, cmap=cm.viridis, norm=Normalize(vmin=-40, vmax=0))
        col = fig.colorbar(f, ax=ax, fraction=0.1)
        col.set_label("normalized antenna pattern[dB]", fontsize=12)
        ax.set_title('RANGE:{:.0f}[m],AZIMUTH:{:.0f}[deg],ZENITH:{:.0f}[rad],\nWINDVECTOR_DIREC:{:.0f}[rad]'.format(self.RANGE,self.mainrobe_azimuth_deg,self.mainrobe_zenith_deg,self.windvector_deg))
        ax.grid()
        plt.show()




#
# radar inversion (no fft, directly use time series)
#
class radar_inversion():
    def __init__(self, zenith):
        self.cc = cc                                    
        self.ff = ff                                    
        self.lamda = self.cc/self.ff                    
        self.zenith = zenith

    def inversion_vr_vv_vh(self, R, xcf_field, lag_time, xi_field, sampling_time_len, initial_param=None, optimize="leastsq"):
        xcf_window = window_correlation(lag=lag_time, width_pass=sampling_time_len)       
        unit_len = R.shape[0]                            
        # print('fftR',fftR.shape)
        # print('unit_len',unit_len)
        label_fc = []                                       
        if (initial_param is None):
            high = np.max(np.abs(R))
            initial_param = np.array( [ 1, 10, 0, high ] )                   
        else:
            initial_param = initial_param                                                   
        method = estimate_get_vr_vv_vh(R=R, GG_xi=xcf_field, WW=xcf_window, \
                                       lag_time=lag_time, xi=xi_field)
        #
        # by leastsq
        #
        if optimize=="leastsq":
            # print('optimization method:',optimize)
            least_sq = leastsq(func=method.get_residual, x0=initial_param)
            FGW = method.get_FGW(param=least_sq[0])
            error = method.get_residual(param=least_sq[0]).reshape(FGW.shape)
            sq_error = np.sum(error**2, axis=1)                                         
            sigma_time = abs(least_sq[0][0])                                            
            sigma_f = 1.0 / (2*np.pi*sigma_time)                                        
            sigma_v = 1.0 / (2*np.pi*sigma_time) * self.lamda/2                         # [m/s]
            # omega = least_sq[0][1]                                                      # [rad/s]
            # vvel = omega * self.lamda / (4*np.pi*np.cos(self.zenith/180*np.pi))         # [m/s]
            hvel = least_sq[0][1]                                                       # [m/s]
            phase0 = least_sq[0][2]                                                     #randar [rad]
            scale = least_sq[0][3]                                                      
            info_opt = least_sq
       
        elif optimize=="least_squares":
            # print('optimization method:',optimize)
            least_sq = least_squares(fun=method.get_residual, x0=initial_param)
            fftFGW = method.get_fftFGW(param=least_sq.x)
            error = method.get_residual(param=least_sq.x).reshape(fftFGW.shape)
            sq_error = np.sum(error**2, axis=1)                                        
            sigma_time = abs(least_sq.x[0])                                            # [s]
            sigma_f = 1.0 / (2*np.pi*sigma_time)                                       # [hz]
            sigma_v = 1.0 / (2*np.pi*sigma_time) * self.lamda/2                        # [m/s]
            # omega = least_sq.x[1]                                                      # [rad/s]
            # vvel = omega * self.lamda / (4*np.pi*np.cos(self.zenith/180*np.pi))        # [m/s]
            hvel = least_sq.x[1]                                                       # [m/s]
            phase0 = least_sq.x[2]                                                     #randar [rad]
            scale = least_sq.x[3]                                                      
            info_opt = least_sq
        else:
            raise TypeError ("optimize option is inappropriate!!")
        

        # print(sq_error.shape)
        param = { "horizontal_velocity": hvel,\
                  "intitial_phase": phase0, \
                  "variance_freq": sigma_f, "variance_vel": sigma_v, "variance_time": sigma_time, \
                  "function_scale": scale, "optimization_information": info_opt
                }
        out ={"param":param, "sq_error":sq_error, "FGW":FGW}
#         out ={"sq_error":sq_error}
        return out

class estimate_get_vr_vv_vh():

    def __init__(self, R, GG_xi, WW, lag_time, xi):
        self.R = np.atleast_2d(R)
        # print('R_shape',self.R.shape)
        self.GG_xi = np.atleast_2d(GG_xi)
        self.WW = WW
        # print('WW',self.WW.shape)
        self.lag_time = lag_time
        self.xi = xi
        self.subarraysize = self.R.shape[0]

    def get_residual(self, param):
        FGW = self.get_FGW(param)
        # print('FGW_shape',FGW.shape)
        res = np.abs( np.ravel( self.R - FGW ) )
        # print('res_shape',res.shape)
        # res[[1,3,6,8,16,18]] = res[[1,3,6,8,16,18]]*1.1
        # res[[1,3,6,8,16,18]] = 0
        # res[[2,7,15]] = res[[2,7,15]]*0.136
        # res[[2,7,15]] = 0
        #print(res.shape)
        return res

    def get_FGW(self, param):
        FGW = np.zeros_like(self.R, dtype=complex)
        tau_model = self.xi / param[1]
        for num in range(self.subarraysize):
            FF = vortex_crrelation(param[3], param[0], self.lag_time)
            cubic = interpolate.interp1d(x=tau_model, y=self.GG_xi[num,:], kind='cubic', fill_value="extrapolate")
            GG = cubic(self.lag_time)
            # FGW = np.zeros_like(GG, dtype=complex)
            # phase = -1j*( param[1]*self.lag_time + param[3] )
            # FGW[num,:] = FF * GG * np.exp(phase) * self.WW 
            FGW[num,:] = FF * GG  * self.WW    
            # FGW = fftpack.fftshift(FGW)                        
            fft = fftpack.fft(FGW[num,:])
            fft[0] = 0
            FGW[num,:] = fftpack.ifft(fft)
            #fft = np.delete(fft, 0)                             
            # fftFGW[num,:] = fft
        # print('getFGW',FGW.shape)  
        return FGW
