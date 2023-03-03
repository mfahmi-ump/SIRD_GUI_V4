import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from plotly.subplots import make_subplots
from datetime import date
#=====================================================
#read data from excel file, combine according to date
dt_fulldata = pd.read_csv('sird_data.csv')
dt_fulldata['date'] = pd.to_datetime(dt_fulldata['date'])
dt_fulldata = dt_fulldata.set_index(['date'])
dt_fulldata = dt_fulldata.tail(dt_fulldata.shape[0] - 8)
Ptotal = int(3.153*10**7);
#=====================================================
#assign initial value
T = len(dt_fulldata);
t0 = 0;
N = T;
M = 500;

from datetime import timedelta as td, datetime as datetime
# initialdatadate = date(2021,1,1)
initialdatadate = dt_fulldata.index.tolist()[0];
sdate1 = initialdatadate.date();
edate1 = sdate1 + td(days = T + 1)
pastplot_t = pd.date_range(sdate1,edate1-td(days=1),freq='d')
sdate2 = edate1 + td(days = 1)
edate2 = sdate2 + td(days = T + 1)
futplot_t = pd.date_range(sdate2,edate2-td(days=1),freq='d')
fullplot_t = pd.date_range(sdate1,edate2-td(days=1),freq='d')

def datetoint(input_date):
    try:
        dateint = input_date - sdate1
        return dateint.days;
    except:
        return 0;
    
def NPIoptiontoint(guientry):
    if guientry == "Strict":
        return  2;
    elif guientry == "Mild":
        return  1;
    else:
        return  0;

datedict = {}
diff = edate2 - sdate1
for i in range(diff.days + 1):
    day = sdate1 + td(days=i)
    datedict[datetime.strptime(str(day),"%Y-%m-%d")] = i

Izero = dt_fulldata['cases_active'].iloc[0];
Rzero = dt_fulldata['cases_recovered'].iloc[0]; 
Dzero = dt_fulldata['deaths_total'].iloc[0]; 

Szero = (Ptotal - Izero - Rzero - Dzero);
dt = T/N;
t = np.linspace(0,N,N+1);
data_t = np.linspace(0,T,T+1);
futureplot = np.linspace(T,2*T,T+1)
dt_mcmcresult = pd.read_csv('mcmc_result_2.csv')
dt_mcmcresult = dt_mcmcresult.dropna();
mcmcsize = np.size(dt_mcmcresult, axis=0);
#=====================================================
#parameter calculations for future approximations
segment_NPI_durations = dt_mcmcresult['t'].diff().dropna();
def log_func(x, a, b):
    return a*np.log(x)+b
pars, cov = curve_fit(f=log_func, xdata=np.linspace(1,T,T), ydata=dt_mcmcresult['r'], p0=[0, 0], bounds=(-np.inf, np.inf))
pars2, cov2 = curve_fit(f=log_func, xdata=np.linspace(1,T,T), ydata=dt_mcmcresult['d'], p0=[0, 0], bounds=(-np.inf, np.inf))
mean_gamma = np.mean(dt_mcmcresult['gamma']);
#=====================================================
#read parameter from mcmc, assign according to step in SRK4
def parameter_func(j,guiinput):
    Beta = 0;
    r = 0;
    d = 0;
    gamma,sigma = 0, 0;
    
    [settings_entry1,userinput_entry11,userinput_entry12,userinput_entry13,userinput_entry21,userinput_entry22,userinput_entry23,userinput_entry31,userinput_entry32,userinput_entry33,userinput_entry41,userinput_entry42,userinput_entry43,userinput_entry51,userinput_entry52,userinput_entry53,userinput_entry61,userinput_entry62,userinput_entry63,userinput_entry71,userinput_entry72,userinput_entry73,userinput_entry81,userinput_entry82,userinput_entry83] = guiinput
    
    if j < T:
        Beta = dt_mcmcresult['beta'].iloc[j.astype(int)-1];
        r = dt_mcmcresult['r'].iloc[j.astype(int)-1];
        d = dt_mcmcresult['d'].iloc[j.astype(int)-1];
        gamma = dt_mcmcresult['gamma'].iloc[j.astype(int)-1];
        sigma = dt_mcmcresult['sigma'].iloc[j.astype(int)-1];
        if settings_entry1 == 1:
            sigma = sigma*5;
    else:
        npidatestart1 = datetoint(userinput_entry11);
        npidateend1   = datetoint(userinput_entry12);
        npidatestart2 = datetoint(userinput_entry21);
        npidateend2   = datetoint(userinput_entry22);
        npidatestart3 = datetoint(userinput_entry31);
        npidateend3   = datetoint(userinput_entry32);
        npidatestart4 = datetoint(userinput_entry41);
        npidateend4   = datetoint(userinput_entry42);
        npidatestart5 = datetoint(userinput_entry51);
        npidateend5   = datetoint(userinput_entry52);
        npidatestart6 = datetoint(userinput_entry61);
        npidateend6   = datetoint(userinput_entry62);
        npidatestart7 = datetoint(userinput_entry71);
        npidateend7   = datetoint(userinput_entry72);
        npidatestart8 = datetoint(userinput_entry81);
        npidateend8   = datetoint(userinput_entry82);
        
        futbetavalues = [0.09,0.06,0.045];

        if settings_entry1 == 1:
            sigma = 0;
        else:
            sigma = 0.005;
               
        if j > npidatestart1 and j <= npidateend1:
            Beta = futbetavalues[NPIoptiontoint(userinput_entry13)]
        if j > npidatestart2 and j <= npidateend2:
            Beta = futbetavalues[NPIoptiontoint(userinput_entry23)]
        if j > npidatestart3 and j <= npidateend3:
            Beta = futbetavalues[NPIoptiontoint(userinput_entry33)]
        if j > npidatestart4 and j <= npidateend4:
            Beta = futbetavalues[NPIoptiontoint(userinput_entry43)]
        if j > npidatestart5 and j <= npidateend5:
            Beta = futbetavalues[NPIoptiontoint(userinput_entry53)]
        if j > npidatestart6 and j <= npidateend6:
            Beta = futbetavalues[NPIoptiontoint(userinput_entry63)]
        if j > npidatestart7 and j <= npidateend7:
            Beta = futbetavalues[NPIoptiontoint(userinput_entry73)]
        if j > npidatestart8 and j <= npidateend8:
            Beta = futbetavalues[NPIoptiontoint(userinput_entry83)]
            
        r, d, gamma, sigma = log_func(j,*pars), log_func(j,*pars2), mean_gamma, sigma
    return Beta, r, d, gamma, sigma

#=====================================================
#SRK-4 Tableus form
A = np.array([[0  ,0  ,0,0],
              [1/2,0  ,0,0],
              [0  ,1/2,0,0],
              [0  ,0  ,1,0]]);
B_1 = np.array([[0            ,0            ,0          ,0],
                [-0.7242916356,0            ,0          ,0],
                [0.4237353406 ,-0.1994437050,0          ,0],
                [-1.578475506 ,0.840100343  ,1.738375163,0]]); 
B_2 = np.array([[0           ,0,0,0],
                [2.702000410 ,0,0,0],
                [1.757261649 ,0,0,0],
                [-2.918524118,0,0,0]]); 
G_1 = np.array([[-.7800788474],
                [0.07363768240],
                [1.486520013],
                [0.2199211524]]); 
G_2 = np.array([[1.693950844],
                [1.636107882],
                [-3.024009558],
                [-0.3060491602]]); 

def SIRD_func(t0,dt,N,M,Szero,Izero,Rzero,Dzero,Ptotal,guiinput):
    #=====================================================
    fS  = lambda S,I,R,D,Beta,r,d,gamma,sigma:0-S/Ptotal*Beta*I+gamma*R   
    gS  = lambda S,I,R,D,Beta,r,d,gamma,sigma:-sigma*I*S/Ptotal
    #=====================================================
    fI  = lambda S,I,R,D,Beta,r,d,gamma,sigma:S/Ptotal*Beta*I-r*I-d*I
    gI  = lambda S,I,R,D,Beta,r,d,gamma,sigma:sigma*I*S/Ptotal
    #=====================================================
    fR  = lambda S,I,R,D,Beta,r,d,gamma,sigma:r*I-gamma*R
    gR  = lambda S,I,R,D,Beta,r,d,gamma,sigma:0
    #=====================================================
    fD  = lambda S,I,R,D,Beta,r,d,gamma,sigma:d*I
    gD  = lambda S,I,R,D,Beta,r,d,gamma,sigma:0
    #=====================================================
        
    SdW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    SdW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    IdW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    IdW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    RdW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    RdW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    DdW1 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
    DdW2 = np.random.standard_normal((N+1,M+1))*np.sqrt(dt);
        
    Stemp = Szero*np.ones((1,M+1));
    Itemp = Izero*np.ones((1,M+1));
    Rtemp = Rzero*np.ones((1,M+1));
    Dtemp = Dzero*np.ones((1,M+1));
    
    Ssrk4_2 = np.ones((N+1,M+1));
    Isrk4_2 = np.ones((N+1,M+1));
    Rsrk4_2 = np.ones((N+1,M+1));
    Dsrk4_2 = np.ones((N+1,M+1));
    #=====================================================
        
    for j in np.linspace(t0,t0+N,N+1):
        Beta,r,d,gamma,sigma = parameter_func(j,guiinput);
        SWinc1 = SdW1[int(j) - t0];
        IWinc1 = IdW1[int(j) - t0];
        RWinc1 = RdW1[int(j) - t0];
        DWinc1 = DdW1[int(j) - t0];
        SWinc2 = SdW2[int(j) - t0];
        IWinc2 = IdW2[int(j) - t0];
        RWinc2 = RdW2[int(j) - t0];
        DWinc2 = DdW2[int(j) - t0];
        
        SJ10 = (0.5*(dt**(1/2))*(SWinc1+(1/np.sqrt(3))*SWinc2));
        IJ10 = (0.5*(dt**(1/2))*(IWinc1+(1/np.sqrt(3))*IWinc2));
        RJ10 = (0.5*(dt**(1/2))*(RWinc1+(1/np.sqrt(3))*RWinc2));
        DJ10 = (0.5*(dt**(1/2))*(DWinc1+(1/np.sqrt(3))*DWinc2));
        
        S1 = Stemp;
        I1 = Itemp;
        R1 = Rtemp;
        D1 = Dtemp;
        
        f1_S = fS(S1,I1,R1,D1,Beta,r,d,gamma,sigma);
        g1_S = gS(S1,I1,R1,D1,Beta,r,d,gamma,sigma);
        f1_I = fI(S1,I1,R1,D1,Beta,r,d,gamma,sigma);
        g1_I = gI(S1,I1,R1,D1,Beta,r,d,gamma,sigma);
        f1_R = fR(S1,I1,R1,D1,Beta,r,d,gamma,sigma);
        g1_R = gR(S1,I1,R1,D1,Beta,r,d,gamma,sigma);
        f1_D = fD(S1,I1,R1,D1,Beta,r,d,gamma,sigma);
        g1_D = gD(S1,I1,R1,D1,Beta,r,d,gamma,sigma);
        
        S2 = Stemp + dt*(A[0,1]*f1_S) + (B_1[0,1]*SWinc1 + B_2[0,1]*SJ10)*g1_S;
        I2 = Itemp + dt*(A[0,1]*f1_I) + (B_1[0,1]*IWinc1 + B_2[0,1]*IJ10)*g1_I;
        R2 = Rtemp + dt*(A[0,1]*f1_R) + (B_1[0,1]*RWinc1 + B_2[0,1]*RJ10)*g1_R;
        D2 = Dtemp + dt*(A[0,1]*f1_D) + (B_1[0,1]*DWinc1 + B_2[0,1]*DJ10)*g1_D;
        
        f2_S = fS(S2,I2,R2,D2,Beta,r,d,gamma,sigma);
        g2_S = gS(S2,I2,R2,D2,Beta,r,d,gamma,sigma);
        f2_I = fI(S2,I2,R2,D2,Beta,r,d,gamma,sigma);
        g2_I = gI(S2,I2,R2,D2,Beta,r,d,gamma,sigma);
        f2_R = fR(S2,I2,R2,D2,Beta,r,d,gamma,sigma);
        g2_R = gR(S2,I2,R2,D2,Beta,r,d,gamma,sigma);
        f2_D = fD(S2,I2,R2,D2,Beta,r,d,gamma,sigma);
        g2_D = gD(S2,I2,R2,D2,Beta,r,d,gamma,sigma);
        
        S3 = Stemp + dt*(A[1,2]*f2_S) + (B_1[0,2]*SWinc1 + B_2[0,2]*SJ10)*g2_S + (B_1[1,2]*SJ10*g2_S);
        I3 = Itemp + dt*(A[1,2]*f2_I) + (B_1[0,2]*IWinc1 + B_2[0,2]*IJ10)*g2_I + (B_1[1,2]*IJ10*g2_I);
        R3 = Rtemp + dt*(A[1,2]*f2_R) + (B_1[0,2]*RWinc1 + B_2[0,2]*RJ10)*g2_R + (B_1[1,2]*RJ10*g2_R);
        D3 = Dtemp + dt*(A[1,2]*f2_D) + (B_1[0,2]*DWinc1 + B_2[0,2]*DJ10)*g2_D + (B_1[1,2]*DJ10*g2_D);
            
        f3_S = fS(S3,I3,R3,D3,Beta,r,d,gamma,sigma);
        g3_S = gS(S3,I3,R3,D3,Beta,r,d,gamma,sigma);
        f3_I = fI(S3,I3,R3,D3,Beta,r,d,gamma,sigma);
        g3_I = gI(S3,I3,R3,D3,Beta,r,d,gamma,sigma);
        f3_R = fR(S3,I3,R3,D3,Beta,r,d,gamma,sigma);
        g3_R = gR(S3,I3,R3,D3,Beta,r,d,gamma,sigma);
        f3_D = fD(S3,I3,R3,D3,Beta,r,d,gamma,sigma);
        g3_D = gD(S3,I3,R3,D3,Beta,r,d,gamma,sigma);
        
        S4 = Stemp + dt*f3_S + (B_1[0,3]*SWinc1 + B_2[0,3]*SJ10)*g3_S + (B_1[1,3]*SWinc1*g3_S + B_1[2,3]*SWinc1*g3_S);
        I4 = Itemp + dt*f3_I + (B_1[0,3]*IWinc1 + B_2[0,3]*IJ10)*g3_I + (B_1[1,3]*IWinc1*g3_I + B_1[2,3]*IWinc1*g3_I);
        R4 = Rtemp + dt*f3_R + (B_1[0,3]*RWinc1 + B_2[0,3]*RJ10)*g3_R + (B_1[1,3]*RWinc1*g3_R + B_1[2,3]*RWinc1*g3_R);
        D4 = Dtemp + dt*f3_D + (B_1[0,3]*DWinc1 + B_2[0,3]*DJ10)*g3_D + (B_1[1,3]*DWinc1*g3_D + B_1[2,3]*DWinc1*g3_D);
        
        f4_S = fS(S4,I4,R4,D4,Beta,r,d,gamma,sigma);
        g4_S = gS(S4,I4,R4,D4,Beta,r,d,gamma,sigma);
        f4_I = fI(S4,I4,R4,D4,Beta,r,d,gamma,sigma);
        g4_I = gI(S4,I4,R4,D4,Beta,r,d,gamma,sigma);
        f4_R = fR(S4,I4,R4,D4,Beta,r,d,gamma,sigma);
        g4_R = gR(S4,I4,R4,D4,Beta,r,d,gamma,sigma);
        f4_D = fD(S4,I4,R4,D4,Beta,r,d,gamma,sigma);
        g4_D = gD(S4,I4,R4,D4,Beta,r,d,gamma,sigma);
        
        Stemp = Stemp + dt*((1/6)*f1_S + (1/3)*f2_S + (1/3)*f3_S + (1/6)*f4_S) + (G_1[0]*SWinc1 + G_2[0]*SJ10)*g1_S + (G_1[1]*SWinc1 + G_2[1]*SJ10)*g2_S + (G_1[2]*SWinc1 + G_2[2]*SJ10)*g3_S + (G_1[3]*SWinc1 + G_2[3]*SJ10)*g4_S;
        Itemp = Itemp + dt*((1/6)*f1_I + (1/3)*f2_I + (1/3)*f3_I + (1/6)*f4_I) + (G_1[0]*IWinc1 + G_2[0]*IJ10)*g1_I + (G_1[1]*IWinc1 + G_2[1]*IJ10)*g2_I + (G_1[2]*IWinc1 + G_2[2]*IJ10)*g3_I + (G_1[3]*IWinc1 + G_2[3]*IJ10)*g4_I;
        Rtemp = Rtemp + dt*((1/6)*f1_R + (1/3)*f2_R + (1/3)*f3_R + (1/6)*f4_R) + (G_1[0]*RWinc1 + G_2[0]*RJ10)*g1_R + (G_1[1]*RWinc1 + G_2[1]*RJ10)*g2_R + (G_1[2]*RWinc1 + G_2[2]*RJ10)*g3_R + (G_1[3]*RWinc1 + G_2[3]*RJ10)*g4_R;
        Dtemp = Dtemp + dt*((1/6)*f1_D + (1/3)*f2_D + (1/3)*f3_D + (1/6)*f4_D) + (G_1[0]*DWinc1 + G_2[0]*DJ10)*g1_D + (G_1[1]*DWinc1 + G_2[1]*DJ10)*g2_D + (G_1[2]*DWinc1 + G_2[2]*DJ10)*g3_D + (G_1[3]*DWinc1 + G_2[3]*DJ10)*g4_D;
        
        Ssrk4_2[int(j) - t0] = Stemp;
        Isrk4_2[int(j) - t0] = Itemp;
        Rsrk4_2[int(j) - t0] = Rtemp;
        Dsrk4_2[int(j) - t0] = Dtemp;
        
    return Ssrk4_2, Isrk4_2, Rsrk4_2, Dsrk4_2

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

with st.sidebar:
    st.write('Predictive Model User Input')
    scol1, scol2, scol3 = st.columns(3)
    with scol2:
        userinput_entry12 = st.date_input("End date",sdate2+ td(days = 30),key=12)
        userinput_entry22 = st.date_input("",sdate2,key=22)
        userinput_entry32 = st.date_input("",sdate2,key=32)
        userinput_entry42 = st.date_input("",sdate2,key=42)
        userinput_entry52 = st.date_input("",sdate2,key=52)
        userinput_entry62 = st.date_input("",sdate2,key=62)
        userinput_entry72 = st.date_input("",sdate2,key=72)
        userinput_entry82 = st.date_input("",sdate2,key=82)
    with scol1:
        userinput_entry11 = st.date_input("Start date",sdate2,key=11)
        userinput_entry21 = st.date_input("",sdate2,key=21)
        userinput_entry31 = st.date_input("",sdate2,key=31)
        userinput_entry41 = st.date_input("",sdate2,key=41)
        userinput_entry51 = st.date_input("",sdate2,key=51)
        userinput_entry61 = st.date_input("",sdate2,key=61)
        userinput_entry71 = st.date_input("",sdate2,key=71)
        userinput_entry81 = st.date_input("",sdate2,key=81)
    with scol3:
        userinput_entry13 = st.selectbox('NPI strictness',('Strict', 'Mild', 'Loose'),key=13)
        userinput_entry23 = st.selectbox("",('Strict', 'Mild', 'Loose'),key=23)
        userinput_entry33 = st.selectbox("",('Strict', 'Mild', 'Loose'),key=33)
        userinput_entry43 = st.selectbox("",('Strict', 'Mild', 'Loose'),key=43)
        userinput_entry53 = st.selectbox("",('Strict', 'Mild', 'Loose'),key=53)
        userinput_entry63 = st.selectbox("",('Strict', 'Mild', 'Loose'),key=63)
        userinput_entry73 = st.selectbox("",('Strict', 'Mild', 'Loose'),key=73)
        userinput_entry83 = st.selectbox("",('Strict', 'Mild', 'Loose'),key=83)

guiinput1 = [0,
        sdate1,edate1,0,
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,0]
 
Ssrk4_2, Isrk4_2, Rsrk4_2, Dsrk4_2 = SIRD_func(t0,dt,N,M,Szero,Izero,Rzero,Dzero,Ptotal,guiinput1)

#=====================================================
lowper = 25;
mean = 50;
highper = 75;
percentileSsrk4 = np.ones((N+1,3));
percentileIsrk4 = np.ones((N+1,3));
percentileRsrk4 = np.ones((N+1,3));
percentileDsrk4 = np.ones((N+1,3));

percentileSfut = np.ones((N+1,3));
percentileIfut = np.ones((N+1,3));
percentileRfut = np.ones((N+1,3));
percentileDfut = np.ones((N+1,3));

settings_entry1 = 1;

for j in np.linspace(t0,t0+N,N+1):
    percentileSsrk4[j.astype(int)] = np.percentile(np.transpose(Ssrk4_2[j.astype(int),:]),[lowper,mean,highper]);
    percentileIsrk4[j.astype(int)] = np.percentile(np.transpose(Isrk4_2[j.astype(int),:]),[lowper,mean,highper]);
    percentileRsrk4[j.astype(int)] = np.percentile(np.transpose(Rsrk4_2[j.astype(int),:]),[lowper,mean,highper]);
    percentileDsrk4[j.astype(int)] = np.percentile(np.transpose(Dsrk4_2[j.astype(int),:]),[lowper,mean,highper]);

guiinput2 = [settings_entry1,
        userinput_entry11,userinput_entry12,userinput_entry13,
        userinput_entry21,userinput_entry22,userinput_entry23,
        userinput_entry31,userinput_entry32,userinput_entry33,
        userinput_entry41,userinput_entry42,userinput_entry43,
        userinput_entry51,userinput_entry52,userinput_entry53,
        userinput_entry61,userinput_entry62,userinput_entry63,
        userinput_entry71,userinput_entry72,userinput_entry73,
        userinput_entry81,userinput_entry82,userinput_entry83]

Sfut_2, Ifut_2, Rfut_2, Dfut_2 = SIRD_func(T,dt,N,M,percentileSsrk4[N,1],percentileIsrk4[N,1],percentileRsrk4[N,1],percentileDsrk4[N,1],Ptotal,guiinput2)

for j in np.linspace(t0,t0+N,N+1):
    percentileSfut[j.astype(int)] = np.percentile(np.transpose(Sfut_2[j.astype(int),:]),[lowper,mean,highper]);
    percentileIfut[j.astype(int)] = np.percentile(np.transpose(Ifut_2[j.astype(int),:]),[lowper,mean,highper]);
    percentileRfut[j.astype(int)] = np.percentile(np.transpose(Rfut_2[j.astype(int),:]),[lowper,mean,highper]);
    percentileDfut[j.astype(int)] = np.percentile(np.transpose(Dfut_2[j.astype(int),:]),[lowper,mean,highper]);

futpltxend = max(datetoint(userinput_entry12),datetoint(userinput_entry22),datetoint(userinput_entry32),datetoint(userinput_entry42),datetoint(userinput_entry52),datetoint(userinput_entry62),datetoint(userinput_entry72),datetoint(userinput_entry82))
futpltxlim = futpltxend-T;
        
fig_SIRD_projection = make_subplots(rows = 6, cols = 1,specs=[[{'rowspan':5}],[None],[None],[None],[None],[{'rowspan':1}]])

fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileSsrk4[:,0] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),visible='legendonly',legendgroup = 'Susceptible'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileSsrk4[:,2] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(47,79,79,0.5)',visible='legendonly',legendgroup = 'Susceptible'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileSsrk4[:,1] , name = "Susceptible", marker = dict(color = 'darkslategray'),visible='legendonly',legendgroup = 'Susceptible'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = dt_fulldata['susceptible'], mode='lines',name = "Susceptible data", line = dict(color = 'darkkhaki', width = 3, dash = 'dash'),visible='legendonly',legendgroup = 'Susceptible'), row =1, col=1)

fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileSfut[:,0] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),visible='legendonly',legendgroup = 'Susceptible'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileSfut[:,2] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(47,79,79,0.5)',visible='legendonly',legendgroup = 'Susceptible'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileSfut[:,1] , showlegend= False, name = "Susceptible", marker = dict(color = 'darkslategray'),visible='legendonly',legendgroup = 'Susceptible'), row =1, col=1)

fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileIsrk4[:,0] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Infected'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileIsrk4[:,2] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(123,104,238,0.5)',legendgroup = 'Infected'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileIsrk4[:,1] ,name = "Infected",marker = dict(color = 'mediumslateblue'),legendgroup = 'Infected'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = dt_fulldata['cases_active'], mode='lines',name = "Infected data", line = dict(color = 'darkorchid', width =3, dash ='dash'),legendgroup = 'Infected'), row =1, col=1)

fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileIfut[:,0] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Infected'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileIfut[:,2] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(123,104,238,0.5)',legendgroup = 'Infected'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileIfut[:,1] , showlegend= False,name = "Infected",marker = dict(color = 'mediumslateblue'),legendgroup = 'Infected'), row =1, col=1)

fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileRsrk4[:,0] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Recovered'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileRsrk4[:,2] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(124,252,0,0.5)',legendgroup = 'Recovered'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileRsrk4[:,1] ,name = "Recovered",marker = dict(color = 'lawngreen'),legendgroup = 'Recovered'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = dt_fulldata['cases_recovered'], mode='lines', name = "Recovered data", line = dict(color = 'olive', width = 3, dash='dash'),legendgroup = 'Recovered'), row =1, col=1)

fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileRfut[:,0] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Recovered'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileRfut[:,2] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(124,252,0,0.5)',legendgroup = 'Recovered'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileRfut[:,1] , showlegend= False,name = "Recovered",marker = dict(color = 'lawngreen'),legendgroup = 'Recovered'), row =1, col=1)

fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileDsrk4[:,0] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Death'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileDsrk4[:,2] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(128, 0, 0,0.5)',legendgroup = 'Death'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = percentileDsrk4[:,1] ,name = "Death",marker = dict(color = 'maroon'),legendgroup = 'Death'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y = dt_fulldata['deaths_total'], mode='lines',name = "Death data", line = dict(color = 'tomato', width =3, dash = 'dash'),legendgroup = 'Death'), row =1, col=1)

fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileDfut[:,0] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),legendgroup = 'Death'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileDfut[:,2] , showlegend= False ,marker = dict(color = 'rgba(0,0,0,0)'),fill='tonexty', fillcolor = 'rgba(128, 0, 0,0.5)',legendgroup = 'Death'), row =1, col=1)
fig_SIRD_projection.add_trace(go.Scatter(x = futplot_t[:futpltxlim], y = percentileDfut[:,1] , showlegend= False,name = "Death",marker = dict(color = 'maroon'),legendgroup = 'Death'), row =1, col=1)

fig_SIRD_projection.add_trace(go.Scatter(x = pastplot_t, y= dt_mcmcresult['beta']/(dt_mcmcresult['r']+dt_mcmcresult['d']),showlegend= False,line = dict(color = 'mediumaquamarine'),name = 'R_{0}'),row =6, col=1)
Idatalist = np.hstack((dt_fulldata['cases_active'].to_numpy(),percentileIfut[:,1]));
Idiff = ((Idatalist[:-1]/Idatalist[1:])-1)*7+1
fig_SIRD_projection.add_trace(go.Scatter(x = fullplot_t[:T+futpltxlim], y= Idiff[:T+futpltxlim],showlegend= False,line = dict(color = 'grey'),name = 'R_{0}'),row =6, col=1)

fig_SIRD_projection.update_layout(height = 700)
fig_SIRD_projection.update_layout(legend = dict(orientation="h",yanchor = 'bottom', xanchor = 'center',x=0.5, y = -0.2),legend_traceorder="normal")
fig_SIRD_projection.update_layout(margin=dict(l=20, r=20, t=20, b=20))
fig_SIRD_projection['layout']['yaxis']['title']='Cases'
fig_SIRD_projection['layout']['yaxis2']['title']='Basic Reproduction Number'
fig_SIRD_projection['layout']['xaxis2']['title']='t(days)'
#fig_SIRD_projection.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
#fig_SIRD_projection.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)

fig_beta_hist = go.Figure()
fig_beta_hist.add_trace(go.Histogram(x=dt_mcmcresult['beta'],showlegend= False,
    xbins=dict(start=0,end=0.03,size=0.01),marker_color='#5A5A5A',opacity=0.75))
fig_beta_hist.add_trace(go.Histogram(x=dt_mcmcresult['beta'],name = 'Strict NPI',
    xbins=dict(start=0.03,end=0.06,size=0.01),marker_color='#BFFF00',opacity=0.75))
fig_beta_hist.add_trace(go.Histogram(x=dt_mcmcresult['beta'],name = 'Mild NPI',
    xbins=dict(start=0.06,end=0.08,size=0.01),marker_color='#00FFFF',opacity=0.75))
fig_beta_hist.add_trace(go.Histogram(x=dt_mcmcresult['beta'],name = 'Loose NPI',
    xbins=dict(start=0.08,end=0.13,size=0.01),marker_color='#C70039',opacity=0.75))
fig_beta_hist.add_trace(go.Histogram(x=dt_mcmcresult['beta'],showlegend= False,
    xbins=dict(start=0.13,end=0.16,size=0.01),
    marker_color='#5A5A5A',opacity=0.75))
fig_beta_hist.update_layout(height=300)
fig_beta_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20),xaxis_title="Infectious rate",yaxis_title="Frequency",)
#fig_beta_hist.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
#fig_beta_hist.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)

fig_rd_trend = make_subplots(specs=[[{"secondary_y": True}]])
fig_rd_trend.add_trace(go.Scatter(x = pastplot_t, y= dt_mcmcresult['r'],showlegend= False,line = dict(color = 'lawngreen')),secondary_y = False)
fig_rd_trend.add_trace(go.Scatter(x = pastplot_t, y = log_func(data_t,*pars),showlegend= False,mode = 'lines', line = dict(color = 'olive', width = 3, dash='dash')), secondary_y = False)
fig_rd_trend.add_trace(go.Scatter(x = pastplot_t, y= dt_mcmcresult['d'],showlegend= False,line = dict(color = 'maroon')), secondary_y = True)
fig_rd_trend.add_trace(go.Scatter(x = pastplot_t, y = log_func(data_t,*pars2),showlegend= False,mode = 'lines', line = dict(color = 'tomato', width = 3, dash='dash')),secondary_y = True)
fig_rd_trend.update_layout(height=300)
fig_rd_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20))
#fig_rd_trend.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
#fig_rd_trend.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
fig_rd_trend.update_layout(
    legend=dict(orientation="h"),
    yaxis=dict(
        title=dict(text= '<b> Recovery rate </b>'),
        side="left",
        titlefont=dict(
            color="olive"
        ),
    ),
    yaxis2=dict(
        title=dict(text='<b> Fatality rate </b>'),
        side="right",
        overlaying="y",
        tickmode="sync",
        titlefont=dict(
            color="tomato"
        ),
    ),
)

fig_beta_hist.update_layout(legend = dict(y = 0.99, x = 0.01),legend_traceorder="normal")
fig_beta_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20),)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Malaysia", "Singapore", "US", "Italy","China"])
#tab1 = st.tabs(["Malaysia"])


with tab1:
    with st.container():
        col1, col2 = st.columns([10, 3.5], gap = 'small')
        with col1:
            st.markdown(""" <style> .font {
            font-size:30px ; font-family: 'courier bold'; color: "white";} 
            </style> """, unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center'; class='font'>SIRD Numerical Projection</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown(""" <style> .font {
            font-size:30px ; font-family: 'courier bold'; color: "white";} 
            </style> """, unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center'; class='font'>Parameter Estimation Results</h1>", unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns([10, 3.5], gap = 'small')
        col1.plotly_chart(fig_SIRD_projection, use_container_width=True)
        col2.plotly_chart(fig_beta_hist, use_container_width=True)
        col2.plotly_chart(fig_rd_trend, use_container_width=True)
        
    with st.container():
        col1, col2 = st.columns([10, 3.5], gap = 'small')
