clear all; close all; clc
%{
===== Modem Design Problem Set #2 =====
Problem 1 - PLL exercise for a noisy input sinusoid
Problem 2 - PLL exercise for an I-Q modulated signal with a fixed phase offset
%}
%% Problem 1, Part a
% phase lock loop demo
% loop parameters
% digital bandwidth: theta_0 = w_0*T/2, 
% damping factor: eta = sqrt(2)/2.0
clear all; close all; clc
theta_0= 2*pi/200;  %loop bandwidth
eta=sqrt(2)/2;      %damping factor
k_i= (4*theta_0*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %Proportional constant
k_p= (4*eta*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %Integral constant
 
% input data
dphi_in=0.025;
dphi_0=ones(1,1200)*dphi_in;
phi_0=filter(1,[1 -1],dphi_0);  %forming phase series
 
s_0=exp(1j*2*pi*phi_0);  %forming complex sinusoid
%s_0=s_0+0.3*(randn(1,1200)+1j*randn(1,1200));  %adding noise to s_0

%initializing and predefining below array sizes
phi_err_sv=zeros(1,1200);
int_sv=zeros(1,1200);
dphi_sv=zeros(1,1200);
accum_sv=zeros(1,1200);
int=0;
accum=0;

%{
PLL for sinusoidal signal s_0
%atan
%PI-loop filter
%DDS
%multiplier
%}
for nn=1:1200
% heterodyne and phase detector
    prod=s_0(nn)*exp(-1j*2*pi*accum);    %down-convert input
    dphi=angle(prod)/(2*pi);    %measuring angle error
    phi_err_sv(nn)=dphi;    %phase error
% loop filter
    int=int+k_i*dphi;       %integrator
    dphi_2=k_p*dphi+int;    %proportional
    dphi_sv(nn)=dphi_2;     %PI loop filter output
% dds
    accum_sv(nn)=accum;     %saving accumulator
    accum=accum+dphi_2;     %incrementing accumulator
end 

%plotting phase of input signal and the phase profile of the PLL phase
%accumulator
figure(1);
plot((unwrap(angle(s_0))/(2*pi)),'linewidth',2); %phase of input signal
hold on; grid on;
plot(accum_sv,'r','linewidth',2); % phase profile of PLL phase accumulator
title('Input and Output Phase Profiles');
xlabel('Time Index'); ylabel('Phase Angle, $\frac{\theta(n)}{2\pi}$','Interpreter','latex');
xlim([0 400]); ylim([0 10]);
legend('input', 'output');

%plotting real and imaginary part of input and output sinusoids
%two subplots
figure(2);
subplot(2,1,1);
plot(real(s_0),'linewidth',2);   %real part of input
grid on; hold on;
%exp(1j*2*pi*accum_sv);  %output signal of complex sinusoid using output phase
plot(real(exp(1j*2*pi*accum_sv)),'r','linewidth',2);  %real part of output
title('Real Part of Input and Output Signal');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([-1.25 1.25]);
legend('input', 'output');

subplot(2,1,2) 
plot(imag(s_0),'linewidth',2);   %imaginary part of input
grid on; hold on;
plot(imag(exp(1j*2*pi*accum_sv)),'r','linewidth',2);    %imaginary part of output
title('Imaginary Part of Input and Output Signal');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([-1.25 1.25]);
legend('input', 'output');

%plotting PLL phase error and loop filter output
%two subplots
figure(3);
subplot(2,1,1);
plot(phi_err_sv,'linewidth',2);
grid on;
title('Phase Error');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([-0.2 0.4]);

subplot(2,1,2);
plot(dphi_sv,'linewidth',2);
grid on;
title('Loop Filter Output');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([0 0.04]);

%plotting log mag(dB) Kaiser-Bessel windowed spectrum (last 1k samples of input and output sinusoids)
%two subplots
figure(4);

N=1000;
w=kaiser(N,0.5)';   %kaiser bessel windowing
w=w/sum(w);     %normalizing

subplot(2,1,1);
fft_s0=fftshift(20*log10(abs(fft(s_0(end-N+1:end).*w,N)))); %last 1k samples
plot(-0.5:1/N:0.5-1/N,fft_s0,'linewidth',2);   %Input spectrum kaiser bessel
grid on;
title('Spectrum, PLL Input Sinusoid');
xlabel('Frequency'); ylabel('Log Mag (dB)');
xlim([-0.25 0.25]); ylim([-50 10]);

subplot(2,1,2);
y=exp(1j*2*pi*accum_sv);  %output signal
fft_y=fftshift(20*log10(abs(fft(y(end-N+1:end).*w,N))));    %last 1k samples
plot(-0.5:1/N:0.5-1/N,fft_y,'r','linewidth',2);     %output spectrum, kaiser bessel
grid on;
title('Spectrum, PLL Output Sinusoid');
xlabel('Frequency'); ylabel('Log Mag (dB)');
xlim([-0.25 0.25]); ylim([-50 10]);

%% Problem 1, part b
%repeat 'part a' using a noisy version of s_0
%s_0=s_0+0.3*(randn(1,1200)+j*randn(1,1200)); %noisy sinusoid
clear all; close all; clc
theta_0= 2*pi/200;  %loop bandwidth
eta=sqrt(2)/2;      %damping factor
k_i= (4*theta_0*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %Proportional constant
k_p= (4*eta*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %Integral constant
 
% input data
dphi_in=0.025;
dphi_0=ones(1,1200)*dphi_in;
phi_0=filter(1,[1 -1],dphi_0);  %forming phase series
 
s_0=exp(1j*2*pi*phi_0);  %forming complex sinusoid
s_0=s_0+0.3*(randn(1,1200)+1j*randn(1,1200));  %adding noise to s_0

%initializing and predefining below array sizes
phi_err_sv=zeros(1,1200);
int_sv=zeros(1,1200);
dphi_sv=zeros(1,1200);
accum_sv=zeros(1,1200);
int=0;
accum=0;

%{
PLL for sinusoidal signal s_0
%atan
%PI-loop filter
%DDS
%multiplier
%}
for nn=1:1200
% heterodyne and phase detector
    prod=s_0(nn)*exp(-1j*2*pi*accum);    %down-convert input
    dphi=angle(prod)/(2*pi);    %measuring angle error
    phi_err_sv(nn)=dphi;    %phase error
% loop filter
    int=int+k_i*dphi;       %integrator
    dphi_2=k_p*dphi+int;    %proportional
    dphi_sv(nn)=dphi_2;     %PI loop filter output
% dds
    accum_sv(nn)=accum;     %saving accumulator
    accum=accum+dphi_2;     %incrementing accumulator
end 

%plotting phase of input signal and the phase profile of the PLL phase
%accumulator
figure(1);
plot((unwrap(angle(s_0))/(2*pi)),'linewidth',2); %phase of input signal
hold on; grid on;
plot(accum_sv,'r','linewidth',2); % phase profile of PLL phase accumulator
title('Input and Output Phase Profiles');
xlabel('Time Index'); ylabel('Phase Angle, $\frac{\theta(n)}{2\pi}$','Interpreter','latex');
xlim([0 400]); ylim([0 10]);
legend('input', 'output');

%plotting real and imaginary part of input and output sinusoids
%two subplots
figure(2);
subplot(2,1,1);
plot(real(s_0),'linewidth',2);   %real part of input
grid on; hold on;
%exp(1j*2*pi*accum_sv);  %output signal of complex sinusoid using output phase
plot(real(exp(1j*2*pi*accum_sv)),'r','linewidth',2);  %real part of output
title('Real Part of Input and Output Signal');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([-1.25 1.25]);
legend('input', 'output');

subplot(2,1,2) 
plot(imag(s_0),'linewidth',2);   %imaginary part of input
grid on; hold on;
plot(imag(exp(1j*2*pi*accum_sv)),'r','linewidth',2);    %imaginary part of output
title('Imaginary Part of Input and Output Signal');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([-1.25 1.25]);
legend('input', 'output');

%plotting PLL phase error and loop filter output
%two subplots
figure(3);
subplot(2,1,1);
plot(phi_err_sv,'linewidth',2);
grid on;
title('Phase Error');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([-0.2 0.4]);

subplot(2,1,2);
plot(dphi_sv,'linewidth',2);
grid on;
title('Loop Filter Output');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([0 0.04]);

%plotting log mag(dB) Kaiser-Bessel windowed spectrum (last 1k samples of input and output sinusoids)
%two subplots
figure(4);

N=1000;
w=kaiser(N,0.5)';   %kaiser bessel windowing
w=w/sum(w);     %normalizing

subplot(2,1,1);
fft_s0=fftshift(20*log10(abs(fft(s_0(end-N+1:end).*w,N)))); %last 1k samples
plot(-0.5:1/N:0.5-1/N,fft_s0,'linewidth',2);   %Input spectrum kaiser bessel
grid on;
title('Spectrum, PLL Input Sinusoid');
xlabel('Frequency'); ylabel('Log Mag (dB)');
xlim([-0.25 0.25]); ylim([-50 10]);

subplot(2,1,2);
y=exp(1j*2*pi*accum_sv);  %output signal
fft_y=fftshift(20*log10(abs(fft(y(end-N+1:end).*w,N))));    %last 1k samples
plot(-0.5:1/N:0.5-1/N,fft_y,'r','linewidth',2);     %output spectrum, kaiser bessel
grid on;
title('Spectrum, PLL Output Sinusoid');
xlabel('Frequency'); ylabel('Log Mag (dB)');
xlim([-0.25 0.25]); ylim([-50 10]);

%% Problem 1, part c
%repeart 'part b' using a reduced loop BW
%theta_0= 2*pi/400;  %reduced loop bandwidth
clear all; close all; clc
%theta_0= 2*pi/200;  %loop bandwidth
theta_0= 2*pi/400;  %reduced loop bandwidth
eta=sqrt(2)/2;      %damping factor
k_i= (4*theta_0*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %Proportional constant
k_p= (4*eta*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %Integral constant
 
% input data
dphi_in=0.025;
dphi_0=ones(1,1200)*dphi_in;
phi_0=filter(1,[1 -1],dphi_0);  %forming phase series
 
s_0=exp(1j*2*pi*phi_0);  %forming complex sinusoid
s_0=s_0+0.3*(randn(1,1200)+1j*randn(1,1200));  %adding noise to s_0

%initializing and predefining below array sizes
phi_err_sv=zeros(1,1200);
int_sv=zeros(1,1200);
dphi_sv=zeros(1,1200);
accum_sv=zeros(1,1200);
int=0;
accum=0;

%{
PLL for sinusoidal signal s_0
%atan
%PI-loop filter
%DDS
%multiplier
%}
for nn=1:1200
% heterodyne and phase detector
    prod=s_0(nn)*exp(-1j*2*pi*accum);    %down-convert input
    dphi=angle(prod)/(2*pi);    %measuring angle error
    phi_err_sv(nn)=dphi;    %phase error
% loop filter
    int=int+k_i*dphi;       %integrator
    dphi_2=k_p*dphi+int;    %proportional
    dphi_sv(nn)=dphi_2;     %PI loop filter output
% dds
    accum_sv(nn)=accum;     %saving accumulator
    accum=accum+dphi_2;     %incrementing accumulator
end 

%plotting phase of input signal and the phase profile of the PLL phase
%accumulator
figure(1);
plot((unwrap(angle(s_0))/(2*pi)),'linewidth',2); %phase of input signal
hold on; grid on;
plot(accum_sv,'r','linewidth',2); % phase profile of PLL phase accumulator
title('Input and Output Phase Profiles');
xlabel('Time Index'); ylabel('Phase Angle, $\frac{\theta(n)}{2\pi}$','Interpreter','latex');
xlim([0 400]); ylim([0 10]);
legend('input', 'output');

%plotting real and imaginary part of input and output sinusoids
%two subplots
figure(2);
subplot(2,1,1);
plot(real(s_0),'linewidth',2);   %real part of input
grid on; hold on;
%exp(1j*2*pi*accum_sv);  %output signal of complex sinusoid using output phase
plot(real(exp(1j*2*pi*accum_sv)),'r','linewidth',2);  %real part of output
title('Real Part of Input and Output Signal');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([-1.25 1.25]);
legend('input', 'output');

subplot(2,1,2) 
plot(imag(s_0),'linewidth',2);   %imaginary part of input
grid on; hold on;
plot(imag(exp(1j*2*pi*accum_sv)),'r','linewidth',2);    %imaginary part of output
title('Imaginary Part of Input and Output Signal');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([-1.25 1.25]);
legend('input', 'output');

%plotting PLL phase error and loop filter output
%two subplots
figure(3);
subplot(2,1,1);
plot(phi_err_sv,'linewidth',2);
grid on;
title('Phase Error');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([-0.2 0.5]);

subplot(2,1,2);
plot(dphi_sv,'linewidth',2);
grid on;
title('Loop Filter Output');
xlabel('Time Index'); ylabel('Amplitude');
xlim([0 400]); ylim([0 0.05]);

%plotting log mag(dB) Kaiser-Bessel windowed spectrum (last 1k samples of input and output sinusoids)
%two subplots
figure(4);

N=1000;
w=kaiser(N,0.5)';   %kaiser bessel windowing
w=w/sum(w);     %normalizing

subplot(2,1,1);
fft_s0=fftshift(20*log10(abs(fft(s_0(end-N+1:end).*w,N)))); %last 1k samples
plot(-0.5:1/N:0.5-1/N,fft_s0,'linewidth',2);   %Input spectrum kaiser bessel
grid on;
title('Spectrum, PLL Input Sinusoid');
xlabel('Frequency'); ylabel('Log Mag (dB)');
xlim([-0.25 0.25]); ylim([-50 10]);

subplot(2,1,2);
y=exp(1j*2*pi*accum_sv);  %output signal
fft_y=fftshift(20*log10(abs(fft(y(end-N+1:end).*w,N))));    %last 1k samples
plot(-0.5:1/N:0.5-1/N,fft_y,'r','linewidth',2);     %output spectrum, kaiser bessel
grid on;
title('Spectrum, PLL Output Sinusoid');
xlabel('Frequency'); ylabel('Log Mag (dB)');
xlim([-0.25 0.25]); ylim([-50 10]);

%% Problem 2
% QPSK modulation
clear all; close all; clc;
%===========================
% Problem 2, part a
%===========================
h=sqrt_nyq_y2(4,0.5,6,0);  % 49 tap shaping filter
%h=rcosine(1,4,'sqrt',0.5,6);
h=h/max(h); %normalizing
N_dat=1000; %1k symbols
x0=(floor(2*rand(1,N_dat))-0.5)/0.5+1j*(floor(2*rand(1,N_dat))-0.5)/0.5;    %generating signal
hh=reshape([h 0 0 0],4,13); %reshaping
reg=zeros(1,13); %initializing empty array
% modulator
x1=zeros(1,4*N_dat);    %initializing empty array
m=0;
for nn=1:N_dat
    reg=[x0(nn) reg(1:12)]; %modulating signal x0
    for k=1:4
        x1(m+k)=reg*hh(k,:)';   %output of shaping filter, x1
    end
    m=m+4;
end
%plotting constellation diagram for modulated signal
figure(1)
subplot(2,1,1)
plot(x1(1:4:4000),'r.')
grid on
axis('equal')
axis([-2 2 -2 2])
title('Shaping Filter Constellation')
%plotting eye diagram for modulated signal
subplot(2,1,2)
plot(0,0);
hold on
%overlaying different bit transitions to overplot eye diagram
for n=48+1:8:4*N_dat-8
    plot(-1:1/4:1,real(x1(n:n+8)),'b');
end
hold off
grid on
title('Eye Diagram Real Part Shaping Filter')
xlabel('Time Index')
ylabel('Amplitude')
%===========================
% Problem 2, part b
%===========================
%x2=exp(1j*2*pi*1/18)*x1+0.001*(randn(1,4*N_dat)+1j*randn(1,4*N_dat));
%output of shaping filter, x1

%rotating complex input sequence
x2=exp(1j*2*pi*1/18)*x1;    %rotate
%adding complex noise, sigma*(n_x+j*n_y)
sigma=0.1;
x2=x2+sigma*(randn(1,4*N_dat)+1j*randn(1,4*N_dat)); %add noise

figure(2)
subplot(2,1,1)
plot(x2(1:4:4000),'r.')
grid on
axis('equal')
axis([-2 2 -2 2])
title('Received Phase Offset Constellation')
 
subplot(2,1,2)
plot(0,0);
hold on
%overlaying different bit transitions to overplot eye diagram
for n=48+1:8:4*N_dat-8
    plot(-1:1/4:1,real(x2(n:n+8)),'b');
end
hold off
grid on
title('Eye Diagram Real Part Received Signal with Phase Offset')
xlabel('Time Index')
ylabel('Amplitude')
%===========================
% Problem 2, part c
%===========================
%processing the time series with the matched filters
h2=h/(h*h');    %normalizing filter response
x3=conv(x2,h2); %convolving
%plotting constellation diagram for matched filter with phase offset
figure(3)
subplot(2,1,1)
plot(x3(1:4:4000),'r.') %4 samples per symbol
grid on
axis('equal')
axis([-1.5 1.5 -1.5 1.5])
title('Matched Filter Phase Offset Constellation')
%plotting eye diagram of real part of matched filter with phase offset
subplot(2,1,2)
plot(0,0);
hold on
%overlaying different bit transitions to overplot eye diagram
for n=48+1:8:4*N_dat-8
    plot(-1:1/4:1,real(x3(n:n+8)),'b'); %plotting real components of signal
end
hold off
grid on
title('Eye Diagram Real Part Matched Filter Signal with Phase Offset')
xlabel('Time Index')
ylabel('Amplitude')

%===========================
% Problem 2, part d
%===========================
%PLL for Phase Lock
theta_0= 2*pi/800;  %loop bandwidth
eta=sqrt(2)/2;  %damping factor
k_i= (4*theta_0*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %integral factor
k_p= (4*eta*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %proportional factor
 
% Initialize registers
accum=0;
int=0;
hld=0;
reg=zeros(1,49);

% Arrays for figures
accum_sv=zeros(1,4*N_dat);
err_sv=zeros(1,N_dat);
lp_sv=zeros(1,N_dat);
y=zeros(1,4*N_dat);
y_atan=zeros(1,4*N_dat);    %atan input

m=1;								% Initialize symbol clock
for n=1:4:4*N_dat-3					% 4-samples per symbol
        prd=x2(n);  % input down convert (removed feedback component)
        reg=[prd reg(1:48)];            % insert in matched filter
        yy=reg*h2';                     % compute matched filter output
        y(n)=yy;                        % save output
        if abs(yy)>0.1                 % test for signal strength    
          y_det=sign(real(yy))+1j*sign(imag(yy)); % QPSK slice yy
        else
            y_det=0+1j*0;
        end
        det_prd=conj(yy)*y_det;         % form conjugate product
        y_atan(n)=det_prd;  %save output of mixer, input to atan
        det=-angle(det_prd)/(2*pi);     % measure angle error
        err_sv(m)=det;                  % save phase error, delta_theta(n)
        int=int+k_i*det;                % loop filter integrator
        lp=int+k_p*det;                 % loop filter output
        lp_sv(m)=lp;                    % save loop filter output
        m=m+1;                          % increment symbol clock
        hld=lp/4;                       % scale lp and store in hld 
        accum_sv(n)=accum;              % save accumulator
        accum=accum+hld;                % Increment accumulator
        for k=1:3                       % process next 3 input samples
           prd=x2(n+k); % without detector or  (removed feedback component)
           reg=[prd reg(1:48)];             % or loop filter
           yy=reg*h2';
           y(n+k)=yy;
           accum_sv(n+k)=accum;     %DDS
           accum=accum+hld;
        end
end

figure(4)
%constellation diagram of input to slicer
subplot(3,1,1)
plot(y(1:4:4000),'r.')
grid on
axis('equal')
axis([-2 2 -2 2])
title('Input to Detector (Slicer) Constellation, 2d (PLL not working)')
%constellation diagram of conjugate product, input to ATAN phase detector
subplot(3,1,2)
plot(y_atan(1:4:4000),'r.')
grid on
axis('equal')
axis([-2 2 -2 2])
title('Conjugate Product Values, Input to ATAN Phase Detector, 2d (PLL not working)')
%plotting phase error in degrees
subplot(3,1,3)
v_peak=20;
v_scale=(v_peak/max(err_sv*360/(2*pi)));
plot(v_scale*err_sv*360/(2*pi),'linewidth',2);
grid on;
title('Phase Error in Degrees, 2d (PLL not working)');
xlabel('Time Index'); ylabel('Amplitude');
ylim([-10 20]);

%===========================
% Problem 2, part e&f
%===========================
%PLL for Phase Lock
theta_0= 2*pi/800;  %loop bandwidth
eta=sqrt(2)/2;  %damping factor
k_i= (4*theta_0*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %integral factor
k_p= (4*eta*theta_0)/(1+2*eta*theta_0+theta_0*theta_0); %proportional factor
 
% Initialize registers
accum=0;
int=0;
hld=0;
reg=zeros(1,49);

% Arrays for figures
accum_sv=zeros(1,4*N_dat);
err_sv=zeros(1,N_dat);
lp_sv=zeros(1,N_dat);
y=zeros(1,4*N_dat);
y_atan=zeros(1,4*N_dat);    %atan input

m=1;								% Initialize symbol clock
for n=1:4:4*N_dat-3					% 4-samples per symbol
        prd=x2(n)*exp(-1j*2*pi*accum);  % input down convert
        reg=[prd reg(1:48)];            % insert in matched filter
        yy=reg*h2';                     % compute matched filter output
        y(n)=yy;                        % save output
        if abs(yy)>0.1                 % test for signal strength    
          y_det=sign(real(yy))+1j*sign(imag(yy)); % QPSK slice yy
        else
            y_det=0+1j*0;
        end
        det_prd=conj(yy)*y_det;         % form conjugate product
        y_atan(n)=det_prd;  %save output of mixer, input to atan
        det=-angle(det_prd)/(2*pi);     % measure angle error
        err_sv(m)=det;                  % save phase error, delta_theta(n)
        int=int+k_i*det;                % loop filter integrator
        lp=int+k_p*det;                 % loop filter output
        lp_sv(m)=lp;                    % save loop filter output
        m=m+1;                          % increment symbol clock
        hld=lp/4;                       % scale lp and store in hld 
        accum_sv(n)=accum;              % save accumulator
        accum=accum+hld;                % Increment accumulator
        for k=1:3                       % process next 3 input samples
           prd=x2(n+k)*exp(-1j*2*pi*accum); % without detector or 
           reg=[prd reg(1:48)];             % or loop filter
           yy=reg*h2';
           y(n+k)=yy;
           accum_sv(n+k)=accum;     %DDS
           accum=accum+hld;
        end
end

figure(5)
%constellation diagram of input to slicer
subplot(3,1,1)
plot(y(1:4:4000),'r.')
grid on
axis('equal')
axis([-2 2 -2 2])
title('Input to Detector (Slicer) Constellation, 2f (PLL working)')
%constellation diagram of conjugate product, input to ATAN phase detector
subplot(3,1,2)
plot(y_atan(1:4:4000),'r.')
grid on
axis('equal')
axis([-2 2 -2 2])
title('Conjugate Product Values, Input to ATAN Phase Detector, 2f (PLL working)')
%plotting phase error in degrees
subplot(3,1,3)
v_peak=20;
v_scale=(v_peak/max(err_sv*360/(2*pi)));
plot(v_scale*err_sv*360/(2*pi),'linewidth',2);
grid on;
title('Phase Error in Degrees, 2f (PLL working)');
xlabel('Time Index'); ylabel('Amplitude');
ylim([-10 20]);
