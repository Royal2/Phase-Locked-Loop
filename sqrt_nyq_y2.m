function h4=sqrt_nyq_y2(f_smpl,alpha,n_sym,flg_plot)
%function hh=sqrt_nyq_y2(f_smpl,alpha,n_sym,flg_plot)
% if flg_plt=1 plots impulse response and spectrum
%
%  hh=sqrt_nyq_y2(4,0.25,12,1) 
%  sqrt-nyq filter, alpha=0.25,12-samples/symbol, 
%  6-symbols delay from start to center of filter
%  filter coefficients normalized to unity amplitude for modulator
% n_sym=10;
% f_smpl=4;
% alpha=0.25;
% flg_plot=1;
arg=-n_sym:1/f_smpl:n_sym;
N=length(arg);
NN=400*f_smpl;
H1=zeros(1,NN);
H1((1+NN/2)+(-NN/(2*f_smpl):NN/(2*f_smpl)))=[0.5 ones(1,-1+NN/f_smpl) 0.5];
%M=alpha*NN/4;
M=alpha*400;
if M-2*floor(M/2)==0;
    M=M+1;
end
W1=kaiser(M,12)';
W1=W1/sum(W1);
H2=conv(H1,W1,'same');
H3=fftshift(sqrt(H2));

h3=real(fftshift(ifft(H3)));
h3=h3/max(h3);

h4=h3((1+NN/2)+(-(N-1)/2:(N-1)/2));

if flg_plot==1
figure(1)
subplot(3,1,1)
plot((-0.5:1/NN:0.5-1/NN)*f_smpl,H1,'k','linewidth',2)
hold on
plot((-0.5:1/NN:0.5-1/NN)*f_smpl,H2,'linewidth',2)
hold off
grid on
axis([-f_smpl/2 f_smpl/2 -0.1 1.2])
title('Nyquist Spectrum')
xlabel('Frequency')
ylabel('Magnitude')
text(-0.2,0.7,['\alpha = ',num2str(alpha)])

subplot(3,1,2)
plot((-0.5:1/NN:0.5-1/NN)*f_smpl,H1,'k','linewidth',2)
hold on
plot((-0.5:1/NN:0.5-1/NN)*f_smpl,fftshift(H3),'linewidth',2)
hold off
grid on
axis([-f_smpl/2 f_smpl/2 -0.1 1.2])
title('SQRT Nyquist Spectrum')
xlabel('Frequency')
ylabel('Magnitude')

% h3=real(fftshift(ifft(H3)));
% h3=h3/max(h3);

subplot(3,1,3)
plot(-NN/2:-1+NN/2,h3,'linewidth',2,'linewidth',2)
grid on
axis([-4-N/2 4+N/2 -0.3 1.2])
title('SQRT Nyquist Filter Impulse Response')
xlabel('Time Index')
ylabel('Amplitude')

% h4=h3((1+NN/2)+(-(N-1)/2:(N-1)/2));
figure(2)
subplot(2,1,1)
plot(0:N-1,h4,'-o','linewidth',2)
grid on
axis([-2 N+2 -0.3 1.2])
title('SQRT Nyquist Filter Impulse Response')
xlabel('Time Index')
ylabel('Amplitude')

subplot(2,1,2)
plot((-0.5:1/1000:0.5-1/1000)*f_smpl,fftshift(20*log10(abs(fft(h4/sum(h4),1000)))),'linewidth',2)
grid on
axis([-f_smpl/2 f_smpl/2 -80 10])
title('SQRT Nyquist Spectrum')
xlabel('Frequency')
ylabel('Magnitude')

figure(3)
hh=conv(h4,h4)/(h4*h4');
subplot(2,1,1)
plot(-(N-1):(N-1),hh,'linewidth',2)
hold on
plot(-(N-1):f_smpl:(N-1),hh(1:f_smpl:2*N),'ro','linewidth',2)
hold off
grid on
axis([-N N -0.3 1.2])
title('SQRT Nyquist Filter Matched Filter Response')
xlabel('Time Index')
ylabel('Amplitude')

hh=conv(h4,h4)/(h4*h4');
subplot(2,1,2)
plot(-(N-1):(N-1),hh,'linewidth',2)
hold on
plot(-(N-1):f_smpl:(N-1),hh(1:f_smpl:2*N),'ro','linewidth',2)
hold off
grid on
axis([-N N -0.001 0.001])
title('Zoom to Zero Crossings Matched Filter Response')
xlabel('Time Index')
ylabel('Amplitude')
end

