function rtrl_speech(audiodata,labels,class1l,class2l)

labele = readtable(labels);%'all_labels_aug.txt');
labele = table2array(labele(:,2));
labele = categorical(labele);

num_monte = 3;
num_files = 200;
ds = 500;
path_directory = audiodata; 
original_files = dir([path_directory '/*.wav']);
class1 = categorical({class1l});
class2 = categorical({class2l});

%Defining parameters Danilo's book
p=4; %input taps
N=2;%number of neurons
alpha=0.01;%learning rate
%len=5000; %length of input sample
max_len = 15000;
len = max_len/ds;

%Initialization of weight parameter
w1=zeros(p+N+1,1)+i*zeros(p+N+1,1); % weights of neuron1 +1 for bias yaa...
w2=zeros(p+N+1,1)+i*zeros(p+N+1,1); % weights of neuron2
W=zeros(p+N+1,N)+i*zeros(p+N+1,N); %total weights matrix

%Initialization of weight change
dW1=zeros(1,p+N+1)+i*zeros(1,p+N+1); %weight change +1 for bias also
dW2=zeros(1,p+N+1)+i*zeros(1,p+N+1); %weight change
DW=zeros(N,p+N+1)+i*zeros(N,p+N+1);

%PI for split case 
PII_old=zeros(p+N+1,N,N);%previous sample imaginary
PII_new=zeros(p+N+1,N,N);%current sample imaginary
PRR_old=zeros(p+N+1,N,N);%previous sample real
PRR_new=zeros(p+N+1,N,N);%current sample real
PRI_old=zeros(p+N+1,N,N);%previous sample imaginary
PRI_new=zeros(p+N+1,N,N);%current sample imaginary
PIR_old=zeros(p+N+1,N,N);%previous sample real
PIR_new=zeros(p+N+1,N,N);%current sample real

%Output Matrix
Y_old1=zeros(len+1,1)+i*zeros(len+1,1); % Previous output matrix 1
Y_out1=zeros(len,1)+i*zeros(len,1); % output matrix 1
Y_old2=zeros(len+1,1)+i*zeros(len+1,1); % Previous output matrix 2
Y_out2=zeros(len,1)+i*zeros(len,1); % output matrix 2

%Preparing the basic format fo the input signal
x=zeros(p,1)+i*zeros(p,1);%input signal to the tap
E_dB=zeros(1,len);
Elog=zeros(1,len);

%weight initialization
W_init1real=0.01*rand(p+N+1,1); %initialise the weight1 randomly
W_init1imag=0.01*rand(p+N+1,1); 
w1=W_init1real+i*W_init1imag;
W_init2real=0.01*rand(p+N+1,1); %initialise the weight2 randomly
W_init2imag=0.01*rand(p+N+1,1);
w2=W_init2real+i*W_init2imag;
kx = 1

for f=1:num_files %length(original_files)

filename=[path_directory '/' original_files(f).name];
[inputo1,fs]=audioread(filename);

if size(inputo1,1)>=max_len && (labele(f)==class1 || labele(f)==class2)

kx = kx + 1

inputo = filter_input(inputo1(1:max_len,1));
inputod = downsample(inputo,ds);
input = fft(inputod)';

for monte=1:num_monte
    monte;
    

%Load Data of Complex Colored Input
d=input(1:len);
xin(1)=0;
xin(2:len)=d(1:len-1);
Ypred = d(1);
%Activity of the Neurons
for k=1:len

    x=[xin(k);x(1:p-1)]; %input of the system due to input 
    Uin=[Y_old1(k);Y_old2(k);1;x]; %the main input to the system, 1 represents the bias

    %First Neuron Activity
    Vout1=w1.'*Uin;
    Sr1=real(w1).'*real(Uin)-imag(w1).'*imag(Uin);
    Si1=imag(w1).'*real(Uin)+real(w1).'*imag(Uin);
    sig_function1R = 1 ./ ( 1 + exp((-(Sr1))));%Output of the neuron 1 real
    sig_function1I= 1 ./ ( 1 + exp( (-(Si1))));%Output of neuron 1 imag
    sig_function1= sig_function1R + i*sig_function1I;
    sig_function_der1R = sig_function1R .* ( 1 - sig_function1R );%the derivative,f'
    sig_function_der1I = sig_function1I .* ( 1 - sig_function1I );
    Y_out1(k)=sig_function1;%store in the output matrix of 1st neuron
    Y_old1(k+1)=Y_out1(k);
    
    %Output of 1st Neuron
    u1(k)=sig_function1R;
    v1(k)=sig_function1I;
    
    %Second Neuron Activity
    Vout2=w2.'*Uin;
    Sr2=real(w2).'*real(Uin)-imag(w2).'*imag(Uin);
    Si2=imag(w2).'*real(Uin)+real(w2).'*imag(Uin);
    sig_function2R = 1 ./ ( 1 + exp( -(Sr2)));%Output of the neuron 1 real
    sig_function2I= 1 ./ ( 1 + exp( -(Si2)));%Output of neuron 1 imag
    sig_function2= sig_function2R + i*sig_function2I;
    sig_function_der2R = sig_function2R .* ( 1 - sig_function2R );%the derivative,f'
    sig_function_der2I = sig_function2I .* ( 1 - sig_function2I );
    Y_out2(k)=sig_function2;%store in the output matrix of 1st neuron
    Y_old2(k+1)=Y_out2(k);
    
    %Error Calculation
    e(:,k) = d(:,k) - Y_out1(k);
    e_real(k)=real(d(k))-u1(k);
    e_imag(k)=imag(d(k))-v1(k);
    Ypred = [Ypred;complex(u1(k),v1(k))];


    %2nd output error calculation
    
    %MSE Error Calculation
    E(k)=(1/2)*((e_real(k)).^2+(e_imag(k)).^2);
    E_dB(k)=10*log10(E(k));%error value at k step
    
    %Matrix ready before calculating Pij
    sig_function_derR=[sig_function_der1R,sig_function_der2R];
    sig_function_derI=[sig_function_der1I,sig_function_der2I];
    %misinterpret formula for splitcomplex; will be back again!
    %Calculating Pij
    for l=1:N%k
        for t=1:N%destination
            for j=1:p+N+1%source
                tempRR=0;
                tempII=0;
                tempRI=0;
                tempIR=0;
                for r=1:N%it is typical to sum both Pij value
                tempRR=tempRR+((real(W(r,l))).*(PRR_old(j,t,r))-imag(W(r,l)).*(PIR_old(j,t,r)));
                tempII=tempII+((real(W(r,l))).*(PII_old(j,t,r))+imag(W(r,l)).*(PRI_old(j,t,r)));
                tempRI=tempRI+((real(W(r,l))).*(PRI_old(j,t,r))-imag(W(r,l)).*(PII_old(j,t,r)));
                tempIR=tempIR+((real(W(r,l))).*(PIR_old(j,t,r))+imag(W(r,l)).*(PRR_old(j,t,r)));
                end
                if l==t
                    tempRR=tempRR+real(Uin(j));
                    tempII=tempII+real(Uin(j));
                    tempRI=tempRI-imag(Uin(j));
                    tempIR=tempIR+imag(Uin(j));
                else
                    tempRR=tempRR;
                    tempII=tempII;
                    tempRI=tempRI;
                    tempIR=tempIR;
                    
                end
                PRR_new(j,t,l)=sig_function_derR(l).*tempRR;
                PII_new(j,t,l)=sig_function_derI(l).*tempII;
                PRI_new(j,t,l)=sig_function_derR(l).*tempRI;
                PIR_new(j,t,l)=sig_function_derI(l).*tempIR;
                
            end
        end
    end
    PRR_old=PRR_new;
    PII_old=PII_new;
    PRI_old=PRI_new;
    PIR_old=PIR_new;
    
    %weight change
    DW=alpha.*(e_real(k).*PRR_new(:,:,1)+e_imag(k).*PIR_new(:,:,1))+i*(e_real(k).*PRI_new(:,:,1)+e_imag(k).*PII_new(:,:,1));
    
    sigma = sqrt(2 * alpha);
    noise1 = sigma * (randn(size(w1)) + 1i * randn(size(w1)));
    noise2 = sigma * (randn(size(w2)) + 1i * randn(size(w2)));

    
    w1=w1+DW(:,1);
    w2=w2+DW(:,2);

    W=[w1,w2];
 
end%k

Elog=Elog+E_dB;

end%monte
Elog=Elog/monte;

end 

end%files

%testing

% save wts_eq.mat 'w1' 'w2'
load('wts_eq_ang_fru.mat'); % comment this if training

kx = 1;
Y_old1=zeros(len+1,1)+i*zeros(len+1,1); % Previous output matrix 1
Y_old2=zeros(len+1,1)+i*zeros(len+1,1); % Previous output matrix 2

for f=2:length(original_files)

if (labele(f) == class1 || labele(f)==class2) && (labele(f-1) == class1 || labele(f-1) == class2)

filename=[path_directory '/' original_files(f-1).name];
[audio1,fs]=audioread(filename);
filename=[path_directory '/' original_files(f).name];
[audio2,fs]=audioread(filename);

if labele(f-1) == labele(f) 
    allchange(kx) = 0;
    inputo = filter_input([audio1(:,1); audio2(:,1)]);
else 
    allchange(kx) = 1;
    inputo = filter_input([audio1(:,1); audio2(:,1)]);
end

inputod = downsample(inputo,ds);
input = fft(inputod)';

%Load Data of Complex Colored Input
d=input;
Ypred = d(1);

%Activity of the Neurons
for k=1:2*len

    if k == 1
       x = d(1:p)';
    else
        if k < p
             x=[d(k+1:k+p-size(Ypred,1))';Ypred];
        else
             x = Ypred(k-p+1:k);
        end
    end  

    Uin=[Y_old1(k);Y_old2(k);1;x]; %the main input to the system, 1 represents the bias

    %First Neuron Activity
    Vout1=w1.'*Uin;
    Sr1=real(w1).'*real(Uin)-imag(w1).'*imag(Uin);
    Si1=imag(w1).'*real(Uin)+real(w1).'*imag(Uin);
    sig_function1R = 1 ./ ( 1 + exp((-(Sr1))));%Output of the neuron 1 real
    sig_function1I= 1 ./ ( 1 + exp( (-(Si1))));%Output of neuron 1 imag
    sig_function1= sig_function1R + i*sig_function1I;
    sig_function_der1R = sig_function1R .* ( 1 - sig_function1R );%the derivative,f'
    sig_function_der1I = sig_function1I .* ( 1 - sig_function1I );
    Y_out1(k)=sig_function1;%store in the output matrix of 1st neuron
    Y_old1(k+1)=Y_out1(k);
    
    %Output of 1st Neuron
    u1(k)=sig_function1R;
    v1(k)=sig_function1I;
    
    %Second Neuron Activity
    Vout2=w2.'*Uin;
    Sr2=real(w2).'*real(Uin)-imag(w2).'*imag(Uin);
    Si2=imag(w2).'*real(Uin)+real(w2).'*imag(Uin);
    sig_function2R = 1 ./ ( 1 + exp( -(Sr2)));%Output of the neuron 1 real
    sig_function2I= 1 ./ ( 1 + exp( -(Si2)));%Output of neuron 1 imag
    sig_function2= sig_function2R + i*sig_function2I;
    sig_function_der2R = sig_function2R .* ( 1 - sig_function2R );%the derivative,f'
    sig_function_der2I = sig_function2I .* ( 1 - sig_function2I );
    Y_out2(k)=sig_function2;%store in the output matrix of 1st neuron
    Y_old2(k+1)=Y_out2(k);
    
    Ypred = [Ypred;complex(u1(k),v1(k))];
end

allslope(kx,:) = max(angle(ifft(Ypred)));
kx = kx + 1;

end
end
       
dlmwrite('allchange.txt',allchange');
dlmwrite('allslope.txt',allslope);

end

