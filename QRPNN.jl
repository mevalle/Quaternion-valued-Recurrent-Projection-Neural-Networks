module QRPNN

using LinearAlgebra
using Quaternions
using Random
rng = MersenneTwister(1234);

function identity(X,a)
    return X
end

function potential(X,L)
    return 1 ./ ((1 + 1.e-8 .- X) .^ L)
end

function high_order(X,q)
    return (1 .+ X) .^ q;
end

function exponential(X,alpha)
    return exp.(alpha*X)
end

function train(f,f_params,U)
    N = size(U)[1]
    C = f(Array{Float64}(real(U'*U))/N,f_params)
    return U*inv(C)
end

function hopfield(U, Up, xinput, it_max = 1.e3, verbose=true)
    ### Quaternionic Hopfield Neural Network
    Name = "Quaternionic Hopfield Neural Network"
    
    N = size(U,1)
    tau = 1.e-6

    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau
    while (Error>tau)&&(it<it_max)
        it = it+1
       
        # Compute the next state
        a = U*(Up*x)
        x = a ./ abs.(a)

        Error = norm(x-xold)
        xold = copy(x)
    end
  
    if verbose == true
        if it_max<=it
            println(Name," failed to converge in ",it_max," iterations.")
        end
    end

    return x
end

function main(f,f_params, U, V, xinput, it_max = 1.e3, verbose = true)
    (N,K) = size(U)
    tau = 1.e-6
    
    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau
    while (Error>tau)&&(it<it_max)
        it = it+1
        
        # Compute the weights
        w = f(Array{Float64}(real((U'*x)))/N,f_params);

        # Compute the next state
        a = V*w
        x = a ./ abs.(a)
        
        Error = norm(x-xold)
        xold = copy(x)
    end
    if verbose == true
        if it_max<=it
            println(split(string(f),".")[end]," QRPNN failed to converge in ",it_max," iterations.")
        end
    end
    return x
end

function CIFAR2Quat(x)
    tau = 1.e-4
    phi_ang = (-pi+tau) .+ 2*(pi-tau)*x[:,:,1]
    psi_ang = (-pi/4+tau) .+ (pi/2-2*tau)*x[:,:,2]
    theta_ang = (-pi/2+tau) .+ (pi-2*tau)*x[:,:,3]
    return Quaternion.(cos.(phi_ang),sin.(phi_ang),zeros(32,32),zeros(32,32)).*
        Quaternion.(cos.(psi_ang),zeros(32,32),zeros(32,32),sin.(psi_ang)).*
        Quaternion.(cos.(theta_ang),zeros(32,32),sin.(theta_ang),zeros(32,32));
end

function imnoise(img,noise_std = 0.1)
    return clamp.(img + noise_std*randn(rng, Float64, size(img)),0,1)
end

function Quat2CIFAR(x)
    tau = 1.e-4
    N = size(x,1)
    img_x = zeros(N,3)
    for i=1:length(x)
        q = x[i]/abs(x[i])
        
        # Take the components of q
        a = q.s;
        b = q.v1;
        c = q.v2;
        d = q.v3;
        
        # Rodriguez entries
        R11 = a.^2+b.^2-c.^2-d.^2;
        R13 = 2*(b.*d+a.*c);
        R22 = a.^2-b.^2+c.^2-d.^2;
        R32 = 2*(c.*d+a.*b);
        R12 = 2*(b.*c-a.*d);
        R23 = 2*(c.*d-a.*b);
        R33 = a.^2-b.^2-c.^2+d.^2;
        
        # Psi, Psi, and theta angles
        psiq = real(asin(-R12)/2);
        phiq   = real(atan(R32,R22)/2);
        thetaq = real(atan(R13,R11)/2);
        if abs(abs(psiq)-pi/4)<=tau
            phiq = real(atan(-R23,R33)/2);
            thetaq = 0;
        end
        
        a1   = cos(phiq).*cos(psiq).*cos(thetaq)+sin(phiq).*sin(psiq).*sin(thetaq);
        if (a1*a<0)
            if phiq>=0
                phiq = phiq - pi;
            else
                phiq = phiq + pi;
            end
        end
        R = (phiq+pi-tau)/(2*(pi-tau));
        G = (psiq+pi/4-tau)/((pi/2-2*tau));
        B = (thetaq+pi/2-tau)/((pi-2*tau));
        img_x[i,:] = [R,G,B]
    end
    return img_x
end 

end
