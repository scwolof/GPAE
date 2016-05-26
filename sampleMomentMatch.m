function [M, S] = sampleMomentMatch (gp, m, s, nsamp)
                                                
% Compute the predicted posterior distribution and its 
% derivatives for GPs with uncertain inputs. 
%
% function call:
%   [M, S] = sampleMomentMatch (gp, m, s, nsamp)
% or
%   out = sampleMomentMatch (gp, m, s, nsamp)
% where out is a struct containing all output fields listed below
%
% inputs:
% gp      Struct with fields:
%           hyp       (D+2) by E marix of log-hyper-parameters
%           inputs    n by D matrix of training inputs
%           target    n by E matrix of training targets
% m       D by 1 vector, mean of the test distribution
% s       D by D covariance matrix of the test distribution
% nsamp   (optional) Number of samples that should be used. If not
%           supplied, the number of samples will lie in the range
%           [10,100], depending on the number of training points.
%           If nsamp = 1, the mean m is assumed to be deterministic,
%           and the variance s will be zero.
%
% outputs:
% M       [E,1]         mean of pred. distribution
% S       [E,E]         covariance of the pred. distribution
% dMdm    [E,D]         mean by input mean partial derivatives
% dSdm    [E,E,D]       covariance by input mean derivatives
% dMds    [E,D,D]       mean by input covariance derivatives
% dSds    [E,E,D,D]     covariance by input covariance derivatives
% dMdi    [E,n,D]       mean by training input derivatives
% dSdi    [E,E,n,D]     covariance by training input derivatives
% dMdt    [E,n,D]       mean by training target derivatives
% dSdt    [E,E,n,E]     covariance by target derivatives
% dMdX    [E,(D+2)*E]   mean by hyper-parameters derivatives
% dSdX    [E,E,(D+2)*E] covariance by hyperparameters derivatives
%
% Copyright (C) 2016 by Simon Olofsson
% Last edited: 2016-05-20

%% Preferences

% Number of samples
if (nargin < 4)
    nsamp = max(10,min(100,round(size(gp.inputs,1)/4)));
end

%% Dimensionalities
X = gp.inputs;
Y = gp.targets;
hyp = gp.hyp;

[n,D] = size(X);
[~,E] = size(Y);

m = m(:);

% Check that everything seems to be correct
fm = 'sampleMomentMatch: ';
assert(size(Y,1) == n,...
    [fm 'Number of training points do not match (' ...
    num2str(size(Y,1)) ' ~= ' num2str(n) ')']);
assert(size(hyp,2) == E,...
    [fm 'Dim(targets) not the same as number of hyp ' ...
    'columns (' num2str(size(hyp,2)) ' ~= ' num2str(E) ')']);
assert(size(hyp,1) == D+2,...
    [fm 'Dim(inputs)+2 not the same as number of hyp ' ...
    'rows (' num2str(size(hyp,1)) ' ~= ' num2str(D+2) ')']);
assert(size(m,1) == D,...
    [fm 'Dim(mean) and dim(inputs) not the same (' ...
    num2str(size(m,1)) ' ~= ' num2str(D) ')']);
assert(nsamp == 1 || (size(s,1) == size(s,2) && size(s,1) == D),...
    [fm 'Variance (s) must be square matrix of size D (' ...
    num2str(size(s,1)) ' ~= ' num2str(D) ', or '  ...
    num2str(size(s,2)) ' ~= ' num2str(D) ')']);

%% Initialisation
M = zeros(E,1); S = zeros(E);
dMdm = zeros(E,D); dSdm = zeros(E,E,D);
dMds = zeros(E,D,D); dSds = zeros(E,E,D,D);
dMdi = zeros(E,n,D); dSdi = zeros(E,E,n,D);
dMdt = zeros(E,n,E); dSdt = zeros(E,E,n,E);
dMdX = zeros(E,D+2,E); dSdX = zeros(E,E,D+2,E);

%% Computation

% Draw sample
if (nsamp > 1)
    Ls = chol(s)';
    Ni = randn(size(s,2),nsamp);
    xi = bsxfun(@plus, m(:), Ls*Ni);
else
    % Deterministic input
    nsamp = 1; xi = m;
end

% For all output dimensions
for y = 1 : E
    % Select hyperparameters for output dimension y
    logtheta = hyp(:,y);
    
    % Faster than using gpr
    K = covSEard(logtheta(1:(D+1)), X) + ...
        covNoise(logtheta(D+2), X);
    L = chol(K); iK = solve_chol(L,eye(n));
    alpha = iK*Y(:,y);
    [Kss, Kstar] = covSEard(logtheta, X, xi');
    mu = Kstar' * alpha;
    v = L'\Kstar;
    Var = Kss - sum(v.*v)';
    
    % Mean
    M(y) = mean(mu);
    % Variance
    Evar = mean(Var); varE = var(mu);
    S(y,y) = Evar + varE;
    
    % Vector of sampled means minus M
    cmus = 2*bsxfun(@minus,mu,M(y))/(nsamp-1);
    
    % (K+sigma2*I)^-1 * k(X,x)
    iKKstar = iK*Kstar;
    % Inverse of lengthscales
    iLambda = exp(-2*logtheta(1:D));
    
    % Prepare for derivatives wrt Cholesky decomposition
    dMdLs = zeros(D); dSdLs = zeros(D);
    
    % For each input dimension
    for j = 1 : D
        invLxX2 = bsxfun(@minus,X(:,j),xi(j,:))*iLambda(j);
        dkxXdx2 = iK*(Kstar.*invLxX2);
        
        % d M / d m
        dmus = alpha'*(Kstar.*invLxX2);
        dMdm(y,j) = sum(dmus,2)/nsamp;
        % d M / d Ls - Used for dMds
        if nsamp > 1
            dmudL = bsxfun(@times,dmus,Ni);
            dMdLs(j,:) = sum(dmudL,2)'/nsamp;
        end
        
        % d S / d m
        dmus2 = sum(Kstar.*dkxXdx2,1);
        dSdm(y,y,j) = -2*sum(sum(Kstar.*dkxXdx2,1),2)/nsamp;
        if nsamp > 1
            dSdm(y,y,j) = dSdm(y,y,j) + ...
                cmus'*bsxfun(@minus,dmus',dMdm(y,j));
            % d S / d Ls - Used for dSds
            dSdLs(j,:) = -2*dmus2*Ni'/nsamp + ...
                cmus'*bsxfun(@minus,dmudL',dMdLs(j,:));
        end
        
        % d M / d i  and  d S / d i
        dKiKKstar = (repmat(iLambda(j) * (K .* ...
            bsxfun(@minus,X(:,j),X(:,j)')),nsamp,1)) .* ...
            repmat(reshape(iKKstar,n*nsamp,1),1,n);
        
        dmus = -repmat(alpha,1,nsamp).*...
            (iLambda(j)*Kstar.*bsxfun(@minus,X(:,j),xi(j,:))) ...
            - reshape(sum(reshape(dKiKKstar',n,n,nsamp),2), ...
            n,nsamp) .* repmat(alpha,1,nsamp) ...
            + reshape(dKiKKstar*alpha,n,nsamp);
        
        dMdi(y,:,j) = sum(dmus,2)/nsamp;
        
        St = 2*(iKKstar .* ...
            (iLambda(j)*Kstar.*bsxfun(@minus,X(:,j),xi(j,:))) ...
            + reshape(sum(reshape(dKiKKstar',...
            n,n,nsamp),2),n,nsamp) .* iKKstar);
        dSdi(y,y,:,j) = sum(St,2)/nsamp;
        
        if nsamp > 1
            dSdi(y,y,:,j) = reshape(dSdi(y,y,:,j),n,1) + ...
                bsxfun(@minus,dmus,reshape(dMdi(y,:,j),n,1))*cmus;
        end
        
        % d M / d Lambda and d S / d Lambda
        % ----------------------------------
        % d k_i(x_star,x_m) / d theta_j
        t1 = -(bsxfun(@minus,X(:,j),xi(j,:)).^2);
        dkidXj = (Kstar.*t1)*(-2*iLambda(j));
        % d (K+sigma*I) / d theta_j
        dKdX = iKKstar'*covSEard(logtheta(1:(D+1)),X,j);
        
        % d mu / d theta_j
        dmus = (0.5*dkidXj' - dKdX)*alpha;
        dMdX(y,j,y) = sum(dmus)/nsamp;
        
        % d Sigma_ii / d theta_j
        dSdX(y,y,j,y) = sum(sum(iKKstar.*(dKdX' - dkidXj)))/nsamp;
        if nsamp > 1
            % Takes care of gradient from var(mu)
            dSdX(y,y,j,y) = dSdX(y,y,j,y) + ...
                cmus'*bsxfun(@minus,dmus,dMdX(y,j,y));
        end
    end
    
    % d M / d t
    dmus = iK*Kstar;
    dMdt(y,:,y) = sum(dmus,2)/nsamp;
    
    % d S / d t
    if nsamp > 1
        dSdt(y,y,:,y) = bsxfun(@minus,dmus,...
            reshape(dMdt(y,:,y),n,1))*cmus;
    end
    
    % d M / d s  and  d S / d s
    if nsamp > 1
        dMdLs = tril(dMdLs);
        dMds(y,:,:) = chol_rev(Ls, dMdLs)';
        dSdLs = tril(dSdLs);
        dSds(y,y,:,:) = chol_rev(Ls, dSdLs)';
    end    
    
    % Compute d M / d sf  and  d S / d sf
    % -----------------------------------
    % d (K+sigma*I) / d theta_(D+1)
    dKdX = iKKstar'*covSEard(logtheta(1:(D+1)),X,D+1);
    
    % d mu / d theta_(D+1)
    dmus = (2*Kstar' - dKdX)*alpha;
    dMdX(y,D+1,y) = sum(dmus)/nsamp;
    
    % d Sigma / d theta_(D+1)
    dSdX(y,y,D+1,y)= 2*exp(2*logtheta(D+1)) - ...
        sum(sum(iKKstar .* (4 * Kstar - dKdX')),2)/nsamp;
    if nsamp > 1
        % Takes care of gradient from var(mu)
        dSdX(y,y,D+1,y) = dSdX(y,y,D+1,y) + ...
            cmus'*bsxfun(@minus,dmus,dMdX(y,D+1,y));
    end
    
    % Compute d M / d sn  and  d S / d sn
    % ------------------------------------
    % d (K+sigma*I) / d theta_(D+2)
    dKdX = iKKstar'*covNoise(logtheta(D+2),X,1);
    
    % d mu / d theta_(D+2)
    dmus = -dKdX*alpha;
    dMdX(y,D+2,y) = sum(dmus)/nsamp;
    
    % d Sigma / d theta_(D+2)
    dSdX(y,y,D+2,y) = sum(sum(iKKstar .* dKdX'),2)/nsamp;
    if nsamp > 1
        dSdX(y,y,D+2,y) = dSdX(y,y,D+2,y) + ...
            cmus'*bsxfun(@minus,dmus,dMdX(y,D+2,y));
    end    
end

%% Outputs
if nargout < 2
    dMdX = reshape(dMdX,E,(D+2)*E);
    dSdX = reshape(dSdX,E,E,(D+2)*E);

    % Create struct with all output fields
    out.M = M;
    out.S = S;
    out.dMdm = dMdm;
    out.dSdm = dSdm;
    out.dMds = dMds;
    out.dSds = dSds;
    out.dMdi = dMdi;
    out.dSdi = dSdi;
    out.dMdt = dMdt;
    out.dSdt = dSdt;
    out.dMdX = dMdX;
    out.dSdX = dSdX;
    
    M = out;
end