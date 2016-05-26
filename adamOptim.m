
% Adam optimisation (http://arxiv.org/pdf/1412.6980v8.pdf)
%
% Function for stochastic optimisation with the Adam algorithm:
%   theta = adamOptim(f, theta0, X, n, opt, varargin)
%
% = Inputs: =
%
%   func                Objective function. Has to be of the form
%                       [f, df] = func (theta, X, varargin)
% 
%   theta0              Initial guess for the parameter theta that
%                       adamOptim should optimise.
%
%   X                   Data set from which to sample. Should be N-by-D 
%                       matrix, where N is the number of data points, and
%                       D is the input dimension.
%
%   n                   Number of samples drawn at each iteration.
%                       Has to fulfill n <= N.
%
%   opt (optional)      Struct with optimisation parameters
%                       (see Adam paper for details):
%                       - alpha (defualt: 0.001)
%                       - beta1 (default: 0.9)
%                       - beta2 (default: 0.999)
%                       - epsilon (defualt: 1e-08)
%                       - maxIter (Number of iterations the optimisation
%                         algorithm should run; default: 100)
%
%   varargin (optional) Input arguments for objective function func.
%
% = Outputs: =
%
%   theta               Optimised parameter theta
%
%
% Copyright (C) 2016 by Simon Olofsson,
% 2016-04-11
%


function theta = adamOptim (func, theta0, X, n, opt, varargin)

if nargin < 5
    opt = 0;
end

if (n > size(X,1) || n < 1 || n ~= floor(n))
    error('adamOptim: Parameter n must be an integer in range [1,size(X,1)]');
end

%% Default values
% Parameter alpha
alpha = assignPar(opt,'alpha',0.001);
% Parameter beta1
beta1 = assignPar(opt,'beta1',0.9,0,1);
% Parameter beta2
beta2 = assignPar(opt,'beta2',0.999,0,1);
% Parameter epsilon
epsilon2 = assignPar(opt,'epsilon',1e-08,0,1)^2;
% Parameter maxIter
maxIter = assignPar(opt,'maxIter',100,1,Inf);


%% Initialisation

% initialise theta
if (~isnumeric(theta0) || sum(abs(imag(theta0(:)))) ~= 0)
    error('Value theta0 must be real-valued number(s)');
else
    theta = theta0;
end

gbar = zeros.*theta;
vbar = zeros.*theta;


%% Starting point

% Draw random samples
Xsample = X(randsample(1:size(X,1),n),:);

% Evaluate objective function
[funcVal,g] = func(theta,Xsample,varargin{:});
disp(['Initial value: ' num2str(funcVal)]);

%% Optimisation loop

beta1t = 1; beta2t = 1;
for t = 1 : min(maxIter)
    % Update
    gbar = beta1*gbar + (1-beta1)*g;
    vbar = beta2*vbar + (1-beta2)*g.^2;
    beta1t = beta1t * beta1; beta2t = beta2t * beta2;
    stepsize = alpha * sqrt(1-beta2t)/(1-beta1t);
    % Gradient descent step
    theta = theta - stepsize*gbar./(sqrt(vbar+epsilon2));
    
    % Draw random samples
    Xsample = X(randsample(1:size(X,1),n),:);

    % Evaluate objective function
    [funcVal,g] = func(theta,Xsample,varargin{:});
    % Print out
    fprintf(['\rFunc evaluation #' num2str(t) ': ' num2str(funcVal)]);
end
fprintf('\n');


end



function par = assignPar (opt, name, defaultValue, minVal, maxVal)

if ~ischar(name)
    error('adamOptim-assignPar: parameter name must be of string type.');
end

if (isfield(opt,name))
    usedefault = 0;
    par = opt.(name);
else
    usedefault = 1;
    par = defaultValue; 
end

%% Messages
msgIllegalDefault = ['adamOptim: illegal defualt value for ' ...
    'parameter ' name '.'];
msgIllegalValue = ['adamOptim: illegal value for parameter ' name '.'];
msgUseDefault = 'Default value used.';

%% Check that parameter value is acceptable
while (~isnumeric(par) || sum(abs(imag(par(:)))) ~= 0 || ...
        (nargin == 4 && ~(sum(par==minVal))) || ...
        (nargin == 5 && (par < minVal || par >= maxVal)))
    if (usedefault)
        error(msgIllegalDefault);
    else
        warning(msgIllegalValue);
        warning(msgUseDefault);
        par = defaultValue;
        usedefault = 1;
    end
end

end







