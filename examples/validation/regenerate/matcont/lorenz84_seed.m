function [x,s,opt] = lorenz84_seed
% Frozen F-sweep; discover event identities without solution imports.
setup_matcont(); clear global cds eds lds
p=[3;.04]; u=[2.9787004394953343;-.03868302503393752;.058232737694740085;-.02105288273117459];
f=lorenz84; for k=1:5, u=u-f{3}(0,u,p(1),p(2))\f{2}(0,u,p(1),p(2)); end
[x0,v0]=init_EP_EP(@lorenz84,u,p,1);
opt=contset; opt=contset(opt,'Singularities',1);opt=contset(opt,'Eigenvalues',1);
opt=contset(opt,'MaxNumPoints',201);opt=contset(opt,'InitStepsize',.002);opt=contset(opt,'MaxStepsize',.05);
opt=contset(opt,'FunTolerance',1e-12);opt=contset(opt,'VarTolerance',1e-12);opt=contset(opt,'TestTolerance',1e-10);
opt=contset(opt,'Backward',0);cont(@equilibrium,x0,v0,opt);
% The forward branch starts at the upper F boundary; all later points are outside.
opt=contset(opt,'Backward',1);[x,~,s]=cont(@equilibrium,x0,v0,opt);
end
