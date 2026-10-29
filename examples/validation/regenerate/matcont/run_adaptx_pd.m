function run_adaptx_pd(output_dir)
%RUN_ADAPTX_PD Independent Hopf -> base PD -> doubled cycle candidate.
script_dir=fileparts(mfilename('fullpath'));addpath(script_dir);setup_matcont();
contract=jsondecode(fileread(fullfile(script_dir,'../../cases/systems/periodic.json'))); c=contract.adaptx;
% Window justification and actual pre-producer computations: periodic.json.
folder=fullfile(output_dir,'adaptx.cycle-pd-switch','matcont');if ~isfolder(folder),mkdir(folder);end
clear global cds eds lds
opt=contset;opt=contset(opt,'Singularities',1);opt=contset(opt,'Multipliers',1);
opt=contset(opt,'Adapt',0);opt=contset(opt,'MaxNumPoints',c.matcont_max_steps);
opt=contset(opt,'InitStepsize',c.ds);opt=contset(opt,'MinStepsize',c.ds_min);opt=contset(opt,'MaxStepsize',c.ds_max);
opt=contset(opt,'FunTolerance',1e-9);opt=contset(opt,'VarTolerance',1e-9);opt=contset(opt,'TestTolerance',1e-7);opt=contset(opt,'MaxCorrIters',25);
% Equilibrium Hopf is independently found rather than supplied as a scalar.
[xe,~]=init_EP_EP(@adaptx,[0;0;0],[1.1;1],1);
eo=contset(opt,'Backward',1);eo=contset(eo,'MaxNumPoints',40);
[xe,~,se]=cont(@equilibrium,xe,[],eo);
hi=select(se,xe,'H',c.hopf_window); hp=xe(end,se(hi).index);
[x0,v0]=init_H_LC(@adaptx,xe(1:3,se(hi).index),[hp;1],1,1e-4,c.ntst,c.ncol);
[x,v,s,~,f]=cont(@limitcycle,x0,v0,opt);
[x,s,f]=through_domain(x,s,f,c.base_domain(1));
pd=select(s,x,'PD',c.pd_window); pd_alpha=x(end,s(pd).index);
% Retain the entire computed base branch. Domain checks below prohibit a
% candidate whose step budget stops before the frozen comparison window.
meta=struct('segment','base','solver_settings',opt,'precision','float64');
base=export_cycle_run('adaptx-base',x,s,f,folder,meta);
[x0,v0]=init_PD_LC(@adaptx,x,s(pd),2*c.ntst,c.ncol,1e-6);
[x2,~,s2,~,f2]=cont(@limitcycle,x0,v0,opt);
[x2,s2,f2]=through_domain(x2,s2,f2,c.doubled_domain(1));
dpi=select(s2,x2,'PD',c.doubled_pd_window);
meta.segment='doubled';doubled=export_cycle_run('adaptx-doubled',x2,s2,f2,folder,meta);
assert(min(x(end,:))<=c.base_domain(1) && max(x(end,:))>=c.base_domain(2),'Incomplete base domain');
assert(min(x2(end,:))<=c.doubled_domain(1) && max(x2(end,:))>=c.doubled_domain(2),'Incomplete doubled domain');
metadata=struct('runtime_version',version,'package_versions',struct('MatCont','7.6'),'precision','float64','parameter_order',{{'alpha','beta'}},'scaling',struct('time','original','metric','MatCont Euclidean collocation coordinates'),'seeds',struct('hopf_alpha',hp,'state',[0,0,0],'beta',1),'mesh',struct('ntst',c.ntst,'ncol',c.ncol,'doubled_ntst',2*c.ntst),'resolved_controls',opt,'solver_strategies',struct('method','MatCont pseudo-arclength','event_selection','type and computed window'),'comparable',false,'noncomparability_reasons',{{'MatCont collocation coordinate norm differs from JaxCont quadrature layout'}},'period_ratio',x2(end-1,1)/x(end-1,s(pd).index),'pd_alpha',pd_alpha,'doubled_pd_alpha',x2(end,s2(dpi).index));
[~,os_name]=system('uname -sr');
metadata.hopf_alpha=hp;metadata.hopf_period=2*pi/max(abs(imag(eig([0,1,0;0,0,1;-1,-1,-hp]))));
metadata.platform=struct('os',strtrim(os_name),'cpu',computer,'blas',version('-blas'),'source','MATLAB native runtime');
native=struct('canonical_id','adaptx.cycle-pd-switch','branch',{[base.branch,doubled.branch]},'events',{[base.events,doubled.events]},'spectra',{[base.spectra,doubled.spectra]},'orbits',{[base.orbits,doubled.orbits]},'metadata',metadata);
fid=fopen(fullfile(folder,'raw.json'),'w');clean=onCleanup(@() fclose(fid));fprintf(fid,'%s\n',jsonencode(native));
end
function index=select(events,x,kind,window)
labels=cellfun(@strtrim,{events.label},'UniformOutput',false);
indices=find(strcmp(labels,kind));indices=indices(arrayfun(@(i) x(end,events(i).index)>=window(1)&&x(end,events(i).index)<=window(2),indices));
assert(numel(indices)==1,'Expected one %s in frozen window',kind);index=indices(1);
end

function [x,s,f]=through_domain(x,s,f,bound)
% Keep the full first traversal through the lower contract boundary,
% including the first point beyond it; no expected event is removed.
last=find(x(end,:)<=bound,1);assert(~isempty(last),'Step budget exhausted before full domain');
x=x(:,1:last);f=f(:,1:last);s=s([s.index]<=last);
end
