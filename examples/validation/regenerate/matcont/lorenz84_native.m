function lorenz84_native(run,states,parameters,s,raw,output_dir,opt)
f=lorenz84;n=size(states,2);r=zeros(n,1);spectra=zeros(n,4,2);
for k=1:n
 r(k)=norm(f{2}(0,states(:,k),parameters(1,k),parameters(2,k)),inf);
 ev=eig(f{3}(0,states(:,k),parameters(1,k),parameters(2,k)));spectra(k,:,1)=real(ev);spectra(k,:,2)=imag(ev);
end
events=struct('kind',{},'parameters',{},'state',{},'data',{});
for k=1:numel(s)
 label=strtrim(s(k).label);i=s(k).index;
 if ismember(label,{'00','99'}),continue;end
 F=raw(5,i);if size(raw,1)>=6,T=raw(6,i);else,T=.04;end
 events(end+1)=struct('kind',label,'parameters',struct('F',F,'T',T),'state',raw(1:4,i)','data',struct('native_label',label));
end
samples=struct('state',{},'parameters',{},'rhs',{},'jacobian',{});
for u={[0;0;0;0],[1.2;-.3;.7;-.2]}
 st=u{1};p=[1.7;.04];samples(end+1)=struct('state',st','parameters',p','rhs',f{2}(0,st,p(1),p(2))','jacobian',f{3}(0,st,p(1),p(2)));
end
out=struct('canonical_id',run,'states',states','parameters',parameters','residuals',r,'spectra',spectra,'events',{num2cell(events)},'samples',samples,'runtime_version',version,'matcont_version','7.6','resolved_controls',opt);
branch=struct([]); modes=struct([]); mapped=struct([]);segment=0;
for k=1:n
 % Preserve the native traversal; the reader splits at the active-coordinate turns.
 if strcmp(run,'lorenz84.eq-f'),parameter=parameters(1,k);else,parameter=parameters(2,k);end
 row=struct('segment_id',segment,'point_id',k-1,'order',k-1,'parameter',parameter,'F',parameters(1,k),'T',parameters(2,k),'residual_norm',r(k),'unstable_dimension',sum(spectra(k,:,1)>0),'status','CONVERGED');
 for j=1:4,row.(sprintf('state_%d',j-1))=states(j,k);end
 if isempty(branch),branch=row;else,branch(k)=row;end
 for j=1:4
  mode=struct('segment_id',segment,'point_id',k-1,'event_id','','kind','EIGENVALUE','mode_id',j-1,'real',spectra(k,j,1),'imag',spectra(k,j,2),'logabs',log(hypot(spectra(k,j,1),spectra(k,j,2))),'phase',atan2(spectra(k,j,2),spectra(k,j,1)),'trivial_mode_method','not_applicable');
  if isempty(modes),modes=mode;else,modes(end+1)=mode;end
 end
end
for k=1:numel(events)
 [~,idx]=min(sum((states-events(k).state').^2,1));b=branch(idx);
 if strcmp(run,'lorenz84.eq-f'),event_parameter=events(k).parameters.F;else,event_parameter=events(k).parameters.T;end
 e=struct('event_id',sprintf('event_%d',k-1),'event_type',events(k).kind,'kind',events(k).kind,'segment_id',b.segment_id,'point_id',b.point_id,'parameter',event_parameter,'F',events(k).parameters.F,'T',events(k).parameters.T,'bracket_lower',NaN,'bracket_upper',NaN,'bracket_status','unknown','status','REFINED');
 for j=1:4,e.(sprintf('state_%d',j-1))=events(k).state(j);end
 e.residual_norm=norm(f{2}(0,events(k).state',events(k).parameters.F,events(k).parameters.T),inf);
 ev=eig(f{3}(0,events(k).state',events(k).parameters.F,events(k).parameters.T));
 for j=1:4
  mode=struct('segment_id',b.segment_id,'point_id',b.point_id,'event_id',e.event_id,'kind','EIGENVALUE','mode_id',j-1,'real',real(ev(j)),'imag',imag(ev(j)),'logabs',log(abs(ev(j))),'phase',angle(ev(j)),'trivial_mode_method','not_applicable');
  modes(end+1)=mode;
 end
 e.frequency=0;
 if strcmp(events(k).kind,'H'),osc=ev(imag(ev)>0);[~,critical]=min(abs(real(osc)));e.frequency=imag(osc(critical));end
 e.fold_coefficient=NaN;e.first_lyapunov=NaN;
 original=find(arrayfun(@(a) strcmp(strtrim(a.label),events(k).kind)&&abs(raw(5,a.index)-events(k).parameters.F)<1e-8,s),1);
 % MatCont 7.6 sout does not expose localization brackets. Preserve an
 % explicitly supplied native scalar-coordinate bracket if a future version does.
 if ~isempty(original)&&isfield(s(original).data,'bracket')
  bracket=s(original).data.bracket;
  if isnumeric(bracket)&&isreal(bracket)&&numel(bracket)==2&&all(isfinite(bracket(:)))
   e.bracket_lower=min(bracket(:));e.bracket_upper=max(bracket(:));e.bracket_status='native';
  end
 end
 if ~isempty(original)&&isfield(s(original).data,'a'),e.fold_coefficient=s(original).data.a;end
 if ~isempty(original)&&isfield(s(original).data,'lyapunov'),e.first_lyapunov=s(original).data.lyapunov;end
 e.coefficient_convention='MatCont 7.6 native normal form, original time, tool-native eigenvector normalization';
 if isempty(mapped),mapped=e;else,mapped(k)=e;end
end
[~,os_name]=system('uname -sr');cpu_name=computer;
if isfile('/proc/cpuinfo')
 cpu_info=fileread('/proc/cpuinfo');cpu_match=regexp(cpu_info,'model name\s*:\s*([^\n]+)','tokens','once');
 if ~isempty(cpu_match),cpu_name=strtrim(cpu_match{1});end
end
platform=struct('os',strtrim(os_name),'cpu',cpu_name,'blas',version('-blas'),'lapack',version('-lapack'),'architecture',computer('arch'),'source','MATLAB native runtime');
native=struct('canonical_id',run,'branch',branch,'events',{num2cell(mapped)},'spectra',modes,'metadata',struct('platform',platform,'precision','float64','scaling',struct('time','original','metric','unweighted Euclidean state and active parameters'),'seeds',struct('F',3,'T',.04,'correction','Newton before continuation'),'mesh',struct('kind','equilibrium'),'chart_status','MatCont returned chart; inspect native events and logs','runtime_version',version,'package_versions',struct('MatCont','7.6'),'parameter_order',{{'F','T'}},'resolved_controls',opt,'samples',samples,'solver_strategies',struct('method','MatCont pseudo-arclength','event_selection','type and frozen parameter window')));
folder=fullfile(output_dir,run,'matcont');if ~isfolder(folder),mkdir(folder);end
fidraw=fopen(fullfile(folder,'raw.json'),'w');fprintf(fidraw,'%s\n',jsonencode(native));fclose(fidraw);
fid=fopen(fullfile(folder,'native.json'),'w');closer=onCleanup(@() fclose(fid));fprintf(fid,'%s\n',jsonencode(out));
end
