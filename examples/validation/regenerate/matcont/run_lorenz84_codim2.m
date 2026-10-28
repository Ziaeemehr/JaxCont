function run_lorenz84_codim2(output_dir)
addpath(fileparts(mfilename('fullpath')),fullfile(fileparts(mfilename('fullpath')),'systems'));
[x,s,opt]=lorenz84_seed;
for kind={'LP','H'}
 label=kind{1}; if strcmp(label,'LP'),win=[1.50 1.58];run='lorenz84.fold-ft';else,win=[1.59 1.65];run='lorenz84.hopf-ft';end
 matches=find(arrayfun(@(e) strcmp(strtrim(e.label),label)&&x(5,e.index)>=win(1)&&x(5,e.index)<=win(2),s));
 assert(numel(matches)==1,'Expected exactly one event of selected type/window');
 e=s(matches);p=[x(5,e.index);.04];u=x(1:4,e.index);
 if strcmp(label,'LP'),[seed,v]=init_LP_LP(@lorenz84,u,p,[1 2]);curve=@limitpoint;else,[seed,v]=init_H_H(@lorenz84,u,p,[1 2]);curve=@hopf;end
 if strcmp(label,'LP'),backward=1;else,backward=0;end
 opt=contset(opt,'Backward',backward);opt=contset(opt,'InitStepsize',.001);opt=contset(opt,'MinStepsize',1e-5);opt=contset(opt,'MaxStepsize',.015);opt=contset(opt,'MaxNumPoints',31);
 [c,~,special]=cont(curve,seed,v,opt);
 if strcmp(label,'H')
  % MatCont's squared-frequency Hopf chart can continue into a neutral saddle.
  % Retain the true Hopf portion, ending at its native BT boundary.
  % The chart boundary uses the frozen positive-T window; it is not
  % standalone BT selection (whose F window is checked separately below).
  boundary=find(arrayfun(@(e) strcmp(strtrim(e.label),'BT')&&c(6,e.index)>=.01&&c(6,e.index)<=.03,special));
  assert(numel(boundary)<=1,'Ambiguous positive-T Hopf chart boundary');
  if ~isempty(boundary)
   stop=special(boundary).index;c=c(:,1:stop);
   special=special([special.index]<=stop);
  end
 end
 lorenz84_native(run,c(1:4,:),c(5:6,:),special,c,output_dir,opt);
 if strcmp(label,'LP')
  bt=find(arrayfun(@(e) strcmp(strtrim(e.label),'BT')&&c(5,e.index)>=1.40&&c(5,e.index)<=1.50&&c(6,e.index)>=.01&&c(6,e.index)<=.03,special));
  if numel(bt)~=1
   folder=fullfile(output_dir,'lorenz84.bt-ft','matcont');if ~isfolder(folder),mkdir(folder);end
   rejected=find(arrayfun(@(e) strcmp(strtrim(e.label),'BT'),special));
   values=arrayfun(@(e) c(5:6,e.index)',special(rejected),'UniformOutput',false);
   failure=struct('status','FAIL','reason','Refined BT lies outside immutable positive-T selection window F=[1.40,1.50], T=[.01,.03]','attempted_parameters',{values});
   fid=fopen(fullfile(folder,'failure.json'),'w');fprintf(fid,'%s\n',jsonencode(failure));fclose(fid);
   continue
  end
  idx=special(bt).index;
  selected=special(bt);selected.index=1;
  stale=fullfile(output_dir,'lorenz84.bt-ft','matcont','failure.json');
  if isfile(stale),delete(stale);end % a success must not keep an old failure
  lorenz84_native('lorenz84.bt-ft',c(1:4,idx),c(5:6,idx),selected,c(:,idx),output_dir,opt);
 end
end
end
