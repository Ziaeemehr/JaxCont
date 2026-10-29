function run_lorenz84_equilibria(output_dir)
addpath(fileparts(mfilename('fullpath')),fullfile(fileparts(mfilename('fullpath')),'systems'));
[x,s,opt]=lorenz84_seed;
keep=x(5,:)>=-1.5 & x(5,:)<=3;
lorenz84_native('lorenz84.eq-f',x(1:4,keep),[x(5,keep);.04*ones(1,sum(keep))],s,x,output_dir,opt);
end
