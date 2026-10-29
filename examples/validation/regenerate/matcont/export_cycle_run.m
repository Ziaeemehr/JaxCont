function native = export_cycle_run(case_id, x, s, processor_data, output_dir, metadata)
%EXPORT_CYCLE_RUN Write normalized periodic branch/event/multiplier artifacts.

global lds
if isempty(x)
    error('JaxContValidation:EmptyCycleRun', ...
        'MatCont returned no cycle points for %s. Check the continuation log and native cycle MEX installation.', ...
        case_id);
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end
npoint = size(x, 2);
nphase = lds.nphase;
ncoords = size(x, 1) - 2;
parameter = x(end, :)';
period = x(end-1, :)';
residual = zeros(npoint, 1);
state_min = zeros(npoint, nphase);
state_max = zeros(npoint, nphase);
curve = feval(@limitcycle);
saved_mesh = lds.msh;
saved_dt = lds.dt;
for k = 1:npoint
    % The final BVP equation is MatCont's moving integral phase condition;
    % its reference orbit is updated during continuation and only represents
    % the latest point after ``cont`` returns.  The collocation defects and
    % continuity equations remain portable for every stored branch point.
    if size(processor_data, 1) >= lds.ntst + 1
        lds.msh = real(processor_data(1:lds.ntst+1, k))';
        lds.dt = lds.msh(lds.tsts+1) - lds.msh(lds.tsts);
    end
    value = feval(curve{1}, x(:, k));
    residual(k) = norm(value(1:end-1), inf);
    orbit = reshape(x(1:ncoords, k), nphase, []);
    state_min(k, :) = min(orbit, [], 2)';
    state_max(k, :) = max(orbit, [], 2)';
end
lds.msh = saved_mesh;
lds.dt = saved_dt;

point = (0:npoint-1)';
branch = table(repmat({case_id}, npoint, 1), point, parameter, period, residual, ...
    'VariableNames', {'case_id', 'point', 'parameter', 'period', 'residual_norm'});
for j = 1:nphase
    branch.(sprintf('state_%d_min', j - 1)) = state_min(:, j);
    branch.(sprintf('state_%d_max', j - 1)) = state_max(:, j);
end
writetable(branch, fullfile(output_dir, [case_id '_branch.csv']));

labels = cellfun(@strtrim, {s.label}, 'UniformOutput', false);
keep = ~ismember(labels, {'00', '99'});
event_struct = s(keep);
labels = labels(keep);
nevent = numel(event_struct);
event_index = (0:nevent-1)';
event_type = labels(:);
event_point = zeros(nevent, 1);
event_parameter = zeros(nevent, 1);
event_period = zeros(nevent, 1);
multiplier_case = {};
multiplier_point = [];
multiplier_event = [];
multiplier_type = {};
multiplier_kind = {};
multiplier_index = [];
multiplier_real = [];
multiplier_imag = [];
if size(processor_data, 1) >= lds.ntst + 1 + nphase
    for k = 1:npoint
        values = processor_data(lds.ntst+2:lds.ntst+1+nphase, k);
        for j = 1:numel(values)
            multiplier_case{end+1, 1} = case_id; %#ok<AGROW>
            multiplier_point(end+1, 1) = k - 1; %#ok<AGROW>
            multiplier_event(end+1, 1) = -1; %#ok<AGROW>
            multiplier_type{end+1, 1} = 'BRANCH'; %#ok<AGROW>
            multiplier_kind{end+1, 1} = 'FLOQUET'; %#ok<AGROW>
            multiplier_index(end+1, 1) = j - 1; %#ok<AGROW>
            multiplier_real(end+1, 1) = real(values(j)); %#ok<AGROW>
            multiplier_imag(end+1, 1) = imag(values(j)); %#ok<AGROW>
        end
    end
end
for k = 1:nevent
    idx = event_struct(k).index;
    event_point(k) = idx - 1;
    event_parameter(k) = parameter(idx);
    event_period(k) = period(idx);
    if isfield(event_struct(k), 'data') && ...
            isfield(event_struct(k).data, 'multipliers')
        values = event_struct(k).data.multipliers(:);
        for j = 1:numel(values)
            multiplier_case{end+1, 1} = case_id; %#ok<AGROW>
            multiplier_point(end+1, 1) = idx - 1; %#ok<AGROW>
            multiplier_event(end+1, 1) = k - 1; %#ok<AGROW>
            multiplier_type{end+1, 1} = labels{k}; %#ok<AGROW>
            multiplier_kind{end+1, 1} = 'FLOQUET'; %#ok<AGROW>
            multiplier_index(end+1, 1) = j - 1; %#ok<AGROW>
            multiplier_real(end+1, 1) = real(values(j)); %#ok<AGROW>
            multiplier_imag(end+1, 1) = imag(values(j)); %#ok<AGROW>
        end
    end
end
events = table(repmat({case_id}, nevent, 1), event_index, event_type, ...
    event_point, event_parameter, event_period, 'VariableNames', ...
    {'case_id', 'event_index', 'event_type', 'point', 'parameter', 'period'});
writetable(events, fullfile(output_dir, [case_id '_events.csv']));
multipliers = table(multiplier_case, multiplier_point, multiplier_event, multiplier_type, ...
    multiplier_kind, multiplier_index, multiplier_real, multiplier_imag, ...
    'VariableNames', {'case_id', 'point', 'event_index', 'event_type', ...
    'spectrum_kind', 'multiplier_index', 'real', 'imag'});
writetable(multipliers, fullfile(output_dir, [case_id '_multipliers.csv']));

metadata.case_id = case_id;
metadata.matlab_version = version;
metadata.matcont_version = '7.6';
metadata.mesh = struct('ntst', lds.ntst, 'ncol', lds.ncol);
metadata.generated_utc = datestr(datetime('now', 'TimeZone', 'UTC'), ...
    'yyyy-mm-ddTHH:MM:SSZ');
fid = fopen(fullfile(output_dir, [case_id '_metadata.json']), 'w');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(metadata));
if nargout > 0
    native = cycle_native(branch, events, multipliers, x, processor_data, metadata);
end
end

function native = cycle_native(branch, events, multipliers, x, data, metadata)
% Native periodic supplement: original ODE integration and physical flow mode.
% Only the adaptx driver requests this output. Historical CSV bytes/columns
% are preserved for legacy consumers of the zero-output exporter.
global lds
rhs = @(u,a) [u(2);u(3);-a*u(3)-u(2)-u(1)+u(1)^2];
jac = @(u,a) [0,1,0;0,0,1;-1+2*u(1),-1,-a];
segment = metadata.segment;
B = {}; E = {}; S = {}; O = {};
for k=1:height(branch)
    orbit=reshape(x(1:end-2,k),3,[]); a=x(end,k); T=x(end-1,k);
    msh=real(data(1:lds.ntst+1,k))';
    phase=[];
    for j=1:lds.ntst
        phase=[phase, msh(j)+(msh(j+1)-msh(j))*(0:lds.ncol-1)/lds.ncol]; %#ok<AGROW>
    end
    phase=[phase,1];
    % Integrate the physical variational ODE independently of collocation.
    y0=[orbit(:,1);reshape(eye(3),9,1)];
    vf=@(t,y) [rhs(y(1:3),a);reshape(jac(y(1:3),a)*reshape(y(4:end),3,3),9,1)];
    [~,Y]=ode113(vf,T*phase,y0,odeset('RelTol',1e-11,'AbsTol',1e-13));
    [V,D]=eig(reshape(Y(end,4:end),3,3)); flow=rhs(orbit(:,1),a);
    align=abs(V'*flow)./(vecnorm(V)'*norm(flow)); [~,triv]=max(align);
    values=complex(multipliers.real(multipliers.point==k-1 & multipliers.event_index==-1), multipliers.imag(multipliers.point==k-1 & multipliers.event_index==-1));
    assert(numel(values)==3,'Missing native MatCont branch multipliers');
    permutations=perms(1:3); costs=zeros(size(permutations,1),1);
    for q=1:size(permutations,1),costs(q)=norm(values-diag(D(permutations(q,:),permutations(q,:))));end
    [~,best]=min(costs); assignment=permutations(best,:);native_triv=find(assignment==triv);
    row=struct('segment_id',segment,'point_id',sprintf('%d',k-1),'order',k-1,'parameter',a,'period',T,'residual_norm',branch.residual_norm(k),'unstable_dimension',sum(abs(values)>1+1e-7),'status','CONVERGED', ...
        'independent_shooting_defect',norm(Y(end,1:3)'-orbit(:,1),inf),'closure',norm(orbit(:,end)-orbit(:,1),inf));
    for j=1:3
        row.(sprintf('state_%d_min',j-1))=min(orbit(j,:)); row.(sprintf('state_%d_max',j-1))=max(orbit(j,:));
    end
    centered=orbit(:,1:end-1)-mean(orbit(:,1:end-1),2);
    row.half_period_difference=max(vecnorm(centered-circshift(centered,size(centered,2)/2,2)))/max(vecnorm(centered));
    B{end+1}=row; %#ok<AGROW>
    ids=find(events.point==k-1);
    for q=ids'
        e=row; e.event_id=sprintf('%s-%d',segment,q-1); e.event_type=events.event_type{q};e.bracket_lower=NaN;e.bracket_upper=NaN;e.bracket_status='unknown';
        E{end+1}=e; %#ok<AGROW>
    end
    for j=1:3
        z=values(j); method='independent variational eigenvector flow alignment';
        mode=struct('segment_id',segment,'point_id',row.point_id,'event_id','','kind','FLOQUET_MULTIPLIER','mode_id',j-1,'real',real(z),'imag',imag(z),'logabs',log(max(abs(z),realmin)),'phase',angle(z),'trivial_mode_method',method,'is_trivial',j==native_triv,'flow_alignment',align(assignment(j)));
        S{end+1}=mode; %#ok<AGROW>
        for q=ids'
            evmode=mode; evmode.event_id=sprintf('%s-%d',segment,q-1); S{end+1}=evmode; %#ok<AGROW>
        end
    end
    for j=1:numel(phase)
        O{end+1}=struct('segment_id',segment,'point_id',row.point_id,'phase_fraction',phase(j),'time',T*phase(j),'state_0',orbit(1,j),'state_1',orbit(2,j),'state_2',orbit(3,j)); %#ok<AGROW>
    end
end
native=struct('branch',{B},'events',{E},'spectra',{S},'orbits',{O});
end
