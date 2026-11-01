"""Independent periodic diagnostics; no continuation residual is reused."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import BarycentricInterpolator


def doubled_shape(states):
    """Dimensionless half-period difference after removing the orbit mean.

    ``states`` sample one period on a uniform phase grid. The half-period
    shift is the trigonometric one (a phase factor per Fourier mode): on an
    even grid it equals rolling by half the samples, and on an odd grid
    (e.g. ``ntst=25``), which has no sample at half the period, it is the
    spectrally accurate interpolant of the smooth orbit.
    """
    states = np.asarray(states, dtype=float)
    centered = states - states.mean(axis=0)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return 0.0
    modes = np.fft.fftfreq(len(states), 1 / len(states))
    shifted = np.real(np.fft.ifft(np.fft.fft(centered, axis=0) * np.exp(-1j * np.pi * modes)[:, None], axis=0))
    return float(np.max(np.linalg.norm(centered - shifted, axis=1)) / scale)


def require_events(events, expected):
    kinds = [row.get('event_type', row.get('kind')) for row in events]
    for kind in expected:
        if kinds.count(kind) != 1:
            raise ValueError(f'Expected exactly one {kind}, found {kinds.count(kind)}')


def independent_defect(rhs, mesh, state, parameter, args=None):
    """Integrate each mesh interval independently and compare its polynomial.

    DOP853 integrates the original ODE in float64. Dense polynomial samples
    include off-collocation points, so this measures interpolation/integration
    error even when the Newton collocation equations vanish.
    """
    import jax
    from jaxcont.core.collocation import gauss_legendre_01
    state = np.asarray(state, dtype=float)
    n = (len(state)-1)//(mesh.ntst*(mesh.ncol+1))
    left = state[:mesh.ntst*n].reshape(mesh.ntst,n)
    interior = state[mesh.ntst*n:-1].reshape(mesh.ntst,mesh.ncol,n)
    nodes = np.r_[0., gauss_legendre_01(mesh.ncol)[0]]
    period = float(state[-1])
    if period <= 0:
        raise ValueError('Period must be positive')
    h = period / mesh.ntst
    defect = closure = 0.
    def vf(t, u):
        with jax.enable_x64():
            return np.asarray(rhs(u, parameter, args), dtype=float)
    for i in range(mesh.ntst):
        sol = solve_ivp(vf, (0,h),left[i],method='DOP853',rtol=1e-11,atol=1e-13,dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        query = np.linspace(0,1,17)
        polynomial = BarycentricInterpolator(nodes,np.vstack([left[i],interior[i]]))(query)
        defect = max(defect,float(np.max(np.abs(sol.sol(h*query).T-polynomial))))
        closure = max(closure,float(np.max(np.abs(sol.y[:,-1]-left[(i+1)%mesh.ntst]))))
    return dict(independent_dense_defect=defect, independent_interval_closure=closure)


def periodic_records(result, mesh, rhs, *, segment='cycle', allow_failed=False):
    """Fresh v2 tables, retaining continuation order and missing native labels."""
    import jax
    import jax.numpy as jnp
    from jaxcont.stability.floquet import floquet_spectrum
    from jaxcont.problems.periodic import periodic_residual
    from jaxcont.problems.periodic_args import PeriodicArgs
    nvalid = int(result.branch.n_valid)
    states = result.branch.states[:nvalid]
    params = result.branch.params[:nvalid]
    # Reuse the public Branch's already computed Floquet data, including
    # physical flow identity and log magnitudes. Do not recompute a second
    # batched spectrum with different shapes or lose tiny multipliers.
    values_all = result.branch.eigenvalues[:nvalid]
    trivial_all = result.branch.trivial_index[:nvalid]
    logs_all = result.branch.floquet_logabs[:nvalid]
    rows, modes, orbits, events = [], [], [], []
    n = (states.shape[1]-1)//(mesh.ntst*(mesh.ncol+1))
    for i in range(nvalid):
        u = np.asarray(states[i]); p = float(params[i]); T = float(u[-1])
        left = u[:mesh.ntst*n].reshape(mesh.ntst,n)
        all_states = u[:-1].reshape(-1,n)
        values = np.asarray(values_all[i]); trivial = int(trivial_all[i])
        if trivial < 0 and not allow_failed:
            raise RuntimeError("Native Floquet flow mode did not converge")
        row = dict(segment_id=segment,point_id=str(i),order=i,parameter=p,period=T,
                   residual_norm=None,unstable_dimension=(int(np.count_nonzero(np.delete(np.abs(values),trivial)>1+1e-7)) if trivial>=0 else None),status=('CONVERGED' if trivial>=0 else 'FLOQUET_FAILED'))
        for j in range(n):
            row[f'state_{j}_min'] = float(all_states[:,j].min())
            row[f'state_{j}_max'] = float(all_states[:,j].max())
        coll = jnp.asarray(u[mesh.ntst*n:-1].reshape(mesh.ntst,mesh.ncol,n))
        args = PeriodicArgs(coll,jnp.zeros_like(coll),None,rhs,mesh)
        row['residual_norm'] = float(jnp.max(jnp.abs(periodic_residual(states[i],params[i],args)[:-1])))
        row['half_period_difference'] = doubled_shape(left)
        if i in (0,nvalid-1):
            row.update(independent_defect(rhs,mesh,u,p))
        rows.append(row)
        for j,z in enumerate(values):
            modes.append(dict(segment_id=segment,point_id=str(i),event_id='',kind='FLOQUET_MULTIPLIER',mode_id=str(j),real=float(z.real),imag=float(z.imag),logabs=float(logs_all[i,j]),phase=float(np.angle(z)),trivial_mode_method=('physical flow alignment' if trivial>=0 else 'unresolved native Floquet flow mode'),is_trivial=int(j==trivial)))
        for k,x in enumerate(left):
            orbits.append(dict(segment_id=segment,point_id=str(i),phase_fraction=k/mesh.ntst,time=T*k/mesh.ntst,**{f'state_{j}':float(v) for j,v in enumerate(x)}))
    labels = {'fold':'LPC','period_doubling':'PD','neimark_sacker':'NS'}
    for i,e in enumerate(result.events):
        if e.kind not in labels:
            continue
        nearest = int(np.argmin(np.linalg.norm(np.asarray(states)-np.asarray(e.u),axis=1)))
        row = dict(rows[nearest])
        row.update(point_id=f'event-{i}',order=float(np.mean(e.index)) if isinstance(e.index,tuple) else nearest+.5,event_id=f'{segment}-{i}',event_type=labels[e.kind],kind=labels[e.kind],parameter=float(e.p),period=float(e.u[-1]),status='CONVERGED',refinement_status='REFINED',bracket_lower=None,bracket_upper=None,bracket_status='unknown')
        eu=np.asarray(e.u);left=eu[:mesh.ntst*n].reshape(mesh.ntst,n);all_states=eu[:-1].reshape(-1,n)
        for j in range(n):
            row[f'state_{j}_min']=float(all_states[:,j].min());row[f'state_{j}_max']=float(all_states[:,j].max())
        coll=jnp.asarray(eu[mesh.ntst*n:-1].reshape(mesh.ntst,mesh.ncol,n))
        args=PeriodicArgs(coll,jnp.zeros_like(coll),None,rhs,mesh)
        row['residual_norm']=float(jnp.max(jnp.abs(periodic_residual(e.u,e.p,args)[:-1])))
        try:
            exact = floquet_spectrum(rhs,mesh,e.u,e.p)
        except RuntimeError as exc:
            if not allow_failed:
                raise
            exact = None
            row['status']='FLOQUET_FAILED'
            row['unstable_dimension']=None
            row['spectrum_error']=str(exc)
        if exact is not None:
            ev=np.asarray(exact.values);triv=int(exact.trivial_index)
            row['unstable_dimension']=int(np.count_nonzero(np.delete(np.abs(ev),triv)>1+1e-7))
        row['half_period_difference']=doubled_shape(left)
        for j in range(n):
            z = complex(np.asarray(exact.values)[j]) if exact is not None else complex(float('nan'),float('nan'))
            modes.append(dict(segment_id=segment,point_id=row['point_id'],event_id=row['event_id'],kind='FLOQUET_MULTIPLIER',mode_id=str(j),real=float(z.real),imag=float(z.imag),logabs=float(exact.logabs[j]) if exact is not None else float('nan'),phase=float(np.angle(z)),trivial_mode_method='physical flow alignment' if exact is not None else 'unresolved native Floquet flow mode',is_trivial=int(exact is not None and j==int(exact.trivial_index))))
        row.update(independent_defect(rhs,mesh,e.u,e.p))
        rows.append(row)
        for k,x in enumerate(left):
            orbits.append(dict(segment_id=segment,point_id=row['point_id'],phase_fraction=k/mesh.ntst,time=float(e.u[-1])*k/mesh.ntst,**{f'state_{j}':float(v) for j,v in enumerate(x)}))
        events.append(row)
    rows.sort(key=lambda r:r['order'])
    return dict(branch=rows,events=events,spectra=modes,orbits=orbits)


def mesh_refinement(rhs, mesh, state, parameter):
    """Re-solve on twice as many intervals, then repeat independent integration."""
    import jax.numpy as jnp
    from jaxcont.core.collocation import Collocation
    from jaxcont.problems.periodic import periodic_orbit_problem
    n=(len(state)-1)//(mesh.ntst*(mesh.ncol+1))
    T=float(state[-1])
    left=np.asarray(state[:mesh.ntst*n]).reshape(mesh.ntst,n)
    fine=Collocation(ntst=2*mesh.ntst,ncol=mesh.ncol)
    # Mesh-point storage excludes phase one. Supply the known periodic
    # endpoint explicitly so even coarse grids cover the full period.
    closed=np.vstack([left,left[0]])
    corrected=periodic_orbit_problem(rhs,jnp.asarray(closed),jnp.linspace(0,T,mesh.ntst+1),T,parameter,fine)
    return dict(coarse_ntst=mesh.ntst,fine_ntst=fine.ntst,
                coarse=independent_defect(rhs,mesh,state,parameter),
                fine=independent_defect(rhs,fine,corrected.u0,parameter),
                period_difference=abs(float(corrected.u0[-1])-T))
