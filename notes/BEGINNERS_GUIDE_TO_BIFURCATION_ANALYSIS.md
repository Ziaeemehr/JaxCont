# A Beginner's Guide to Bifurcation Calculation

> A theory-first path into numerical continuation and bifurcation analysis, written for future users and contributors to JaxCont.

## What this tutorial will teach you

A bifurcation is a qualitative change in a system's long-term behavior when a parameter changes. A stable rest state may disappear, two equilibria may exchange stability, or steady behavior may give way to oscillation. **Bifurcation calculation** is the numerical work of tracing solution families, determining their stability, detecting candidate changes, and accurately locating and classifying the special points.

By the end, you should be able to:

- read a one-parameter bifurcation diagram;
- turn an ordinary differential equation (ODE) into an equilibrium equation;
- determine local equilibrium stability from Jacobian eigenvalues;
- recognize fold, transcritical, pitchfork, and Hopf bifurcations;
- explain why repeated simulation or a simple parameter sweep can miss important solutions;
- derive the predictor and corrector equations used by pseudo-arclength continuation;
- understand how software detects and refines candidate bifurcations;
- diagnose the most common numerical failures; and
- follow a practical learning path before using a continuation package.

This guide concentrates on **equilibria of smooth continuous-time ODEs**, because that is the best entry point and the current mature target of JaxCont. Periodic orbits, maps, delay equations, PDEs, and two-parameter continuation use the same philosophy but need additional machinery.

## 1. The big picture

Suppose a model is

$$
\dot u = f(u,p), \qquad u\in\mathbb R^n,\quad p\in\mathbb R,
$$

where $u$ is the state and $p$ is a parameter. An equilibrium $u^*$ is a state that does not move:

$$
f(u^*,p)=0.
$$

For one chosen value of $p$, this is a nonlinear root-finding problem. Bifurcation analysis asks a larger question:

> How do **all relevant roots** and their stability change as $p$ varies?

Numerically, that becomes a five-stage loop:

```mermaid
flowchart LR
    A[Find one known solution] --> B[Predict a nearby solution]
    B --> C[Correct it by solving nonlinear equations]
    C --> D[Compute stability indicators]
    D --> E[Detect and refine special points]
    E --> B
```

The repeated prediction and correction is called **numerical continuation** or **path following**. The result is a branch of solutions $(u(s),p(s))$, usually drawn as a bifurcation diagram. The parameter $s$ measures progress along the branch; it need not be the physical parameter $p$.

## 2. What to know first

You do not need graduate-level analysis to begin. You do need working knowledge of:

1. **Calculus:** derivatives, partial derivatives, Taylor expansion, and the chain rule.
2. **Linear algebra:** matrices, linear systems, eigenvalues/eigenvectors, rank, null spaces, and norms.
3. **ODEs:** what a trajectory, equilibrium, and initial condition mean.
4. **Numerical methods:** Newton's method and the idea of a tolerance.
5. **Basic Python/NumPy:** useful for experiments, but not necessary for the mathematical sections.

Learn these topics in that order. Do not wait until every theorem feels complete; calculate the one-dimensional examples while learning the prerequisites.

## 3. From trajectories to equilibria

Simulation answers:

> Starting from this initial condition at this parameter, where does the trajectory go?

Continuation answers:

> What invariant solutions exist across a range of parameters, including unstable ones that simulation will normally avoid?

That difference is central. If an equilibrium is unstable, nearly every simulated trajectory leaves it. A root solver can still find it, and continuation can still follow it. Unstable branches often organize the boundaries between observable behaviors, so hiding them gives an incomplete picture.

### 3.1 A first one-dimensional system

Consider

$$
\dot x = p-x^2.
$$

Equilibria solve

$$
F(x,p)=p-x^2=0,
$$

so $x^*=\pm\sqrt p$ for $p>0$, one double root exists at $(x,p)=(0,0)$, and no real equilibrium exists for $p<0$.

Local stability in one dimension follows from $F_x=\partial F/\partial x$:

- $F_x(x^*,p)<0$: nearby perturbations decay, so the equilibrium is stable;
- $F_x(x^*,p)>0$: perturbations grow, so it is unstable.

Here $F_x=-2x$. The upper branch $x=+\sqrt p$ is stable and the lower branch $x=-\sqrt p$ is unstable. At $p=0$, the branches meet and disappear in a **fold**, also called a **saddle-node** or **limit point** bifurcation.

This example already shows why a diagram is more informative than a few simulations: it displays two coexisting equilibria, their stability, and the parameter at which both cease to exist.

## 4. How to read a bifurcation diagram

A typical equilibrium diagram uses:

- horizontal axis: continuation parameter $p$;
- vertical axis: one state component, a norm $\|u\|$, or another observable $h(u)$;
- solid curve: stable solution;
- dashed curve: unstable solution;
- marked points: folds, Hopf points, branch points, or other detected events.

Always check the legend. These conventions are common, not universal.

When reading a diagram, ask:

1. For this value of $p$, how many solutions intersect a vertical line?
2. Which of those solutions are stable?
3. Where does stability change?
4. Where does a branch turn, split, begin, or end?
5. Does the plotted observable hide distinct states? Two different vectors can have the same norm or same first component.

### 4.1 Do not confuse two kinds of diagram

An **equilibrium continuation diagram** plots roots of $f(u,p)=0$. A diagram made by simulating a map or ODE for many parameter values and plotting late-time samples is an **asymptotic orbit diagram**. Both are called bifurcation diagrams, but the data and algorithms are different. Continuation can reveal unstable invariant solutions; brute-force time simulation usually cannot.

## 5. Stability from linearization

Let $u(t)=u^*+v(t)$, where $v$ is a small perturbation. Taylor expansion around an equilibrium gives

$$
\dot v = J(u^*,p)v + \mathcal O(\|v\|^2),
\qquad
J(u^*,p)=\frac{\partial f}{\partial u}(u^*,p).
$$

The eigenvalues $\lambda_i$ of $J$ determine local linear stability for a continuous-time ODE:

- all $\operatorname{Re}\lambda_i<0$: asymptotically stable equilibrium;
- at least one $\operatorname{Re}\lambda_i>0$: unstable equilibrium;
- one or more $\operatorname{Re}\lambda_i=0$: the linear test is inconclusive and a bifurcation may be nearby.

The **stability margin** is often taken as

$$
m=\max_i \operatorname{Re}\lambda_i.
$$

Then $m<0$ is stable, $m>0$ is unstable, and $m\approx0$ is critical. In numerical work, “zero” always means “small relative to a scale-aware tolerance.”

For discrete-time maps $u_{k+1}=g(u_k,p)$, the rule changes: stability requires all eigenvalues of $D_ug$ to lie **inside the unit circle**. Do not apply the ODE left-half-plane rule to maps.

## 6. The common local bifurcations

A **normal form** is a simple equation that captures the local geometry near a generic bifurcation. It is not the full physical model; it is a local prototype.

| Type | Example normal form | Critical eigenvalue pattern | Local picture |
|---|---|---|---|
| Fold / saddle-node | $\dot x=p-x^2$ | one real eigenvalue reaches $0$ | two equilibria meet and disappear |
| Transcritical | $\dot x=px-x^2$ | one real eigenvalue reaches $0$ | two branches cross and exchange stability |
| Pitchfork | $\dot x=px-x^3$ | one real eigenvalue reaches $0$ | one symmetric branch meets two symmetry-broken branches |
| Hopf | a 2-D pair with parameter-dependent damping | one complex-conjugate pair crosses the imaginary axis | an equilibrium changes stability and a small periodic orbit may appear/disappear |

These eigenvalue patterns identify **candidates**, not a complete classification. Fold, transcritical, and pitchfork points can all have a zero real eigenvalue. Distinguishing them requires branch geometry, null vectors, symmetry information, derivative/nondegeneracy conditions, or normal-form coefficients.

### 6.1 Fold conditions

For the scalar equation $F(x,p)=0$, a generic fold at $(x_0,p_0)$ satisfies

$$
F=0,\qquad F_x=0,\qquad F_p\ne0,\qquad F_{xx}\ne0.
$$

The last two conditions rule out more degenerate behavior. In $n$ dimensions, $D_uF$ has a one-dimensional null space at a simple fold, together with appropriate transversality conditions.

### 6.2 Transcritical and pitchfork examples

For $\dot x=px-x^2$, equilibria are $x=0$ and $x=p$. They cross at the origin and exchange stability: a transcritical bifurcation.

For $\dot x=px-x^3=x(p-x^2)$, equilibria are $x=0$ for every $p$ and $x=\pm\sqrt p$ for $p\ge0$. The symmetric pair branches from the trivial solution at the origin: a supercritical pitchfork. Exact pitchforks usually depend on symmetry; imperfections often unfold them into folds.

### 6.3 Hopf bifurcation

For a Hopf candidate, a complex pair behaves locally like

$$
\lambda_{1,2}(p)=a(p)\pm i\omega(p),
$$

with $a(p_0)=0$ and $\omega(p_0)\ne0$. Generically, the pair must cross with nonzero speed, $a'(p_0)\ne0$, and no other eigenvalue may be critical. A normal-form coefficient, commonly the first Lyapunov coefficient, determines whether the emerging periodic orbit is supercritical or subcritical and its local stability.

Detecting the crossing of a complex pair does **not** by itself calculate the periodic-orbit branch. That needs periodic-orbit continuation, normally using shooting or collocation, plus Floquet multipliers for stability.

## 7. Why a simple parameter sweep fails at a fold

The simplest approach is **natural-parameter continuation**:

1. choose $p_{k+1}=p_k+\Delta p$;
2. use $u_k$ as the initial guess for Newton's method at $p_{k+1}$;
3. repeat.

Away from singular points this can work well. The implicit function theorem says that if $D_uF$ is nonsingular, the branch can locally be written as $u=u(p)$. At a fold, $D_uF$ is singular and $p$ is no longer a good coordinate along the curve. The derivative $du/dp$ becomes unbounded, Newton correction becomes ill-conditioned, and a monotone $p$ sweep cannot naturally turn back.

![Natural continuation stalls at a fold while pseudo-arclength continuation follows the curve through it.](assets/bifurcation-continuation.svg)

Reducing $\Delta p$ may postpone failure, but it does not repair the coordinate problem. The remedy is to parameterize progress by distance along the solution curve.

## 8. Pseudo-arclength continuation

The formulation below follows the standard predictor-corrector presentation used in [Thiele's introductory lecture](https://zenodo.org/records/4544848) and the [MatCont manual](https://www.staff.science.uu.nl/~kouzn101/NBA/ManualMatcontAug2019.pdf).

Write the combined state as

$$
z=\begin{bmatrix}u\\p\end{bmatrix}\in\mathbb R^{n+1}.
$$

The solution set $F(u,p)=0$ is typically a one-dimensional curve in this $(n+1)$-dimensional space. Pseudo-arclength continuation alternates a tangent prediction with a constrained Newton correction.

### 8.1 Compute a tangent

Differentiate $F(u(s),p(s))=0$ with respect to path coordinate $s$:

$$
D_uF\,\dot u + D_pF\,\dot p=0.
$$

Thus the tangent $t=(\dot u,\dot p)$ lies in the null space of the $n\times(n+1)$ matrix $[D_uF\;D_pF]$. It is normalized and oriented consistently with the previous tangent:

$$
\|t\|_2=1,\qquad t^Tt_{\text{old}}>0.
$$

A common numerical construction solves the bordered system

$$
\begin{bmatrix}
D_uF & D_pF\\
t_{\text{old},u}^T & t_{\text{old},p}
\end{bmatrix}
\begin{bmatrix}v_u\\v_p\end{bmatrix}
=
\begin{bmatrix}0\\1\end{bmatrix},
\qquad t=\frac{v}{\|v\|_2}.
$$

This formulation remains usable at a simple fold even though $D_uF$ alone is singular.

### 8.2 Predict

For step length $\Delta s$,

$$
z_{\text{pred}}=z_k+\Delta s\,t_k.
$$

The prediction is only a nearby point on the tangent line, not generally an exact solution.

### 8.3 Correct

Correct the prediction by solving $n+1$ equations for $n+1$ unknowns:

$$
G(z)=
\begin{bmatrix}
F(u,p)\\
t_k^T(z-z_{\text{pred}})
\end{bmatrix}=0.
$$

The first row puts the point back on the solution branch. The second confines it to a hyperplane normal to the old tangent, removing the extra degree of freedom. Newton's method repeatedly solves

$$
\begin{bmatrix}
D_uF & D_pF\\
t_{k,u}^T & t_{k,p}
\end{bmatrix}
\Delta z=-G(z),
\qquad z\leftarrow z+\Delta z.
$$

This bordered Newton system is the numerical heart of pseudo-arclength continuation.

### 8.4 Accept, adapt, repeat

If Newton converges and the corrected point is plausible, accept it, recompute the tangent and stability, and continue. A typical adaptive controller:

- increases $\Delta s$ after easy corrections;
- decreases it when Newton needs many iterations, curvature is high, or a special point is near;
- rejects a failed step and retries with a smaller $\Delta s$;
- stops below a minimum step size or outside requested bounds.

The step size is a numerical resolution, not a physical time step.

### 8.5 Algorithm in pseudocode

```text
given a corrected solution z0 = (u0, p0)
compute an initial unit tangent t0

for k = 0, 1, ...:
    z_pred = z[k] + ds * t[k]
    solve [F(z), t[k] dot (z - z_pred)] = 0 by Newton

    if correction failed:
        reduce ds and retry
    else:
        accept corrected z[k+1]
        compute and orient t[k+1]
        compute eigenvalues / test functions
        bracket and refine candidate events
        adapt ds
```

## 9. Newton correction, convergence, and scaling

For a nonlinear system $G(z)=0$, Newton's iteration is

$$
DG(z_j)\Delta z_j=-G(z_j),\qquad z_{j+1}=z_j+\Delta z_j.
$$

Near a regular root and from a good initial guess, convergence is fast. In continuation, the predictor supplies that guess. Failure can mean:

- the step was too large;
- the branch has high curvature;
- the Jacobian or bordered matrix is ill-conditioned;
- variables have wildly different scales;
- the initial point was not actually an accurate root;
- the model is nonsmooth or contains NaNs/discontinuities;
- the branch reached a singularity not handled by the algorithm.

Scale state variables and parameters so that one component does not dominate the arclength norm. In serious applications, use weighted norms and residuals. Always report both the nonlinear residual $\|F\|$ and the correction/constraint residual; “Newton returned” is not evidence of convergence.

## 10. Detecting and locating bifurcations numerically

Continuation creates a sequence of samples. A detector evaluates a scalar **test function** or tracks spectral information at those samples.

### 10.1 Detection is usually a bracket

For equilibrium ODEs, common signals include:

- a real Jacobian eigenvalue changing sign near zero: a zero-eigenvalue candidate;
- the real part of a complex-conjugate pair changing sign while its imaginary part stays nonzero: a Hopf candidate;
- a change in the number of unstable eigenvalues;
- a tangent parameter component $t_p$ changing sign: fold geometry;
- determinant or bordered-system test functions for small dense systems.

If a continuous test function $\phi$ changes sign between two accepted points, then those points bracket a candidate event. The location can be refined by bisection, secant interpolation, or an augmented nonlinear solve.

### 10.2 Why the determinant is often a poor test

$\det(D_uF)=0$ at a singular Jacobian, but determinants can overflow, underflow, and lose useful scale information in large systems. A smallest singular value, a tracked critical eigenvalue, a bordered solve, or a problem-specific test function is usually more robust.

### 10.3 Track eigenvalue identity carefully

Sorting eigenvalues independently at every continuation point can make modes exchange labels, producing false crossings or hiding real ones. Robust implementations match eigenvalues between steps using proximity and, when available, eigenvector information. Near repeated or tightly clustered eigenvalues, classification is intrinsically harder.

### 10.4 Candidate, refined point, classified point

Keep these three claims separate:

1. **Candidate:** a numerical indicator crossed or became small.
2. **Refined point:** a root-finding procedure located the indicator more accurately.
3. **Classified bifurcation:** mathematical nondegeneracy and transversality conditions, and sometimes normal-form coefficients, support a specific type.

An eigenvalue sign change is strong evidence of lost hyperbolicity, but it is not a proof that every genericity condition holds.

## 11. A complete beginner workflow

### Step 1: state the scientific question

Choose one physical parameter and one behavior you want to understand. For example: “At what input current does a resting neuron lose stability?” Avoid varying every parameter at once.

### Step 2: write a deterministic, smooth model

Express the model as $\dot u=f(u,p)$ and list units, parameter ranges, constraints, and known singularities. Continuation differentiates the model, so hidden clipping, conditionals, table lookups, and discontinuities matter.

### Step 3: find and verify one starting equilibrium

Use an analytic solution, physical limiting case, long simulation followed by root solving, or a dedicated nonlinear solver. Verify

$$
\|f(u_0,p_0)\| \ll 1.
$$

Also inspect the Jacobian condition and eigenvalues. Starting from a stable equilibrium is convenient, not required.

### Step 4: do a small natural sweep

Use it as a diagnostic. Plot the state, residual, Newton iterations, and stability margin. If it stalls near a turning point, that is a reason to switch to pseudo-arclength, not to conclude the branch ends.

### Step 5: run pseudo-arclength continuation in both directions

A single starting point has two tangent orientations. Following both directions reduces the chance that a relevant part of the branch is omitted.

### Step 6: compute stability along the branch

For small dense systems, compute the full spectrum. For large sparse systems, target only the rightmost eigenvalues. Record the number of unstable eigenvalues as well as a stable/unstable Boolean.

### Step 7: detect, refine, and classify events

Save the bracket, the refined state/parameter, critical eigenvalue(s), residual, tolerance, and classification evidence. For a Hopf point, record the estimated frequency $|\operatorname{Im}\lambda|$.

### Step 8: validate independently

Useful checks include:

- halve the continuation step size and compare event locations;
- tighten Newton and event tolerances;
- compare automatic derivatives against directional finite differences at a few points;
- simulate slightly to either side of stable branches;
- compare a normal-form example against its analytic solution;
- repeat using a second continuation implementation for publication-critical results.

### Step 9: interpret the diagram physically

Translate mathematical changes into the model's language: loss of a steady operating point, onset of oscillation, hysteresis, multistability, or sensitivity to parameters. Continuation describes invariant solutions; it does not by itself give basin sizes, transient durations, or noise-driven transition probabilities.

## 12. A tiny calculation you can do by hand

Use the equilibrium residual $F(x,p)=p-x^2=0$ and start at $(x_0,p_0)=(1,1)$.

1. The tangent equation is $-2x\,t_x+t_p=0$.
2. At $(1,1)$, choose the positive orientation. Setting $t_x=1$ gives $t_p=2$, so
   $t=(1,2)/\sqrt5$.
3. With $\Delta s=0.1$, predict

   $$
   (x_{\rm pred},p_{\rm pred})=(1,1)+0.1(1,2)/\sqrt5.
   $$

4. Correct by solving

   $$
   p-x^2=0,
   \qquad
   t_x(x-x_{\rm pred})+t_p(p-p_{\rm pred})=0.
   $$

5. Repeat with the opposite tangent orientation to move toward the fold.

As $x\to0$, natural continuation sees $dx/dp=1/(2x)\to\infty$. Pseudo-arclength instead reaches a tangent with $t_p=0$ and can continue onto the other branch. Work through one Newton correction on paper; it makes the bordered matrix far less mysterious.

## 13. Minimal experiments before using a full tool

Implement or calculate these in sequence:

1. **Root finding:** solve $x^2-p=0$ with Newton for fixed $p$.
2. **Natural continuation:** sweep $p$ downward and observe failure near $p=0$.
3. **Pseudo-arclength:** follow the full sideways parabola through the fold.
4. **Stability:** color the two branches using the sign of $-2x$ for $\dot x=p-x^2$.
5. **Pitchfork:** trace all branches of $x(p-x^2)=0$; notice that one starting branch does not automatically reveal every other branch.
6. **Hopf spectrum:** for

   $$
   J(p)=\begin{bmatrix}p&-\omega\\\omega&p\end{bmatrix},
   $$

   compute $\lambda=p\pm i\omega$ and detect the real-part crossing at $p=0$.

These exercises separate four distinct tasks: tracing, stability analysis, event detection, and branch switching.

## 14. Frequent misconceptions and failure modes

### “A finer simulation sweep is continuation”

No. Simulation samples attractors reached from chosen initial conditions. It usually misses unstable equilibria and may jump between attractors.

### “An eigenvalue close to zero proves a fold”

No. It marks a nonhyperbolic candidate. A transcritical or pitchfork point can show the same zero-eigenvalue signal, and numerical scaling can create false alarms.

### “Finding a Hopf point gives the oscillation amplitude”

No. It gives a local onset candidate and linear frequency estimate. Amplitude and stability of finite periodic orbits require a normal form locally or periodic-orbit continuation globally.

### “If Newton fails, the branch ended”

Often false. Retry with a smaller step, better scaling, a robust linear solver, or pseudo-arclength. Treat repeated failure as diagnostic evidence, not automatic physics.

### “Automatic differentiation makes everything exact”

It removes truncation error from derivative approximation for differentiable code, up to floating-point arithmetic. It does not repair a nonsmooth model, poor conditioning, wrong equations, or incorrect bifurcation criteria.

### “One continuation run finds the whole diagram”

Continuation follows one connected branch. Disconnected branches require other starting points; secondary branches require branch switching; isolas can remain hidden.

## 15. A six-week learning plan

### Week 1 — phase lines and equilibria

- Solve one-dimensional equilibria analytically.
- Draw phase lines and label stable/unstable points.
- Study saddle-node, transcritical, and pitchfork normal forms.
- Deliverable: three hand-drawn, stability-labeled diagrams.

### Week 2 — multidimensional stability

- Review Jacobians and eigenvalues.
- Classify 2-D linear systems.
- Linearize nonlinear systems around equilibria.
- Deliverable: a script that plots the maximum real eigenvalue along known roots.

### Week 3 — nonlinear solvers

- Implement scalar and vector Newton methods.
- Study conditioning, residuals, damping, and tolerances.
- Deliverable: convergence plots from multiple initial guesses.

### Week 4 — continuation

- Implement natural continuation.
- Derive the tangent, predictor, and bordered corrector.
- Implement pseudo-arclength on $x^2-p=0$.
- Deliverable: a branch that passes through the fold.

### Week 5 — detection and validation

- Track critical eigenvalues.
- Bracket and refine a zero crossing.
- Distinguish candidate detection from mathematical classification.
- Deliverable: a convergence table for the fold location versus step size/tolerance.

### Week 6 — one applied model

- Choose a two- or three-state model from your field.
- Find a trusted starting equilibrium.
- Continue it, calculate stability, and verify one special point.
- Deliverable: a reproducible notebook and a diagram whose axes, conventions, and numerical settings are fully documented.

## 16. How this maps onto JaxCont

The conceptual pieces above correspond directly to the project's architecture:

| Mathematical idea | JaxCont area | Present interpretation |
|---|---|---|
| $F(u,p)=0$ and derivatives | `problems/equilibrium.py` | equilibrium residual, state Jacobian, parameter derivative |
| predictor, corrector, tangent | `core/pseudo_arclength.py` and `core/predictor_corrector.py` | pseudo-arclength branch traversal |
| branch/result storage | `core/continuation.py` and `api.py` | continuation points, parameters, tangents, eigenvalues, events |
| equilibrium eigenvalues | `stability/eigenvalue.py` | local stability along a branch |
| fold/Hopf candidate tests | `bifurcations/fold.py`, `hopf.py`, and `detector.py` | detection and optional location refinement |
| diagrams | `utils/plotting.py` | stable/unstable branches and event markers |

JaxCont's intended advantage is that JAX can obtain $D_uF$ and $D_pF$ by automatic differentiation and can compile/vectorize numerical kernels. Those are implementation benefits; the mathematical assumptions, conditioning issues, validation duties, and distinction between detection and classification remain exactly the same.

At the time this guide was written, the safest learning-to-project bridge is:

1. equilibrium continuation;
2. natural versus pseudo-arclength algorithms;
3. equilibrium eigenvalue stability;
4. fold and Hopf candidate detection/refinement;
5. plotting and validation with the classical examples.

Periodic-orbit continuation, Floquet analysis, normal-form coefficients, robust branch switching, and two-parameter continuation should be treated as later/experimental topics until their implementations and documentation are complete. When JaxCont's public API stabilizes, this section can be expanded into a project-specific walkthrough without rewriting the theory chapters.

## 17. Recommended resources

### Start here: free and beginner-friendly

1. **MIT OpenCourseWare, “Flows and Bifurcations in One Dimension.”** Concise notes on fixed points, stability, saddle-node, transcritical, and pitchfork bifurcations. This is the best first reading before numerical algorithms: [lecture notes (PDF)](https://ocw.mit.edu/courses/12-006j-nonlinear-dynamics-chaos-fall-2022/mit12_006jf22_lec2-3.pdf).
2. **Blyth, Renson, and Marucci, “Tutorial of numerical continuation and bifurcation theory for systems and synthetic biology.”** A readable bridge from nonlinear dynamics to a complete applied continuation workflow: [open-access tutorial](https://arxiv.org/abs/2008.05226).
3. **Uwe Thiele, “Introduction to Numerical Continuation.”** A self-contained video lecture and slides aimed at advanced undergraduates, master's students, and beginning PhD students; it develops pseudo-arclength continuation step by step: [video and slides on Zenodo](https://zenodo.org/records/4544848).
4. **MIT OpenCourseWare, “Nonlinear dynamics: Stability and bifurcations.”** An undergraduate-level visual lecture hosted on YouTube: [course page](https://ocw.mit.edu/courses/18-s191-introduction-to-computational-thinking-fall-2020/external-resources/nonlinear-dynamics-stability-and-bifurcations_caffa9dd-f217-486f-9b6f-abf8a06f059b/) or [YouTube video](https://www.youtube.com/watch?v=D3jpfeQCISU).
5. **MIT OpenCourseWare, “Homotopy and Bifurcation.”** Useful after learning Newton's method; it connects root tracking, Jacobian singularity, and branching: [video/transcript](https://ocw.mit.edu/courses/10-34-numerical-methods-applied-to-chemical-engineering-fall-2015/resources/session-9-homotopy-and-bifurcation/).

### Books: choose by level

- **First theory book:** Steven Strogatz, *Nonlinear Dynamics and Chaos*, especially the chapters on one-dimensional flows, phase plane analysis, limit cycles, and bifurcations. It emphasizes geometric intuition.
- **Practical numerical book:** Rüdiger Seydel, *Practical Bifurcation and Stability Analysis*. It assumes roughly calculus-level preparation and includes continuation, branching calculations, periodic-solution stability, and many applications: [publisher page](https://link.springer.com/book/10.1007/978-1-4419-1740-9).
- **Advanced reference:** Yuri Kuznetsov, *Elements of Applied Bifurcation Theory*. Use it when you need nondegeneracy conditions, normal forms, codimension-two points, or rigorous classification: [4th-edition bibliographic page](https://research-portal.uu.nl/en/publications/elements-of-applied-bifurcation-theory-3/).

Do not begin by reading Kuznetsov cover to cover. Start with one-dimensional geometry, work examples, learn continuation, then use the advanced book as a reference.

### Software documentation for the next stage

- **AUTO-07p:** the classical continuation and bifurcation package for algebraic systems and ODE boundary-value problems; its site includes the manual and Doedel's numerical-analysis lecture notes: [official site](https://auto-07p.github.io/).
- **BifurcationKit.jl:** modern documentation with detailed continuation, eigensolver, bifurcation, and branch-switching examples: [official documentation](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/).
- **MatCont:** a mature MATLAB toolbox; its manual is valuable for standard terminology and the taxonomy of detected/continued points: [MatCont manual](https://www.staff.science.uu.nl/~kouzn101/NBA/ManualMatcontAug2019.pdf).
- **Kuznetsov's numerical bifurcation lecture collection:** notes on natural, pseudo-arclength, and Moore–Penrose continuation and numerical bifurcation analysis: [course materials](https://webspace.science.uu.nl/~kouzn101/NBA/index.html).

Reading another tool's tutorial is valuable even if you never install it: the mathematics transfers, while the API syntax does not.

## 18. Checklist before trusting a result

- [ ] The starting point satisfies the equilibrium residual tolerance.
- [ ] State variables and the continuation parameter are sensibly scaled.
- [ ] The branch was followed in both relevant directions.
- [ ] Accepted points have recorded residuals and solver status.
- [ ] Stability uses the correct rule for ODEs versus maps.
- [ ] Critical eigenvalues are tracked consistently between steps.
- [ ] Every reported bifurcation was refined, not merely sampled.
- [ ] Classification evidence goes beyond “an eigenvalue was small.”
- [ ] Results are stable under smaller steps and tighter tolerances.
- [ ] At least one independent analytic, simulation, or software check was performed.
- [ ] The diagram states its observable, line conventions, parameter units, and settings.
- [ ] Claims do not exceed the implemented feature (for example, equilibrium Hopf detection is not periodic-orbit continuation).

## 19. Where to go after this tutorial

Once equilibrium continuation feels natural, learn in this order:

1. branch switching at transcritical/pitchfork/other branch points;
2. periodic-orbit formulation by shooting and collocation;
3. Floquet multipliers and periodic-orbit bifurcations;
4. normal forms and Lyapunov coefficients;
5. two-parameter continuation of fold and Hopf curves;
6. matrix-free continuation and targeted eigensolvers for large systems;
7. problem-specific extensions for delay equations, PDE discretizations, or nonsmooth systems.

The durable mental model is simple: **find one invariant solution, follow its branch with a well-conditioned coordinate, calculate stability, detect loss of regularity, refine and classify the event, and validate independently.** Every advanced continuation method elaborates one of those steps.

---

### Suggested citation for this guide's core background

For academic work, cite the original mathematical or software references appropriate to the method you actually use. A strong general tutorial citation is:

> M. Blyth, L. Renson, and L. Marucci, “Tutorial of numerical continuation and bifurcation theory for systems and synthetic biology,” 2020, [arXiv:2008.05226](https://doi.org/10.48550/arXiv.2008.05226).

For pseudo-arclength algorithms and classification conditions, supplement it with a numerical bifurcation text such as Seydel or Kuznetsov and the documentation of the software used to produce the result.
