# JaxCont Documentation Plan
## From Dynamical System to Complete Bifurcation Analysis
### A Practical Workflow Using JaxCont (and Applicable to Other Packages)

---

# Goal

This document aims to bridge the gap between bifurcation theory textbooks and software manuals by teaching a **systematic workflow** for analyzing nonlinear dynamical systems.

Instead of focusing only on mathematical theory or software commands, it answers the practical question:

> **Given a new dynamical system, what should I do first, what comes next, and when do I stop?**

Although examples will use **JaxCont**, the methodology is universal and can be applied with AUTO, MatCont, BifurcationKit.jl, COCO, PyDSTool, or other continuation software.

---

# Target Audience

- Beginners learning bifurcation analysis
- Researchers analyzing new dynamical systems
- Users of JaxCont
- Users migrating from AUTO, MatCont, or BifurcationKit
- Graduate students in applied mathematics, physics, biology, engineering, and neuroscience

---

# Learning Objectives

After completing this guide, readers should be able to

- Understand the complete bifurcation analysis workflow
- Choose appropriate continuation parameters
- Compute and analyze equilibria
- Determine stability using Jacobian eigenvalues
- Detect local bifurcations
- Continue bifurcation points
- Compute periodic orbits
- Analyze codimension-2 bifurcations
- Validate continuation results using numerical simulations
- Use JaxCont to perform the complete analysis

---

# Organization

The guide consists of four major parts.

---

# Part I — Foundations

## Chapter 1 — Introduction

- What is bifurcation analysis?
- Why continuation methods?
- Numerical simulation vs continuation
- Typical workflow
- Overview of JaxCont

---

## Chapter 2 — Mathematical Background

- Dynamical systems
- Phase space
- Equilibria
- Stability
- Jacobian matrix
- Eigenvalues
- Local linearization

---

## Chapter 3 — Types of Bifurcations

Introduce

- Saddle-node (Fold)
- Pitchfork
- Transcritical
- Hopf
- Period doubling
- Homoclinic
- SNIC
- Global bifurcations

Each section includes

- intuition
- mathematics
- diagrams
- numerical examples

---

# Part II — Practical Workflow

This is the core of the document.

---

## Chapter 4 — Step 1: Understand the Model

Given

\[
\dot{x}=f(x,p)
\]

Identify

- state variables
- parameters
- parameter ranges
- physical interpretation
- expected behavior

Questions to answer

- What are the unknowns?
- Which parameters are interesting?
- Which variables should be plotted?

---

## Chapter 5 — Step 2: Compute Equilibria

Solve

\[
f(x,p)=0
\]

Topics

- analytical solutions
- numerical Newton solver
- multiple equilibria
- initialization strategies

JaxCont examples

---

## Chapter 6 — Step 3: Analyze Stability

Compute

\[
J=\frac{\partial f}{\partial x}
\]

Topics

- Jacobian
- eigenvalues
- stable node
- saddle
- spiral
- center

Decision tree

Stable?

- yes → continuation
- no → investigate instability

---

## Chapter 7 — Step 4: Continue Equilibrium Branches

Choose continuation parameter.

Topics

- natural continuation
- pseudo-arclength continuation
- predictor-corrector
- branch diagrams

Outputs

- equilibrium curves
- stability changes

---

## Chapter 8 — Step 5: Detect Bifurcations

Monitor

- eigenvalues
- determinant
- test functions

Detect

- Fold
- Hopf
- Pitchfork
- Transcritical

Explain

- mathematical condition
- numerical detection
- geometric interpretation

---

## Chapter 9 — Step 6: Branch Switching

Once a bifurcation is detected

How to

- switch branches
- initialize new continuation
- verify correctness

---

## Chapter 10 — Step 7: Periodic Orbits

Starting from Hopf

Topics

- initial cycle generation
- continuation
- Floquet multipliers
- stable and unstable cycles

Outputs

- orbit diagrams
- period
- amplitude

---

## Chapter 11 — Step 8: Codimension-2 Analysis

Continue special points

Examples

- Bogdanov–Takens
- Generalized Hopf
- Hopf-Hopf
- Cusp
- Bautin

Explain why codimension-2 points organize parameter space.

---

## Chapter 12 — Step 9: Global Bifurcations

Topics

- homoclinic
- heteroclinic
- SNIC
- fold of cycles
- torus bifurcation

---

## Chapter 13 — Step 10: Validation

Always compare continuation with simulation.

Checklist

- trajectory simulation
- phase portraits
- basin of attraction
- transient behavior
- parameter sweeps

---

# Part III — Using JaxCont

Each workflow step is paired with corresponding JaxCont commands.

Each chapter follows the same structure.

## Concept

What is being computed?

## Mathematics

Relevant equations.

## Numerical Algorithm

Underlying continuation algorithm.

## JaxCont Implementation

Minimal code example.

## Expected Output

Plots and interpretation.

---

# Part IV — Complete Case Studies

The entire workflow is demonstrated on several systems.

---

## Case Study 1 — Brusselator

From equations to

- equilibria
- Hopf bifurcation
- periodic orbit
- parameter continuation

---

## Case Study 2 — Van der Pol Oscillator

Topics

- limit cycles
- continuation
- relaxation oscillations

---

## Case Study 3 — FitzHugh–Nagumo

Topics

- Hopf
- excitability
- periodic solutions

---

## Case Study 4 — Lorenz System

Topics

- equilibria
- symmetry
- continuation
- chaos (discussion only)

---

## Case Study 5 — Duffing Oscillator

Topics

- multiple equilibria
- folds
- periodic forcing

---

## Additional Examples

Possible future additions

- Hindmarsh–Rose
- Morris–Lecar
- Schnakenberg
- Oregonator
- Predator–Prey models

---

# Appendices

## Appendix A

Newton's Method

---

## Appendix B

Pseudo-Arclength Continuation

---

## Appendix C

Automatic Differentiation with JAX

---

## Appendix D

Bordered Linear Solvers

---

## Appendix E

Eigenvalue Computation

---

## Appendix F

Floquet Theory

---

## Appendix G

Normal Forms

---

## Appendix H

Glossary

Definitions of common terminology.

---

## Appendix I

Reference Tables

Summary of

- bifurcation conditions
- test functions
- numerical methods
- stability criteria

---

# Workflow Summary

```text
Define the model
        │
        ▼
Compute equilibria
        │
        ▼
Compute Jacobian
        │
        ▼
Analyze eigenvalues
        │
        ▼
Stable?
   ├──────────────┐
   │              │
Yes             No
   │              │
Continue     Detect bifurcation
   │              │
   └──────┬───────┘
          ▼
Locate bifurcation point
          │
          ▼
Branch switching
          │
          ▼
Continue new branch
          │
          ▼
Periodic orbits
          │
          ▼
Codimension-2 continuation
          │
          ▼
Global bifurcations
          │
          ▼
Validate using simulations
          │
          ▼
Interpret results
```

---

# Design Principles

The guide should emphasize

- **Workflow before algorithms**
- **Intuition before mathematics**
- **Examples before abstraction**
- **Theory immediately followed by JaxCont implementation**
- **Consistent terminology throughout**
- **Decision trees and flowcharts**
- **One running example carried through the entire guide**
- **Standalone case studies for independent practice**

---

# Long-Term Vision

This guide should become the primary reference for users learning practical bifurcation analysis with JaxCont. Rather than serving only as software documentation, it should function as a comprehensive handbook that teaches readers how to approach and solve bifurcation problems systematically, regardless of the continuation software they ultimately use.