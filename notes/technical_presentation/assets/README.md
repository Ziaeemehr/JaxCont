# Technical presentation assets

These assets are reviewed snapshots. Refresh them whenever their source example
changes, and do not edit them by hand.

Periodic-event snapshots were regenerated and reviewed on 2026-08-06 from
source revision `fd1b5e0cb228bf2e15c594cf0ff0cb06e3debe65`.

The PRC snapshot was regenerated and reviewed on 2026-08-06 from the Task 10
working tree based on source revision
`f2b8d8c9564e29f18f9dafb5628ef679236bd108`, by running `MPLBACKEND=Agg python
example_13_phase_response_curve.py` from `examples/`.

The PRC shooting-validation snapshot was regenerated and reviewed on
2026-08-06 from the Task 11 working tree based on source revision
`c0c5b92162eb7bc269b1d5ad500b7bda21e3a502`, by running
`MPLBACKEND=Agg python example_14_prc_shooting_validation.py` from `examples/`.

The MatCont gallery snapshots were regenerated and reviewed on 2026-08-20
from source revision `405f95d849780efccdd66259c07e030b3529118e`. Each example was
run from an empty temporary working directory with the repository on
`PYTHONPATH`; the figures are presentation copies of the generated
`images/matcont_*_overlay.png` files. The systematic command-line validator,
not visual overlap in these figures, remains the pass/fail authority.

| Asset | Produced from | Regeneration command | Used in |
|---|---|---|---|
| `example_08_period_doubling.png` | `examples/example_08_period_doubling.py` | `MPLBACKEND=Agg python example_08_period_doubling.py` from `examples/` | periodic-orbit chapter |
| `example_09_neimark_sacker.png` | `examples/example_09_neimark_sacker.py` | `MPLBACKEND=Agg python example_09_neimark_sacker.py` from `examples/` | periodic-orbit chapter |
| `example_13_phase_response_curve.png` | `examples/example_13_phase_response_curve.py` | `MPLBACKEND=Agg python example_13_phase_response_curve.py` from `examples/` | PRC chapter |
| `example_14_prc_shooting_validation.png` | `examples/example_14_prc_shooting_validation.py` | `MPLBACKEND=Agg python example_14_prc_shooting_validation.py` from `examples/` | validation chapter |
| `example_16_matcont_cubic_overlay.png` | `examples/example_16_matcont_cubic_overlay.py` | `JAX_PLATFORMS=cpu MPLBACKEND=Agg PYTHONPATH=<repo> python <repo>/examples/example_16_matcont_cubic_overlay.py` from an empty temporary directory | validation chapter |
| `example_17_matcont_vanderpol_overlay.png` | `examples/example_17_matcont_vanderpol_overlay.py` | `JAX_PLATFORMS=cpu MPLBACKEND=Agg PYTHONPATH=<repo> python <repo>/examples/example_17_matcont_vanderpol_overlay.py` from an empty temporary directory | validation chapter |
| `example_18_matcont_adaptive_control_overlay.png` | `examples/example_18_matcont_adaptive_control_overlay.py` | `JAX_PLATFORMS=cpu MPLBACKEND=Agg PYTHONPATH=<repo> python <repo>/examples/example_18_matcont_adaptive_control_overlay.py` from an empty temporary directory | validation chapter |
| `example_19_matcont_radial_cycle_overlay.png` | `examples/example_19_matcont_radial_cycle_overlay.py` | `JAX_PLATFORMS=cpu MPLBACKEND=Agg PYTHONPATH=<repo> python <repo>/examples/example_19_matcont_radial_cycle_overlay.py` from an empty temporary directory | validation chapter |
| `example_20_matcont_torbpc_overlay.png` | `examples/example_20_matcont_torbpc_overlay.py` | `JAX_PLATFORMS=cpu MPLBACKEND=Agg PYTHONPATH=<repo> python <repo>/examples/example_20_matcont_torbpc_overlay.py` from an empty temporary directory | validation chapter |

## Reviewed snapshot hashes

| Asset | SHA-256 |
|---|---|
| `example_08_period_doubling.png` | `47f36c955acae5d88761b108ccb72512eea182c6c2ab77989a3ca0b045e37ee3` |
| `example_09_neimark_sacker.png` | `5892bd7f9cd4fb6f39149d8f007770d9b1709a85edc9ebf50ffb1ec11d7098c5` |
| `example_13_phase_response_curve.png` | `ebebf73b2bf77ce8735b6b983cdfed1c09f2adc6c2e82873cc8a7505e1ddac0f` |
| `example_14_prc_shooting_validation.png` | `bfba008e049602d6687f8a69dad76d4e10b09d92dc9e08cc88d42589c762f1f3` |
| `example_16_matcont_cubic_overlay.png` | `b793328f9f8ab81f711533d588c059d339be73f4b3e2c148b5d239f5d9394a62` |
| `example_17_matcont_vanderpol_overlay.png` | `2fcc3b9f34874a33fb6c481249e524b864640fecf4e196efe1c9f3a7b57d1b43` |
| `example_18_matcont_adaptive_control_overlay.png` | `82882b7ba9a04379e6bd67178457fb0afef80b995777759b2cc1d0a21457e0a1` |
| `example_19_matcont_radial_cycle_overlay.png` | `575d3709c3909f6a10253c9c35bf0bbad47d7f49b16915588227c2a7b1b216b8` |
| `example_20_matcont_torbpc_overlay.png` | `addfa67f1b586b821d149f371fd8e71961e1f0c6b73c2ce992fde2cd0d5c35ff` |
