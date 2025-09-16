## Cyclic Variational Quantum Eigensolver (CVQE) 🚀 

Cyclic Variational Quantum Eigensolver (CVQE) is a hardware-efficient framework for accurate ground-state quantum simulation on noisy intermediate-scale quantum (NISQ) devices.

CVQE introduces a measurement-driven feedback cycle: important Slater determinants are iteratively added to the reference superposition, while a fixed entangler (e.g., single-layer UCCSD) is reused throughout. This adaptive scheme enables CVQE to:

- Escape barren plateaus via a staircase-like descent in optimization

- Maintain chemical accuracy across weakly and strongly correlated regimes

- Achieve favorable accuracy–cost trade-offs compared to selected CI (e.g., SHCI)