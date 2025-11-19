# A Hybrid MATLAB–Python Framework for Accelerating Solid-State
Nanopore Signal Processing

Solid-state nanopore sensing generates large, high-resolution ionic current datasets that challenge the efficiency of conventional
MATLAB-based signal processing workflows. To address this, a hybrid MATLAB–Python computational
framework was developed, optimizing five critical nanopore signal processing routines—resampling, moving-mean
baseline correction, Whittaker smoothing, Poisson-based thresholding, and peak detection—by porting them from
MATLAB into Python. After refactoring and optimizations, Python implementations utilized vectorized operations
and just-in-time compilation to significantly improve computational performance. Integration between MATLAB and
Python was facilitated through a custom shared-memory interface, with automatic handling of data structures and dimensional
consistency. Although integration introduced overhead that reduced the performance gains for smaller-scale
routines, substantial runtime improvements were observed for computationally intensive processes. Future improvements
could expand performance analysis to include memory usage and more rigorous fidelity testing. The resulting
framework demonstrates a practical and scalable approach to accelerating nanopore data analysis without disrupting
existing experimental workflows.
