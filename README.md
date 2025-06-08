# KochAntiSnoflake
A comparison between a classic Python and a JAX-optimized implementation of the Koch anti-snowflake fractal. Includes performance benchmarks and GPU acceleration using JAX.

## Description
In this repository you will find to aproaches to generate an anti-snowflake fractal.

- `ClassicKoch.py` A traditional, recursive python version.
- `JaxKoch.py` An optimized version with GPU acceleration, vectorization and just in time compilation that includes a benchmark with the classic version.

## Dependencies

```
pip install -U jax[cuda12] numpy matplotlib
```

## Execution
In your console of choice, make sure you are in the same folder as the two files of this repository.
- Classic version:
```
python ClassicKoch.py
```
- Jax version:
```
python JaxKoch.py
```
