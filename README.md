# high-performance-linear-algebra → C++ matrix multiplication optimization
![C++](https://img.shields.io/badge/C++-20-blue?logo=c%2B%2B&logoColor=white)
![Build](https://img.shields.io/badge/Build-CMake-ff006e)
![Course](https://img.shields.io/badge/COE-322-white)
![Focus](https://img.shields.io/badge/Topic-High--Performance%20Computing-ff4d6d)

## Author

**Casey Hawthorne**  
Computational Engineering — University of Texas at Austin  
COE 322 – High Performance Linear Algebra
## Description
Below are descriptions of each file within this project folder.

### [highperformancelinearalgebra.cpp](highperformancelinearalgebra.cpp)
- Main implementation file containing the `Matrix` class + matrix multiplication algorithms.
- Includes:
  - Reference triple-loop matrix multiplication
  - Optimized **IKJ** loop ordering
  - Blocked matrix multiplication
  - Recursive matrix multiplication
  - Memory-aware matrix views using `std::span`
  - Raw-pointer optimized kernels for improved performance
  - Correctness tests and timing benchmarks for multiple matrix sizes.

### [CMakeLists.txt](CMakeLists.txt)
- Build configuration file to compile the project with CMake.
- Builds the executable:

```
highperformancelinearalgebra
```

### [writeup.pdf](writeup.pdf)
- Documentation describing the design process, matrix class implementation, performance optimization strategies, and benchmarking results.
- Includes figures explaining algorithm structure and performance comparisons.

## Installation

All dependencies for this project are provided through a standard C++ toolchain and CMakefile.

## Usage

### Build the project

From the project root directory:

```bash
mkdir build
cd build
cmake ..
make
```

This will generate executable:

```
highperformancelinearalgebra
```

### Run executable

Execute the compiled binary:

```bash
./highperformancelinearalgebra
```

The program will:

- Run correctness tests for small matrices
- Compare multiple matrix multiplication implementations
- Perform timing benchmarks across several matrix sizes
- Print comparative results

### Matrix Sizes Tested

Implemented benchmarking tests the following matrix sizes:

```
4
16
32
128
256
512
1024
```

### Recursive Thresholds Tested

Recursive multiplication is evaluated with the following thresholds:

```
8
16
32
64
```

## Output Metrics

Performance is reported as:

- Execution time (seconds)
- GFLOP/s
- Percentage of estimated peak performance

Peak performance estimate used in this program )In reference to my personal device which this was created on:
- Used CPU: Intel Ultra 9

```
44 GFLOP/s
```

## Matrix Class Features

- dynamic matrix allocation
- configurable leading dimensions (LDA)
- bounds-checked element access
- optimized raw pointer access for high-performance kernels

Submatrices allow efficient block decomposition without copying data and wasting of cache space:

```
TopLeft()
TopRight()
BotLeft()
BotRight()
```

The above are used in blocked and recursive multiplication algorithms.


## Implemented Algorithms

### Reference Matrix Multiplication
Standard triple-loop implementation used as a correctness baseline.

\[
C_{ij} = \sum_k A_{ik} B_{kj}
\]


### Optimized IKJ Multiplication
Improves cache performance by changing loop ordering:

```
for i
  for k
    for j
```

This increases memory locality and reuse of matrix rows.


### Blocked Matrix Multiplication
Divides matrices into smaller blocks to improve cache utilization and to better handle larger matrices.

### Recursive Matrix Multiplication
Partitions matrices into quadrants:

```
A = [A11 A12]
    [A21 A22]
```

Whose multiplications are then computed recursively until a threshold size is reached, where the leaf kernel is applied.


## Testing

The program performs built-in errorhandling checks including:

- reference multiplication comparison
- optimized multiplication comparison
- blocked multiplication validation
- recursive multiplication validation
- padded memory tests with varying leading dimensions (LDA)

Matrices are verified using approximate equality with tolerance **1e-9**
