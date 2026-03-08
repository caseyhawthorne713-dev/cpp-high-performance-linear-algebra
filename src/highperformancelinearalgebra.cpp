//////////////////////////////////////////////////
// HIGH PERFORMANCE LINEAR ALGEBRA IN C++       //
// COMBINED SOLUTIONS FOR EXERCISES 53.1–53.10  //
// INCLUDES FINAL MATRIX CLASS AND TESTS        //
// FALL2025 UT COURSE COE322 - CASEY HAWTHORNE  //
//////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <span>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <algorithm>

class Matrix {
    int M{}, N{}, LDA{};                // rows, cols, leading dimension (stride)
    std::vector<double> owner;          // owns memory when non-empty
    std::span<double> data;             // view into memory (owner or external)
    bool is_external = false;

public:
    // Owned matrix: contiguous rows with LDA == n (default)
    Matrix(int m = 0, int n = 0)
        : M(m), N(n), LDA(n), owner(m * n), data(owner.data(), owner.size()) {}

    // Owned with explicit lda (lda >= n)
    Matrix(int m, int n, int lda)
        : M(m), N(n), LDA(lda), owner(m * lda), data(owner.data(), owner.size()) {
        if (lda < n) throw std::invalid_argument("lda must be >= n");
    }

    // External view constructor (does NOT take ownership)
    Matrix(int m, int n, int lda, double* ptr)
        : M(m), N(n), LDA(lda), owner(), data(ptr, m * lda), is_external(true) {
        if (lda < n) throw std::invalid_argument("lda must be >= n");
        if (ptr == nullptr && m * lda > 0) throw std::invalid_argument("null pointer for external data");
    }

    // Factory names (book vs intuitive)
    static Matrix External(int m, int n, int lda, double* ptr) { return Matrix(m, n, lda, ptr); }
    static Matrix External_book(int m, int lda, int n, double* ptr) { return Matrix(m, n, lda, ptr); }

    // Accessors
    int rows() const { return M; }
    int cols() const { return N; }
    int lda()  const { return LDA; }

    // raw_data: const and non-const overloads
    double* raw_data() { return data.data(); }
    const double* raw_data() const { return data.data(); }

    // get_double_data() analogue
    double* get_double_data() { return raw_data(); }
    const double* get_double_data() const { return raw_data(); }

    // bounds-checked element access
    double& at(int i, int j) {
        if (i < 0 || i >= M || j < 0 || j >= N) throw std::out_of_range("Matrix::at index out of range");
        return data[i * LDA + j];
    }
    const double& at(int i, int j) const {
        if (i < 0 || i >= M || j < 0 || j >= N) throw std::out_of_range("Matrix::at index out of range");
        return data[i * LDA + j];
    }

    // zero logical MxN region
    void zero() {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                data[i * LDA + j] = 0.0;
    }

    // printing
    void print(std::ostream& os = std::cout) const {
        os << std::fixed << std::setprecision(6);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j)
                os << std::setw(10) << at(i, j);
            os << "\n";
        }
        os << std::defaultfloat;
    }

    // approx equal
    bool approx_equal(const Matrix& B, double tol = 1e-9) const {
        if (rows() != B.rows() || cols() != B.cols()) return false;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                if (std::abs(at(i, j) - B.at(i, j)) > tol) return false;
        return true;
    }

    // add_inplace: dest += (*this) + other
    // Used in recursive algorithm: T1.add_inplace(T2, C11) -> C11 += T1 + T2
    void add_inplace(const Matrix& other, Matrix& dest) const {
        if (M != other.M || N != other.N || M != dest.M || N != dest.N)
            throw std::runtime_error("add_inplace: dimension mismatch");
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                dest.data[i * dest.LDA + j] += this->data[i * LDA + j] + other.data[i * other.LDA + j];
    }

    // Submatrix views (TopLeft, TopRight, BotLeft, BotRight)
    // These create external views pointing into the same storage.
    Matrix TopLeft(int m_sub, int n_sub) const {
        if (m_sub < 0 || n_sub < 0 || m_sub > M || n_sub > N) throw std::runtime_error("TopLeft: bad dims");
        double* ptr = const_cast<double*>(raw_data()) + 0 * LDA + 0;
        return Matrix::External(m_sub, n_sub, LDA, ptr);
    }
    Matrix TopRight(int m_sub, int n_sub) const {
        if (m_sub < 0 || n_sub < 0 || m_sub > M || n_sub > N) throw std::runtime_error("TopRight: bad dims");
        double* ptr = const_cast<double*>(raw_data()) + 0 * LDA + (N - n_sub);
        return Matrix::External(m_sub, n_sub, LDA, ptr);
    }
    Matrix BotLeft(int m_sub, int n_sub) const {
        if (m_sub < 0 || n_sub < 0 || m_sub > M || n_sub > N) throw std::runtime_error("BotLeft: bad dims");
        double* ptr = const_cast<double*>(raw_data()) + (M - m_sub) * LDA + 0;
        return Matrix::External(m_sub, n_sub, LDA, ptr);
    }
    Matrix BotRight(int m_sub, int n_sub) const {
        if (m_sub < 0 || n_sub < 0 || m_sub > M || n_sub > N) throw std::runtime_error("BotRight: bad dims");
        double* ptr = const_cast<double*>(raw_data()) + (M - m_sub) * LDA + (N - n_sub);
        return Matrix::External(m_sub, n_sub, LDA, ptr);
    }

    // Declarations (definitions below) 
    void MatMult(const Matrix& B, Matrix& C) const;        // C = A*B (overwrite)
    Matrix MatMult(const Matrix& B) const;                 // return A*B

    void BlockedMatMult(const Matrix& B, Matrix& C, int BLOCK_SIZE = 64) const;
    void RecursiveMatMult(const Matrix& B, Matrix& C, int THRESHOLD = 32) const;

    static Matrix ReferenceMatMult(const Matrix& A, const Matrix& B);
};

// 53.7: macros and optimized indexing
// Release (non-DEBUG) leaf kernel will use these macros.
#ifdef DEBUG
  // When debugging we will use at() directly in leaf kernel.
#else
  #define A_AT_raw(ad, ldaA, i, j) (ad[(size_t)(i)*(size_t)(ldaA) + (size_t)(j)])
  #define B_AT_raw(bd, ldaB, i, j) (bd[(size_t)(i)*(size_t)(ldaB) + (size_t)(j)])
  #define C_AT_raw(cd, ldaC, i, j) (cd[(size_t)(i)*(size_t)(ldaC) + (size_t)(j)])
#endif

// Leaf kernel: C += A*B (no zeroing)
// This is the single place that performs the raw-pointer fast multiply. Higher-level
// algorithms (blocked/recursive) call this to accumulate partial products.
inline void MatMultLeafAdd(const Matrix& A, const Matrix& B, Matrix& C) {
    // dimension check: A.rows x A.cols  *  B.rows x B.cols  -> C.rows x C.cols,
    if (A.cols() != B.rows() || A.rows() != C.rows() || B.cols() != C.cols())
        throw std::runtime_error("MatMultLeafAdd: dimension mismatch");

#ifdef DEBUG
    // safe, but slower: use at()
    for (int i = 0; i < A.rows(); ++i) {
        for (int k = 0; k < A.cols(); ++k) {
            double aik = A.at(i, k);
            for (int j = 0; j < B.cols(); ++j)
                C.at(i, j) += aik * B.at(k, j);
        }
    }
#else
    const double* ad = A.raw_data();
    const double* bd = B.raw_data();
    double* cd = C.raw_data();
    int ldaA = A.lda(), ldaB = B.lda(), ldaC = C.lda();

    // IKJ ordering: for reuse of A row and B row
    for (int i = 0; i < A.rows(); ++i) {
        for (int k = 0; k < A.cols(); ++k) {
            double aik = A_AT_raw(ad, ldaA, i, k);
            // unroll or pointer bumps could be applied here in future
            for (int j = 0; j < B.cols(); ++j) {
                C_AT_raw(cd, ldaC, i, j) += aik * B_AT_raw(bd, ldaB, k, j);
            }
        }
    }
#endif
}

// Simple wrapper: C = A*B (zeros C then accumulates)
inline void MatMultOptimized(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.cols() != B.rows() || A.rows() != C.rows() || B.cols() != C.cols())
        throw std::runtime_error("MatMultOptimized: dimension mismatch");
    C.zero();
    MatMultLeafAdd(A, B, C);
}

// Matrix method definitions 
void Matrix::MatMult(const Matrix& B, Matrix& C) const {
    if (rows() != C.rows() || B.cols() != C.cols() || cols() != B.rows())
        throw std::runtime_error("MatMult: dimension mismatch");
    MatMultOptimized(*this, B, C);
}

Matrix Matrix::MatMult(const Matrix& B) const {
    Matrix C(rows(), B.cols());
    MatMult(B, C);
    return C;
}

Matrix Matrix::ReferenceMatMult(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) throw std::runtime_error("ReferenceMatMult: dimension mismatch");
    Matrix C(A.rows(), B.cols());
    C.zero();
    int M = A.rows(), N = B.cols(), K = A.cols();
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += A.at(i, k) * B.at(k, j);
            C.at(i, j) = s;
        }
    return C;
}

// 53.3.1 BlockedMatMult (one-level blocking)
void Matrix::BlockedMatMult(const Matrix& B, Matrix& C, int BLOCK_SIZE) const {
    const Matrix& A = *this;
    int M_ = A.rows(), N_ = B.cols(), K_ = A.cols();
    if (M_ != C.rows() || N_ != C.cols() || K_ != B.rows())
        throw std::runtime_error("BlockedMatMult: dimension mismatch");

    C.zero();

    // Use submatrix views and call the leaf adder for each block product:
    for (int ii = 0; ii < M_; ii += BLOCK_SIZE) {
        int i_max = std::min(ii + BLOCK_SIZE, M_);
        for (int kk = 0; kk < K_; kk += BLOCK_SIZE) {
            int k_max = std::min(kk + BLOCK_SIZE, K_);
            for (int jj = 0; jj < N_; jj += BLOCK_SIZE) {
                int j_max = std::min(jj + BLOCK_SIZE, N_);

                // create external submatrix views: A_sub (i_max-ii x k_max-kk), B_sub (k_max-kk x j_max-jj), C_sub (i_max-ii x j_max-jj)
                double* A_ptr = const_cast<double*>(A.raw_data()) + ii * A.lda() + kk;
                double* B_ptr = const_cast<double*>(B.raw_data()) + kk * B.lda() + jj;
                double* C_ptr = C.raw_data() + ii * C.lda() + jj;

                Matrix A_sub(i_max - ii, k_max - kk, A.lda(), A_ptr);
                Matrix B_sub(k_max - kk, j_max - jj, B.lda(), B_ptr);
                Matrix C_sub(i_max - ii, j_max - jj, C.lda(), C_ptr);

                // accumulate partial product into C_sub
                MatMultLeafAdd(A_sub, B_sub, C_sub);
            }
        }
    }
}

// 53.9 RecursiveMatMult (2x2 block recursion)
// Uses TopLeft/TopRight/BotLeft/BotRight for submatrix views and uses MatMultLeafAdd at leaves.
void Matrix::RecursiveMatMult(const Matrix& B, Matrix& C, int THRESHOLD) const {
    const Matrix& A = *this;
    if (A.rows() != C.rows() || B.cols() != C.cols() || A.cols() != B.rows())
        throw std::runtime_error("RecursiveMatMult: dimension mismatch");

    // base case: if small enough call leaf add (zero C first)
    if (A.rows() <= THRESHOLD || A.cols() <= THRESHOLD || B.cols() <= THRESHOLD) {
        C.zero();
        MatMultLeafAdd(A, B, C);
        return;
    }

    // split sizes (floor halves)
    int m2 = A.rows() / 2;
    int k2 = A.cols() / 2;
    int n2 = B.cols() / 2;

    // build submatrix views (careful with odd sizes)
    Matrix A11 = A.TopLeft(m2, k2);
    Matrix A12 = A.TopRight(m2, A.cols() - k2);
    Matrix A21 = A.BotLeft(A.rows() - m2, k2);
    Matrix A22 = A.BotRight(A.rows() - m2, A.cols() - k2);

    Matrix B11 = B.TopLeft(k2, n2);
    Matrix B12 = B.TopRight(k2, B.cols() - n2);
    Matrix B21 = B.BotLeft(B.rows() - k2, n2);
    Matrix B22 = B.BotRight(B.rows() - k2, B.cols() - n2);

    Matrix C11 = C.TopLeft(m2, n2);
    Matrix C12 = C.TopRight(m2, C.cols() - n2);
    Matrix C21 = C.BotLeft(C.rows() - m2, n2);
    Matrix C22 = C.BotRight(C.rows() - m2, C.cols() - n2);

    // temporaries for partial products (owned, with lda >= n)
    Matrix T1(m2, n2, C11.lda());
    Matrix T2(m2, n2, C11.lda());

    // C11 += A11*B11 + A12*B21
    T1.zero(); T2.zero();
    A11.RecursiveMatMult(B11, T1, THRESHOLD);
    A12.RecursiveMatMult(B21, T2, THRESHOLD);
    T1.add_inplace(T2, C11);

    // C12 += A11*B12 + A12*B22
    T1.zero(); T2.zero();
    A11.RecursiveMatMult(B12, T1, THRESHOLD);
    A12.RecursiveMatMult(B22, T2, THRESHOLD);
    T1.add_inplace(T2, C12);

    // C21 += A21*B11 + A22*B21
    T1.zero(); T2.zero();
    A21.RecursiveMatMult(B11, T1, THRESHOLD);
    A22.RecursiveMatMult(B21, T2, THRESHOLD);
    T1.add_inplace(T2, C21);

    // C22 += A21*B12 + A22*B22
    T1.zero(); T2.zero();
    A21.RecursiveMatMult(B12, T1, THRESHOLD);
    A22.RecursiveMatMult(B22, T2, THRESHOLD);
    T1.add_inplace(T2, C22);
}

// Timing helper 
template <typename F>
double measure_time(F&& func) {
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    func();
    auto t2 = clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    return elapsed.count();
}

// Main: small correctness tests + timing 
int main() {
    std::cout << "Running small correctness tests (4x4)...\n";

    // A_small
    Matrix A_small(4, 4);
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) A_small.at(i, j) = i * 4 + j + 1;

    double raw[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    Matrix B_small = Matrix::External_book(4, 4, 4, raw);

    std::cout << "A_small:\n"; A_small.print();
    std::cout << "B_small:\n"; B_small.print();

    // Reference naive IJK
    Matrix C_ref = Matrix::ReferenceMatMult(A_small, B_small);

    // MatMult (top-level: zero + leaf add)
    Matrix C_opt(4,4);
    A_small.MatMult(B_small, C_opt);

    std::cout << "C_ref (reference):\n"; C_ref.print();
    std::cout << "C_opt (optimized IKJ):\n"; C_opt.print();
    std::cout << "IKJ equals ref? " << (C_ref.approx_equal(C_opt) ? "YES" : "NO") << "\n";

    // Blocked MatMult
    Matrix C_blocked(4,4);
    A_small.BlockedMatMult(B_small, C_blocked, 2);
    std::cout << "C_blocked (blocked):\n"; C_blocked.print();
    std::cout << "Blocked equals ref? " << (C_ref.approx_equal(C_blocked) ? "YES" : "NO") << "\n";

    // Recursive MatMult
    Matrix C_rec(4,4);
    A_small.RecursiveMatMult(B_small, C_rec, 2);
    std::cout << "C_rec (recursive):\n"; C_rec.print();
    std::cout << "Recursive equals ref? " << (C_ref.approx_equal(C_rec) ? "YES" : "NO") << "\n";

    // 53.6 test: add_inplace for different LDA
    std::cout << "\nTesting add_inplace (53.6)...\n";
    Matrix X(2,3,5); // m=2, n=3, lda=5 (padding per row)
    Matrix Y(2,3,6); // lda different
    Matrix D(2,3);   // destination
    X.zero(); Y.zero(); D.zero();
    for (int i=0;i<2;i++) for (int j=0;j<3;j++){ X.at(i,j) = 1.0 + i + j; Y.at(i,j) = 10.0 + i + j; }
    Matrix T1 = X; Matrix T2 = Y;
    T1.add_inplace(T2, D);
    std::cout << "D after add_inplace (should be X+Y):\n"; D.print();

// Timing / performance with multiple recursive thresholds 
std::cout << "\nTiming tests (naive / blocked / recursive with thresholds) — various sizes\n";
const double peak_GFLOPs = 44.0; // estimated peak

// Include small, moderate, and large matrices
std::vector<int> sizes = {4, 16, 32, 128, 256, 512, 1024};
std::vector<int> thresholds = {8, 16, 32, 64}; // recursive cutoff points

for (int N : sizes) {
    std::cout << "\nMatrix size: " << N << " x " << N << "\n";
    Matrix A(N,N), B(N,N), C(N,N);

    // Initialize matrices with simple values
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++){
            A.at(i,j) = (i+j)%100 + 1;
            B.at(i,j) = (i-j)%100 + 1;
        }

    double flops = 2.0 * N * (double)N * (double)N;

    // Naive triple-loop 
    C.zero();
    double t_naive = measure_time([&](){
        for (int i=0;i<N;i++)
            for (int j=0;j<N;j++){
                double s = 0.0;
                for (int k=0;k<N;k++) s += A.at(i,k) * B.at(k,j);
                C.at(i,j) = s;
            }
    });
    double g_naive = flops / (t_naive * 1e9);

    // Blocked 
    C.zero();
    double t_block = measure_time([&](){ A.BlockedMatMult(B, C, 64); });
    double g_block = flops / (t_block * 1e9);

    // Recursive for multiple thresholds 
    for (int TH : thresholds) {
        if (N < TH) continue; // skip if matrix smaller than threshold

        C.zero();
        double t_rec = measure_time([&](){ A.RecursiveMatMult(B, C, TH); });
        double g_rec = flops / (t_rec * 1e9);

        std::cout << "Recursive (TH=" << TH << "): " 
                  << t_rec << " s, " 
                  << g_rec << " GFLOP/s, " 
                  << (g_rec/peak_GFLOPs*100.0) << "% of peak\n";
    }

    std::cout << "Naive:    " << t_naive << " s, " << g_naive << " GFLOP/s, "
              << (g_naive/peak_GFLOPs*100.0) << "% of peak\n";
    std::cout << "Blocked:  " << t_block << " s, " << g_block << " GFLOP/s, "
              << (g_block/peak_GFLOPs*100.0) << "% of peak\n";
}


std::cout << "\nAll tests completed.\n";
return 0;


}
