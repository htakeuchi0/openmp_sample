/**
 * @file main.cpp
 * @brief メインメソッドをもつソースファイル．
 */

#include "include/matutil.hpp"
#include <iostream>
#include <random>
#include <chrono>

namespace {

/**
 * 定義どおりに行列積を計算して返す．
 *
 * @param[in] a 行列A
 * @param[in] b 行列B
 * @param[out] c A, Bの積
 * @param[in] l 行列Aの行数
 * @param[in] m 行列Aの列数
 * @param[in] n 行列Bの列数
 */
void mmul_naive(double *a, double *b, double *c, int l, int m, int n) {
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            int jm = j*m;
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += a[i + k*l] * b[k + jm];
            }
            c[i + j*l] = sum;
        }
    }
}

/**
 * (A^T)^TBとして計算した行列積を返す．
 *
 * @param[in] a 行列A
 * @param[in] b 行列B
 * @param[out] c A, Bの積
 * @param[in] l 行列Aの行数
 * @param[in] m 行列Aの列数
 * @param[in] n 行列Bの列数
 */
void mmul_simple(double *a, double *b, double *c, int l, int m, int n) {
    double *at = new double[l*m];
    matutil::transpose(a, at, l, m);

    for (int i = 0; i < l; i++) {
        int il = i*l;
        for (int j = 0; j < n; j++) {
            int jm = j*m;
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += at[k + il] * b[k + jm];
            }
            c[i + j*l] = sum;
        }
    }

    delete[] at; at = nullptr;
}

} // namespace

/**
 * メインメソッド
 *
 * @param[in] argc コマンドライン引数の数
 * @param[in] argv コマンドライン引数
 * @return int 終了コード
 */
int main(int argc, char **argv) {

#ifndef _OPENMP
    printf("OpenMPが利用できません．処理を終了します．\n");
    return 1;
#endif // _OPENMP

    std::cout << "AB=C..." << std::endl;

    int m = 2048;
    double *a = new double[m*m];
    double *b = new double[m*m];
    double *c_naive = new double[m*m];
    double *c_simple = new double[m*m];
    double *c_openmp = new double[m*m];

    std::mt19937 mt;
    mt.seed(1234L);
    std::uniform_real_distribution<double> rand;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            a[i*m + j] = rand(mt);
            b[i*m + j] = rand(mt);
        }
    }

    // 定義どおりの素朴な実装
    auto start = std::chrono::system_clock::now();
    ::mmul_naive(a, b, c_naive, m, m, m);
    auto end = std::chrono::system_clock::now();
    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0);
    std::cout << "Naive:\n";
    std::cout << time << " [sec]\n";
    std::cout << std::endl;

    // 通常の実装
    start = std::chrono::system_clock::now();
    ::mmul_simple(a, b, c_simple, m, m, m);
    end = std::chrono::system_clock::now();
    time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0);
    std::cout << "Simple:\n";
    std::cout << time << " [sec]\n";
    std::cout << std::endl;

    // OpenMPを利用した実装
    start = std::chrono::system_clock::now();
    matutil::mmul(a, b, c_openmp, m, m, m);
    end = std::chrono::system_clock::now();
    time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0);
    std::cout << "OpenMP:\n";
    std::cout << time << " [sec]\n";
    std::cout << std::endl;

    double *sub = new double[m*m];

    // 素朴な実装の通常の実装の結果の差
    matutil::sub(c_naive, c_simple, sub, m, m);
    double diff_norm = matutil::norm(sub, m*m);
    std::cout << "Naive - Simple:\n";
    std::cout << diff_norm << "\n";
    std::cout << std::endl;

    // 素朴な実装のOpenMPで並列化した実装の結果の差
    matutil::sub(c_naive, c_openmp, sub, m, m);
    diff_norm = matutil::norm(sub, m*m);
    std::cout << "Naive - OpenMP:\n";
    std::cout << diff_norm << "\n";
    std::cout << std::endl;
    
    delete[] a; a = nullptr;
    delete[] b; b = nullptr;
    delete[] c_naive; c_naive = nullptr;
    delete[] c_simple; c_simple = nullptr;
    delete[] c_openmp; c_openmp = nullptr;
    delete[] sub; sub = nullptr;
    return 0;
}
