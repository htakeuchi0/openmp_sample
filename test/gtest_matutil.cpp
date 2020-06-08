/**
 * @file gtest_matutil.cpp
 * @brief テストケース．
 */

#include <random>
#include <iostream>
#include "gtest/gtest.h"
#include "include/matutil.hpp"

namespace matutil {

/**
 * 線形計算のテストケースクラス．
 */
class MatutilTest : public ::testing::Test {
protected:
    /**
     * 初期化する．
     */
    virtual void SetUp() {
        mt_.seed(1234L);
    }

public:
    /** 許容誤差 */
    static constexpr double EPS = 1.0e-8;

    /** メルセンヌツイスタ */
    std::mt19937 mt_;

    /** 乱数生成器 */
    std::uniform_real_distribution<double> rand_;
};

/**
 * 2ノルムが正しく計算できることを確認する．
 */
TEST_F(MatutilTest, norm) {
    int n = 2048;
    double *a = new double[n];

    double expected = 0.0;
    for (int i = 0; i < n; i++) {
        a[i] = rand_(mt_);
        double val = a[i];
        expected += val * val;
    }
    expected = std::sqrt(expected);

    double actual = matutil::norm(a, n);

    // 相対誤差を評価
    bool approx_equals = (std::abs(actual - expected) < MatutilTest::EPS * std::abs(expected));

    delete[] a; a = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 2ノルムによる正規化が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, normalize) {
    int n = 2048;
    double *actual = new double[n];
    double *expected = new double[n];

    double norm_val = 0.0;
    for (int i = 0; i < n; i++) {
        actual[i] = rand_(mt_);
        double val = actual[i];
        norm_val += val * val;
    }
    norm_val = std::sqrt(norm_val);

    for (int i = 0; i < n; i++) {
        expected[i] = actual[i] / norm_val;
    }

    matutil::normalize(actual, n);

    // 相対誤差を評価
    bool approx_equals = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(actual[i] - expected[i]) >= MatutilTest::EPS * std::abs(expected[i])) {
            approx_equals = false;
            break;
        }
    }

    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 転置が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, transpose) {
    int m = 2048;
    int n = 2048;
    double *a = new double[m*n];
    double *actual = new double[n*m];
    double *expected = new double[n*m];

    for (int i = 0, mn = m*n; i < mn; i++) {
        a[i] = rand_(mt_);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            expected[i + j*n] = a[i*m + j];
        }
    }

    matutil::transpose(a, actual, m, n);

    // 相対誤差を評価
    bool approx_equals = true;
    for (int i = 0, mn = m*n; i < mn; i++) {
        if (std::abs(actual[i] - expected[i]) >= MatutilTest::EPS * std::abs(expected[i])) {
            approx_equals = false;
            break;
        }
    }

    delete[] a; a = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 符号反転が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, minus) {
    int n = 2048;
    double *actual = new double[n];
    double *expected = new double[n];

    for (int i = 0; i < n; i++) {
        actual[i] = rand_(mt_);
        expected[i] = -actual[i];
    }

    matutil::minus(actual, n);

    // 相対誤差を評価
    bool approx_equals = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(actual[i] - expected[i]) >= MatutilTest::EPS * std::abs(expected[i])) {
            approx_equals = false;
            break;
        }
    }

    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * スカラー倍が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, times) {
    int n = 2048;
    double *actual = new double[n];
    double *expected = new double[n];
    double scalar = rand_(mt_);

    for (int i = 0; i < n; i++) {
        actual[i] = rand_(mt_);
        expected[i] = scalar*actual[i];
    }

    matutil::times(actual, scalar, n);

    // 相対誤差を評価
    bool approx_equals = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(actual[i] - expected[i]) >= MatutilTest::EPS * std::abs(expected[i])) {
            approx_equals = false;
            break;
        }
    }

    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * ベクトルの和が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, add_vector) {
    int n = 2048;
    double *a = new double[n];
    double *b = new double[n];
    double *actual = new double[n];
    double *expected = new double[n];

    for (int i = 0; i < n; i++) {
        a[i] = rand_(mt_);
        b[i] = rand_(mt_);
        expected[i] = a[i] + b[i];
    }

    matutil::add(a, b, actual, n);

    // 相対誤差を評価
    bool approx_equals = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(actual[i] - expected[i]) >= MatutilTest::EPS * std::abs(expected[i])) {
            approx_equals = false;
            break;
        }
    }

    delete[] a; a = nullptr;
    delete[] b; b = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 行列の和が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, add_matrix) {
    int m = 2048;
    int n = 2048;
    double *a = new double[m*n];
    double *b = new double[m*n];
    double *actual = new double[m*n];
    double *expected = new double[m*n];

    for (int i = 0, mn = m*n; i < mn; i++) {
        a[i] = rand_(mt_);
        b[i] = rand_(mt_);
        expected[i] = a[i] + b[i];
    }

    matutil::add(a, b, actual, m, n);

    // 相対誤差を評価
    bool approx_equals = true;
    for (int i = 0, mn = m*n; i < mn; i++) {
        if (std::abs(actual[i] - expected[i]) >= MatutilTest::EPS * std::abs(expected[i])) {
            approx_equals = false;
            break;
        }
    }

    delete[] a; a = nullptr;
    delete[] b; b = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * ベクトルの差が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, sub_vector) {
    int n = 2048;
    double *a = new double[n];
    double *b = new double[n];
    double *actual = new double[n];
    double *expected = new double[n];

    for (int i = 0; i < n; i++) {
        a[i] = rand_(mt_);
        b[i] = rand_(mt_);
        expected[i] = a[i] - b[i];
    }

    matutil::sub(a, b, actual, n);

    // 相対誤差を評価
    bool approx_equals = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(actual[i] - expected[i]) >= MatutilTest::EPS * std::abs(expected[i])) {
            approx_equals = false;
            break;
        }
    }

    delete[] a; a = nullptr;
    delete[] b; b = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 行列の差が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, sub_matrix) {
    int m = 2048;
    int n = 2048;
    double *a = new double[m*n];
    double *b = new double[m*n];
    double *actual = new double[m*n];
    double *expected = new double[m*n];

    for (int i = 0, mn = m*n; i < mn; i++) {
        a[i] = rand_(mt_);
        b[i] = rand_(mt_);
        expected[i] = a[i] - b[i];
    }

    matutil::sub(a, b, actual, m, n);

    // 相対誤差を評価
    bool approx_equals = true;
    for (int i = 0, mn = m*n; i < mn; i++) {
        if (std::abs(actual[i] - expected[i]) >= MatutilTest::EPS * std::abs(expected[i])) {
            approx_equals = false;
            break;
        }
    }

    delete[] a; a = nullptr;
    delete[] b; b = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 行列ベクトル積が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, mmul_matvec) {
    int m = 2048;
    int n = 1024;
    double *a = new double[m*n];
    double *b = new double[n];
    double *actual = new double[m];
    double *expected = new double[m];

    for (int i = 0, mn = m*n; i < mn; i++) {
        a[i] = rand_(mt_);
    }

    for (int i = 0; i < n; i++) {
        b[i] = rand_(mt_);
    }

    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += a[i + j*m] * b[j];
        }
        expected[i] = sum;
    }

    matutil::mmul(a, b, actual, m, n);

    // 差のノルムで評価
    double *sub = new double[m];
    matutil::sub(expected, actual, sub, m);
    double norm_val = matutil::norm(sub, m);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] a; a = nullptr;
    delete[] b; b = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 行列積が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, mmul_matmat) {
    int l = 2048;
    int m = 1024;
    int n = 4096;
    double *a = new double[l*m];
    double *b = new double[m*n];
    double *actual = new double[l*n];
    double *expected = new double[l*n];

    for (int i = 0, lm = l*m; i < lm; i++) {
        a[i] = rand_(mt_);
    }

    for (int i = 0, mn = m*n; i < mn; i++) {
        b[i] = rand_(mt_);
    }

    for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += a[i + k*l] * b[k + j*m];
            }
            expected[i + j*l] = sum;
        }
    }

    matutil::mmul(a, b, actual, l, m, n);

    // 差のノルムで評価
    double *sub = new double[m*n];
    matutil::sub(expected, actual, sub, m*n);
    double norm_val = matutil::norm(sub, m*n);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] a; a = nullptr;
    delete[] b; b = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * Cholesky分解が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, cholesky) {
    int n = 2048;
    double *a = new double[n*n];
    double *expected = new double[n*n];
    double *l = new double[n*n];
    double *lt = new double[n*n];
    double *actual = new double[n*n];

    // 対称行列になるように要素を配置
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
           a[i*n + j] = a[i + j*n] = rand_(mt_);
        }
    }

    // 狭義対角優位になるように対角成分を更新
    // これで正定値対称行列になる (Gerschgorinの定理)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
           sum += a[i*n + j];
        }
        for (int j = i+1; j < n; j++) {
           sum += a[i*n + j];
        }

        a[i*n + i] += sum + 1;
    }

    for (int i = 0, nn = n*n; i < nn; i++) {
        expected[i] = a[i];
    }

    matutil::cholesky(a, n);
    
    // Lの成分を抽出
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            l[i + j*n] = a[i + j*n];
        }

        for (int j = i+1; j < n; j++) {
            l[i + j*n] = 0.0;
        }
    }

    // LL^T が A に戻るかを確認
    matutil::transpose(l, lt, n, n);
    matutil::mmul(l, lt, actual, n, n, n);

    // 差のノルムで評価
    double *sub = new double[n*n];
    matutil::sub(expected, actual, sub, n*n);
    double norm_val = matutil::norm(sub, n*n);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] a; a = nullptr;
    delete[] l; l = nullptr;
    delete[] lt; lt = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * ベクトルのコピーが正しく計算できることを確認する．
 */
TEST_F(MatutilTest, copy_vec) {
    int n = 2048;
    double *a = new double[n];
    double *actual = new double[n];
    double *expected = new double[n];

    for (int i = 0; i < n; i++) {
        expected[i] = a[i] = rand_(mt_);
    }

    matutil::copy(a, actual, n);

    // 差のノルムで評価
    double *sub = new double[n];
    matutil::sub(expected, actual, sub, n);
    double norm_val = matutil::norm(sub, n);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] a; a = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 行列のコピーが正しく計算できることを確認する．
 */
TEST_F(MatutilTest, copy_mat) {
    int n = 1024;
    double *a = new double[n*n];
    double *actual = new double[n*n];
    double *expected = new double[n*n];

    for (int i = 0, nn = n*n; i < nn; i++) {
        expected[i] = a[i] = rand_(mt_);
    }

    matutil::copy(a, actual, n, n);

    // 差のノルムで評価
    double *sub = new double[n*n];
    matutil::sub(expected, actual, sub, n*n);
    double norm_val = matutil::norm(sub, n*n);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] a; a = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 下三角行列を係数行列にした線形方程式の解が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, solve_l_vec) {
    int n = 2048;
    double *a = new double[n*n];
    double *l = new double[n*n];
    double *expected = new double[n];
    double *actual = new double[n];

    for (int i = 0, nn = n*n; i < nn; i++) {
       a[i] = rand_(mt_);
    }

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
           sum += a[i*n + j];
        }
        for (int j = i+1; j < n; j++) {
           sum += a[i*n + j];
        }

        a[i*n + i] += sum + 1;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            l[i + j*n] = a[i + j*n];
        }

        for (int j = i+1; j < n; j++) {
            l[i + j*n] = 0.0;
        }
    }

    for (int i = 0; i < n; i++) {
        expected[i] = 1.0;
    }

    matutil::mmul(l, expected, actual, n, n);

    matutil::solve_l(a, actual, n);

    // 差のノルムで評価
    double *sub = new double[n];
    matutil::sub(expected, actual, sub, n);
    double norm_val = matutil::norm(sub, n);
    bool approx_equals = (norm_val < EPS);

    delete[] sub; sub = nullptr;
    delete[] a; a = nullptr;
    delete[] l; l = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 下三角行列を転置した行列を係数行列にした線形方程式の解が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, solve_lt_vec) {
    int n = 2048;
    double *a = new double[n*n];
    double *l = new double[n*n];
    double *lt = new double[n*n];
    double *expected = new double[n];
    double *actual = new double[n];

    for (int i = 0, nn = n*n; i < nn; i++) {
       a[i] = rand_(mt_);
    }

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
           sum += a[i*n + j];
        }
        for (int j = i+1; j < n; j++) {
           sum += a[i*n + j];
        }

        a[i*n + i] += sum + 1;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            l[i + j*n] = a[i + j*n];
        }

        for (int j = i + 1; j < n; j++) {
            l[i + j*n] = 0.0;
        }
    }

    matutil::transpose(l, lt, n, n);

    for (int i = 0; i < n; i++) {
        expected[i] = 1.0;
    }

    matutil::mmul(lt, expected, actual, n, n);

    matutil::solve_lt(a, actual, n);

    // 差のノルムで評価
    double *sub = new double[n];
    matutil::sub(expected, actual, sub, n);
    double norm_val = matutil::norm(sub, n);
    bool approx_equals = (norm_val < EPS);

    delete[] sub; sub = nullptr;
    delete[] a; a = nullptr;
    delete[] l; l = nullptr;
    delete[] lt; lt = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 上三角行列を係数行列にした線形方程式の解が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, solve_r_vec) {
    int n = 2048;
    double *a = new double[n*n];
    double *r = new double[n*n];
    double *expected = new double[n];
    double *actual = new double[n];

    for (int i = 0, nn = n*n; i < nn; i++) {
       a[i] = rand_(mt_);
    }

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
           sum += a[i*n + j];
        }
        for (int j = i+1; j < n; j++) {
           sum += a[i*n + j];
        }

        a[i*n + i] += sum + 1;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            r[i + j*n] = 0.0;
        }

        for (int j = i; j < n; j++) {
            r[i + j*n] = a[i + j*n];
        }
    }

    for (int i = 0; i < n; i++) {
        expected[i] = 1.0;
    }

    matutil::mmul(r, expected, actual, n, n);

    matutil::solve_r(a, actual, n);

    // 差のノルムで評価
    double *sub = new double[n];
    matutil::sub(expected, actual, sub, n);
    double norm_val = matutil::norm(sub, n);
    bool approx_equals = (norm_val < EPS);

    delete[] sub; sub = nullptr;
    delete[] a; a = nullptr;
    delete[] r; r = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 正定値対称行列を係数行列にした線形方程式の解が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, solve_vec) {
    int n = 2048;
    double *a = new double[n*n];
    double *expected = new double[n];
    double *actual = new double[n];

    // 対称行列になるように要素を配置
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
           a[i*n + j] = a[i + j*n] = rand_(mt_);
        }
    }

    // 狭義対角優位になるように対角成分を更新
    // これで正定値対称行列になる (Gerschgorinの定理)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
           sum += a[i*n + j];
        }
        for (int j = i+1; j < n; j++) {
           sum += a[i*n + j];
        }

        a[i*n + i] += sum + 1;
    }

    for (int i = 0; i < n; i++) {
        expected[i] = 1.0;
    }

    matutil::mmul(a, expected, actual, n, n);

    matutil::solve(a, actual, n);

    // 差のノルムで評価
    double *sub = new double[n];
    matutil::sub(expected, actual, sub, n);
    double norm_val = matutil::norm(sub, n);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] a; a = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 正定値対称行列を係数行列にした行列方程式の解が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, solve_mat) {
    int n = 2048;
    int m = 1024;
    double *a = new double[n*n];
    double *expected = new double[n*m];
    double *actual = new double[n*m];

    // 対称行列になるように要素を配置
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
           a[i*n + j] = a[i + j*n] = rand_(mt_);
        }
    }

    // 狭義対角優位になるように対角成分を更新
    // これで正定値対称行列になる (Gerschgorinの定理)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
           sum += a[i*n + j];
        }
        for (int j = i+1; j < n; j++) {
           sum += a[i*n + j];
        }

        a[i*n + i] += sum + 1;
    }

    for (int i = 0, nm = n*m; i < nm; i++) {
        expected[i] = rand_(mt_);
    }

    matutil::mmul(a, expected, actual, n, n, m);

    matutil::solve(a, actual, n, m);

    // 差のノルムで評価
    double *sub = new double[n*m];
    matutil::sub(expected, actual, sub, n*m);
    double norm_val = matutil::norm(sub, n*m);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] a; a = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * QR分解が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, qrdecomp) {
    int n = 2048;
    int m = 1024;
    double *a = new double[n*m];
    double *q = new double[n*n];
    double *expected = new double[n*m];
    double *actual = new double[n*m];

    for (int i = 0, nm = n*m; i < nm; i++) {
       a[i] = rand_(mt_);
    }

    for (int i = 0, nm = n*m; i < nm; i++) {
        expected[i] = a[i];
    }

    matutil::qrdecomp(a, q, n, m);

    // QR が A に戻ることを確認する
    matutil::mmul(q, a, actual, n, n, m);

    // 相対誤差を評価
    double *sub = new double[n*m];
    matutil::sub(expected, actual, sub, n*m);
    double norm_val = matutil::norm(sub, n*m);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] a; a = nullptr;
    delete[] q; q = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 直交行列Qを生成しないQR分解が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, qrdecomp_noq) {
    int n = 2048;
    int m = 1024;
    double *q = new double[n*n];
    double *expected = new double[n*m];
    double *actual = new double[n*m];

    for (int i = 0, nm = n*m; i < nm; i++) {
       actual[i] = rand_(mt_);
    }

    for (int i = 0, nm = n*m; i < nm; i++) {
        expected[i] = actual[i];
    }

    matutil::qrdecomp(expected, q, n, m);

    matutil::qrdecomp(actual, n, m);

    // 差のノルムで評価
    double *sub = new double[n*m];
    matutil::sub(expected, actual, sub, n*m);
    double norm_val = matutil::norm(sub, n*m);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] q; q = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * (A,b)に対してQR分解により(R,Q^Tb)が正しく計算できることを確認する．
 */
TEST_F(MatutilTest, qrdecompb) {
    int n = 2048;
    int m = 1024;
    double *a = new double[n*m];
    double *a_copy = new double[n*m];
    double *q = new double[n*n];
    double *qt = new double[n*n];
    double *expected = new double[n];
    double *actual = new double[n];

    for (int i = 0, nm = n*m; i < nm; i++) {
        a_copy[i] = a[i] = rand_(mt_);
    }

    for (int i = 0; i < n; i++) {
        actual[i] = rand_(mt_);
    }

    // QとRを計算
    matutil::qrdecomp(a_copy, q, n, m);

    // Q^Tbを計算
    matutil::transpose(q, qt, n, n);
    matutil::mmul(qt, actual, expected, n, n);

    // 直接RとQ^Tbを計算
    matutil::qrdecompb(a, actual, n, m);

    // 双方の計算によるQ^Tbの差のノルムで評価
    double *sub = new double[n];
    matutil::sub(expected, actual, sub, n);
    double norm_val = matutil::norm(sub, n);
    bool approx_equals = (norm_val < EPS);
    delete[] sub; sub = nullptr;

    delete[] a; a = nullptr;
    delete[] a_copy; a_copy = nullptr;
    delete[] q; q = nullptr;
    delete[] qt; qt = nullptr;
    delete[] actual; actual = nullptr;
    delete[] expected; expected = nullptr;

    ASSERT_TRUE(approx_equals); 
}

/**
 * 行列のランクが正しく計算できることを確認する．
 */
TEST_F(MatutilTest, rank) {
    int n = 2048;
    int m = 2048;
    int expected = 1024;
    double *a = new double[n*m];

    for (int i = 0; i < expected; i++) {
        for (int j = 0; j < m; j++) {
            a[i + j*n] = rand_(mt_);
        }
    }
    
    // ある行以下はそれまでの行ベクトルを使って定義する
    for (int i = expected; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i + j*n] = i*a[(i - expected) + j*n];
        }
    }

    int actual =  matutil::rank(a, n, m);

    delete[] a; a = nullptr;

    ASSERT_EQ(expected, actual); 
}

/**
 * 方程式が解をもつと正しく判断できることを確認する．
 */
TEST_F(MatutilTest, has_solution_true) {
    int n = 2048;
    int m = 2048;
    int rank = 1024;
    bool expected = true;
    double *a = new double[n*m];
    double *b = new double[n];

    // 指定ランクの行列を作る
    for (int i = 0; i < rank; i++) {
        for (int j = 0; j < m; j++) {
            a[i + j*n] = rand_(mt_);
        }
    }

    for (int i = rank; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i + j*n] = i*a[(i - rank) + j*n];
        }
    }

    // 方程式が解をもつように0で埋める
    for (int i = 0; i < rank; i++) {
        b[i] = rand_(mt_);
    }

    for (int i = rank; i < n; i++) {
        b[i] = 0.0;
    }

    // 直交行列Qで変換しておく
    double *q = new double[n*n];
    double *a_copy = new double[n*m];
    matutil::copy(a, a_copy, n, m);
    matutil::qrdecomp(a_copy, q, n, m);
    double *qb = new double[n];
    matutil::mmul(q, b, qb, n, n);

    bool actual =  matutil::has_solution(a, qb, n, m);

    delete[] q; q = nullptr;
    delete[] a_copy; a_copy = nullptr;
    delete[] qb; qb = nullptr;
    delete[] a; a = nullptr;
    delete[] b; b = nullptr;

    ASSERT_EQ(expected, actual); 
}

/**
 * 方程式が解をもたないと正しく判断できることを確認する(1)．
 */
TEST_F(MatutilTest, has_solution_false_top) {
    int n = 2048;
    int m = 2048;
    int rank = 1024;
    bool expected = false;
    double *a = new double[n*m];
    double *b = new double[n];

    // 指定ランクの行列を作る
    for (int i = 0; i < rank; i++) {
        for (int j = 0; j < m; j++) {
            a[i + j*n] = rand_(mt_);
        }
    }

    for (int i = rank; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i + j*n] = i*a[(i - rank) + j*n];
        }
    }

    // 方程式が解をもつように0で埋める
    for (int i = 0; i < rank; i++) {
        b[i] = rand_(mt_);
    }

    for (int i = rank; i < n; i++) {
        b[i] = 0.0;
    }

    // 0部分の先頭だけ非ゼロ成分を入れる
    b[rank] = rand_(mt_);

    // 直交行列Qで変換しておく
    double *q = new double[n*n];
    double *a_copy = new double[n*m];
    matutil::copy(a, a_copy, n, m);
    matutil::qrdecomp(a_copy, q, n, m);
    double *qb = new double[n];
    matutil::mmul(q, b, qb, n, n);

    bool actual =  matutil::has_solution(a, qb, n, m);

    delete[] q; q = nullptr;
    delete[] a_copy; a_copy = nullptr;
    delete[] qb; qb = nullptr;
    delete[] a; a = nullptr;
    delete[] b; b = nullptr;

    ASSERT_EQ(expected, actual); 
}

/**
 * 方程式が解をもたないと正しく判断できることを確認する(2)．
 */
TEST_F(MatutilTest, has_solution_false_last) {
    int n = 2048;
    int m = 2048;
    int rank = 1024;
    bool expected = false;
    double *a = new double[n*m];
    double *b = new double[n];

    // 指定ランクの行列を作る
    for (int i = 0; i < rank; i++) {
        for (int j = 0; j < m; j++) {
            a[i + j*n] = rand_(mt_);
        }
    }

    for (int i = rank; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i + j*n] = i*a[(i - rank) + j*n];
        }
    }

    // 方程式が解をもつように0で埋める
    for (int i = 0; i < rank; i++) {
        b[i] = rand_(mt_);
    }

    for (int i = rank; i < n; i++) {
        b[i] = 0.0;
    }

    // 0部分の末尾だけ非ゼロ成分を入れる
    b[n - 1] = rand_(mt_);

    // 直交行列Qで変換しておく
    double *q = new double[n*n];
    double *a_copy = new double[n*m];
    matutil::copy(a, a_copy, n, m);
    matutil::qrdecomp(a_copy, q, n, m);
    double *qb = new double[n];
    matutil::mmul(q, b, qb, n, n);

    bool actual =  matutil::has_solution(a, qb, n, m);

    delete[] q; q = nullptr;
    delete[] a_copy; a_copy = nullptr;
    delete[] qb; qb = nullptr;
    delete[] a; a = nullptr;
    delete[] b; b = nullptr;

    ASSERT_EQ(expected, actual); 
}

} // namespace


