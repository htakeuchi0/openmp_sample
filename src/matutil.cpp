/**
 * @file matutil.cpp
 * @brief 行列演算の基本的機能を提供するソースファイル．
 *
 * すべて行列は列指向である．    
 * つまり，行列Aを表す1次元配列がaのとき，第(i, j)成分はa[i + j*m]である．
 */

#include <cmath>
#include <stdexcept>
#include <omp.h>
#include "include/matutil.hpp"

namespace matutil {

/*
 * 半正定値対称行列AのCholesky分解A=LL^Tを計算する．
 *
 * @param[in, out] a 半正定値対称行列Aを入力し，下三角部分にLを上書きして返す．
 * @param[in] m 行列の大きさ．
 * @return なし．
 */
void cholesky(double *a, int m) {
    double *at = new double[m*m];
    transpose(a, at, m, m);
    for (int j = 0; j < m; j++) {
        int jm = j*m;
        double sum = a[j + j*m];
        for (int k = 0; k < j; k++) {
            double entry = at[jm + k];
            sum -= entry * entry;
        }

        if (sum < 0) {
            double eps = 1.0e-5;
            if (::abs(sum) < eps) {
                sum = -sum;
            }
            else {
                throw std::runtime_error("Cholesky分解に失敗しました．");
            }
        }

        at[j + jm] = ::sqrt(sum);

        int k;
#ifdef _OPENMP
        #pragma omp parallel for private(k, sum)
#endif // _OPENMP
        for (int i = j + 1; i < m; i++) {
            int im = i*m;
            sum = at[i + jm];
            for (k = 0; k < j; k++) {
                sum -= at[im + k]*at[jm + k];
            }
            at[im + j] = sum / at[j + jm];
        }
    }
    transpose(at, a, m, m);
    delete[] at; at = nullptr;
}

/*
 * 与えられた正則な下三角行列Lとベクトルbについて，Lx=b の解を返す．
 *
 * @param[in] l 下三角行列．ただし，上三角部分に成分が入っていてもよい．
 * @param[in, out] b Lx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 下三角行列の大きさ．
 * @return なし．
 */
void solve_l(double *l, double *b, int m) {
    double *lt = new double[m*m];
    transpose(l, lt, m, m);
    for (int i = 0; i < m; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= lt[i*m + j]*b[j];
        }
        b[i] = sum / lt[i + i*m];
    }
    delete[] lt; lt = nullptr;
}

/*
 * 与えられた正則な下三角行列Lとベクトルbについて，Lx=b の解を返す．
 *
 * @param[in] l 下三角行列．ただし，上三角部分に成分が入っていてもよい．
 * @param[in, out] b Lx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 下三角行列の大きさ．
 * @param[in] start_idx ベクトルの参照開始インデックス
 * @return なし．
 */
void solve_l(double *l, double *b, int m, int start_idx) {
    double *lt = new double[m*m];
    transpose(l, lt, m, m);
    for (int i = 0, bi = start_idx; i < m; i++, bi++) {
        double sum = b[bi];
        for (int j = 0; j < i; j++) {
            sum -= lt[i*m + j]*b[start_idx + j];
        }
        b[bi] = sum / lt[i + i*m];
    }
    delete[] lt; lt = nullptr;
}

/*
 * 与えられた正則な下三角行列Lとベクトルbについて，L^Tx=b の解を返す．
 *
 * @param[in] l 下三角行列．ただし，上三角部分に成分が入っていてもよい．
 * @param[in, out] b L^Tx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 下三角行列の大きさ．
 * @return なし．
 */
void solve_lt(double *l, double *b, int m) {
    for (int i = m - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = m - 1; j > i; j--) {
            sum -= l[j + i*m]*b[j];
        }
        b[i] = sum / l[i + i*m];
    }
}

/*
 * 与えられた正則な下三角行列Lとベクトルbについて，L^Tx=b の解を返す．
 *
 * @param[in] l 下三角行列．ただし，上三角部分に成分が入っていてもよい．
 * @param[in, out] b L^Tx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 下三角行列の大きさ．
 * @param[in] start_idx ベクトルの参照開始インデックス
 * @return なし．
 */
void solve_lt(double *l, double *b, int m, int start_idx) {
    for (int i = m - 1, bi = m - 1 + start_idx; i >= 0; i--, bi--) {
        double sum = b[bi];
        for (int j = m - 1; j > i; j--) {
            sum -= l[j + i*m]*b[start_idx + j];
        }
        b[bi] = sum / l[i + i*m];
    }
}

/*
 * 与えられた正則な上三角行列Rとベクトルbについて，Rx=b の解を返す．
 *
 * @param[in] r 上三角行列．ただし，下三角部分に成分が入っていてもよい．
 * @param[in, out] b Rx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 上三角行列の大きさ．
 * @return なし．
 */
void solve_r(double *r, double *b, int m) {
    double *rt = new double[m*m];
    transpose(r, rt, m, m);
    for (int i = m - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = m - 1; j > i; j--) {
            sum -= rt[i*m + j]*b[j];
        }
        b[i] = sum / rt[i + i*m];
    }
    delete[] rt; rt = nullptr;
}

/*
 * 与えられた正則な上三角行列Rとベクトルbについて，Rx=b の解を返す．
 *
 * @param[in] r 上三角行列．ただし，下三角部分に成分が入っていてもよい．
 * @param[in, out] b Rx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 上三角行列の大きさ．
 * @param[in] start_idx ベクトルの参照開始インデックス
 * @return なし．
 */
void solve_r(double *r, double *b, int m, int start_idx) {
    double *rt = new double[m*m];
    transpose(r, rt, m, m);
    for (int i = m - 1, bi = m - 1 + start_idx; i >= 0; i--, bi--) {
        double sum = b[bi];
        for (int j = m - 1; j > i; j--) {
            sum -= rt[i*m + m]*b[start_idx + j];
        }
        b[bi] = sum / rt[i + i*m];
    }
    delete[] rt; rt = nullptr;
}

/*
 * 正定値対称行列Aとベクトルbについて，Ax=b の解を返す．
 *
 * @param[in] a 正定値対称行列．ただし，メソッド実行後，この行列の中身は破壊される．
 * @param[in, out] b Ax=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 行列の大きさ．
 * @return なし．
 */
void solve(double *a, double *b, int m) {
    cholesky(a, m);
    solve_l(a, b, m);
    solve_lt(a, b, m);
}

/*
 * 正定値対称行列Aと行列Bについて，AX=B の解を返す．
 *
 * @param[in] a 正定値対称行列．ただし，メソッド実行後，この行列の中身は破壊される．
 * @param[in, out] b AX=Bの右辺行列Bを入力し，解Xを上書きして返す．
 * @param[in] m 行列Aの大きさ．
 * @param[in] n 行列Bの列数．
 * @return なし．
 */
void solve(double *a, double *b, int m, int n) {
    cholesky(a, m);
    int mn = m*n;
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < mn; i += m) {
        solve_l(a, b, m, i);
        solve_lt(a, b, m, i);
    }
}

/*
 * ベクトルの2ノルムを返す．
 *
 * @param[in] v ベクトル
 * @param[in] m ベクトルの大きさ
 * @return ベクトルの2ノルム
 */
double norm(double *v, int m) {
    int i;
    double norm = 0.0;
    for (i = 0; i < m; i++) {
        double entry = v[i];
        norm += entry * entry;
    }
    norm = ::sqrt(norm);
    return norm;
}

/*
 * ベクトルを2ノルムで正規化する．
 *
 * @param[in, out] v ベクトル．正規化したベクトルで上書きして返す．
 * @param[in] m ベクトルの大きさ
 * @return 成功したときtrue
 */
bool normalize(double *v, int m) {
    double norm_val = norm(v, m);
    double eps = 1.0e-15;
    if (norm_val < eps) {
        return false;
    }

    int i;
    for (i = 0; i < m; i++) {
        v[i] = v[i] / norm_val;
    }
    return true;
}

/*
 * QR分解におけるベクトルをuをつくる．
 *
 * @param[in] a QR分解をする行列
 * @param[out] u ベクトルu
 * @param[in] m ベクトルの大きさ
 * @param[in] i 着目している列
 */
void qrdecomp_create_u(double *a, double *u, int m, int i) {
    int j;
    for (j = i; j < m; j++) {
        u[j - i] = a[j + i*m];
    }
}

/*
 * QR分解におけるベクトルuを更新する．
 *
 * @param[in, out] u ベクトルu. 更新したベクトルを上書きする．
 * @param[in] m ベクトルのサイズ
 * @param[in] entry uの先頭
 */
void qrdecomp_update_u(double *u, int m) {
    double norm_val = norm(u, m);
    double sign = (u[0] >= 0.0) ? +1.0 : -1.0;
    double s = -sign * norm_val;
    u[0] -= s;
    normalize(u, m);
}

/*
 * QR分解における行列Aを更新する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[in] u ベクトルu
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] i 着目している列のインデックス
 */
void qrdecomp_update_a(double *a, double *u, int m, int n, int i) {
    for (int j = i; j < n; j++) {
        double v_j = 0.0;
        int k;
        for (k = i; k < m; k++) {
            v_j += u[k - i] * a[k + j*m];
        }

        for (k = i; k < m; k++) {
            a[k + j*m] -= 2.0*u[k - i]*v_j;
        }
    }
}

/*
 * QR分解の第1列目に対する処理を実行する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[out] q 直交行列Q
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] u ベクトルu
 */
void qrdecomp_firstiter(double *a, double *q, int m, int n, double *u) {
    qrdecomp_create_u(a, u, m, 0);
    qrdecomp_update_u(u, m);
    qrdecomp_update_a(a, u, m, n, 0);

    int j, k;
#ifdef _OPENMP
    #pragma omp parallel for private(k)
#endif // _OPENMP
    for (j = 0; j < m; j++) {
        for (k = 0; k < m; k++) {
            q[k + j*m] = -2.0*u[k]*u[j];
        }
        q[j + j*m] += 1.0;
    }
}

/*
 * QR分解の第2列目以降の処理を実行する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[in, out] q 直交行列Q．更新後の行列で上書きする．
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] u ベクトルu
 * @param[in] i 着目している列のインデックス
 */
void qrdecomp_iter(double *a, double *q, int m, int n, double *u, int i) {
    qrdecomp_create_u(a, u, m, i);
    qrdecomp_update_u(u, m - i);
    qrdecomp_update_a(a, u, m, n, i);

    int j, k;
#ifdef _OPENMP
    #pragma omp parallel for private(k)
#endif // _OPENMP
    for (j = 0; j < m; j++) {
        double qv_j = 0.0;
        for (k = i; k < m; k++) {
            qv_j += q[j*m + k] * u[k - i];
        }

        for (k = i; k < m; k++) {
            q[j*m + k] -= 2.0*qv_j*u[k - i];
        }
    }
}

/*
 * QR分解の反復処理を直交行列Qを構成せずに実行する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] u ベクトルu
 * @param[in] i 着目している列のインデックス
 */
void qrdecomp_iter_noq(double *a, int m, int n, double *u, int i) {
    qrdecomp_create_u(a, u, m, i);
    qrdecomp_update_u(u, m - i);
    qrdecomp_update_a(a, u, m, n, i);
}

/*
 * 行列のQR分解をする．
 *
 * @param[in, out] a 行列A．上三角行列Rを上書きして返す．
 * @param[out] q 直交行列Q
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void qrdecomp(double *a, double *q, int m, int n) {
    double *u = new double[m];
    qrdecomp_firstiter(a, q, m, n, u);

    for (int i = 1; i < n; i++) {
        qrdecomp_iter(a, q, m, n, u, i);
    }

    int i, j;
#ifdef _OPENMP
    #pragma omp parallel for private(j)
#endif // _OPENMP
    for (i = 0; i < m; i++) {
        for (j = 0; j < i; j++) {
            double tmp = q[i + j*m];
            q[i + j*m] = q[i*m + j];
            q[i*m + j] = tmp;
        }
    }

    delete[] u; u = nullptr;
}

/*
 * 行列のQR分解を直交行列Qを構成せずに行う．
 *
 * @param[in, out] a 行列A．上三角行列Rを上書きして返す．
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void qrdecomp(double *a, int m, int n) {
    double *u = new double[m];
    for (int i = 0; i < n; i++) {
        qrdecomp_iter_noq(a, m, n, u, i);
    }

    delete[] u; u = nullptr;
}

/*
 * 与えたベクトルを更新する場合のQR分解の各列の処理を実行する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[in, out] b ベクトルb. A=QR に対する Q^Tb で上書きして返す．
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] u ベクトルu
 */
void qrdecompb_iter(double *a, double *b, int m, int n, double *u, int i) {
    qrdecomp_create_u(a, u, m, i);
    qrdecomp_update_u(u, m - i);
    qrdecomp_update_a(a, u, m, n, i);
    
    double qv_j = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+: qv_j)
#endif // _OPENMP
    for (int k = i; k < m; k++) {
        qv_j += b[k] * u[k - i];
    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int k = i; k < m; k++) {
        b[k] -= 2.0*qv_j*u[k - i];
    }
}

/*
 * 行列AのQR分解により，A=QR となる上三角行列Rを求め，ベクトルbに対してQ^Tbを返す．
 *
 * @param[in, out] a 行列a．上三角行列rを上書きして返す．
 * @param[in, out] b ベクトルb. A=QR に対する Q^Tb で上書きして返す．
 * @param[in] m 行列aの行数
 * @param[in] n 行列aの列数
 */
void qrdecompb(double *a, double *b, int m, int n) {
    double *u = new double[m];

    for (int i = 0; i < n; i++) {
        qrdecompb_iter(a, b, m, n, u, i);
    }

    delete[] u; u = nullptr;
}

/*
 * 行列積を返す．
 *
 * @param[in] a 行列A
 * @param[in] b 行列B
 * @param[out] c A, Bの積
 * @param[in] l 行列Aの行数
 * @param[in] m 行列Aの列数
 * @param[in] n 行列Bの列数
 */
void mmul(double *a, double *b, double *c, int l, int m, int n) {
    int i, j, k;

    double *at = new double[l*m];
    transpose(a, at, l, m);

#ifdef _OPENMP
    #pragma omp parallel for private(j, k)
#endif // _OPENMP
    for (i = 0; i < l; i++) {
        int im = i*m;
        for (j = 0; j < n; j++) {
            int jm = j*m;
            double sum = 0.0;
            for (k = 0; k < m; k++) {
                sum += at[k + im] * b[k + jm];
            }
            c[i + j*l] = sum;
        }
    }

    delete[] at; at = nullptr;
}

/*
 * 行列ベクトル積を返す．
 *
 * @param[in] a 行列A
 * @param[in] v ベクトルv
 * @param[out] b A, vの積
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void mmul(double *a, double *v, double *b, int m, int n) {
    int i, j;

    double *at = new double[n*m];
    transpose(a, at, m, n);

#ifdef _OPENMP
    #pragma omp parallel for private(j)
#endif // _OPENMP
    for (i = 0; i < m; i++) {
        int in = i*n;
        double sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += at[in + j] * v[j];
        }
        b[i] = sum;
    }

    delete[] at; at = nullptr;
}

/*
 * ベクトルをコピーする．
 *
 * @param[in] a コピー元ベクトル
 * @param[out] b コピー先ベクトル
 * @param[in] m ベクトルの大きさ
 */
void copy(double *a, double *b, int m) {
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < m; i++) {
        b[i] = a[i];
    }
}

/*
 * 行列をコピーする．
 *
 * @param[in] a コピー元行列A
 * @param[out] b コピー先行列B
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void copy(double *a, double *b, int m, int n) {
    int mn = m*n;
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < mn; i++) {
        b[i] = a[i];
    }
}

/*
 * 行列を表示する．
 *
 * @param[in] a 行列
 * @param[in] m 行数
 * @param[in] n 列数
 */
void print(double *a, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f\t", a[i + j*m]);
        }
        printf("\n");
    }
}

/*
 * ベクトルを表示する．
 *
 * @param[in] v ベクトル
 * @param[in] m 要素数
 */
void print(double *v, int m) {
    for (int i = 0; i < m; i++) {
        printf("%f\t", v[i]);
    }
    printf("\n");
}

/*
 * 行列を転置する．
 *
 * @param[in] a 行列A
 * @param[out] at 行列Aの転置行列
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void transpose(double *a, double *at, int m, int n) {
    int j;
#ifdef _OPENMP
    #pragma omp parallel for private(j)
#endif // _OPENMP
    for (int i = 0; i < m; i++) {
        int in = i*n;
        for (j = 0; j < n; j++) {
            at[j + in] = a[i + j*m];
        }
    }
}

/*
 * ベクトルの符号を反転して返す．
 *
 * @param[in, out] b ベクトル．この符号を反転したベクトルを上書きして返す．
 * @param[in] m ベクトルの大きさ
 * @return なし．
 */
void minus(double *b, int m) {
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < m; i++) {
        b[i] = -b[i];
    }
}

/*
 * ベクトルをスカラー倍して返す
 *
 * @param[in, out] b ベクトル．スカラー倍したベクトルを上書きして返す．
 * @param[in] a スカラー
 * @param[in] m ベクトルの大きさ
 * @return なし．
 */
void times(double *b, double a, int m) {
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < m; i++) {
        b[i] *= a;
    }
}

/*
 * 行列の和を返す．
 *
 * @param a 行列A
 * @param b 行列B
 * @param c 行列 A+B
 * @param m 行列Aの行数
 * @param n 行列Bの列数
 */
void add(double *a, double *b, double *c, int m, int n) {
    int size = m*n;
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

/*
 * 行列の差を返す．
 *
 * @param a 行列A
 * @param b 行列B
 * @param c 行列 A-B
 * @param m 行列Aの行数
 * @param n 行列Bの列数
 */
void sub(double *a, double *b, double *c, int m, int n) {
    int size = m*n;
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < size; i++) {
        c[i] = a[i] - b[i];
    }
}

/*
 * ベクトルの和を返す．
 *
 * @param u ベクトルu
 * @param v ベクトルv
 * @param w ベクトル u+v
 * @param m ベクトルuの要素数
 */
void add(double *u, double *v, double *w, int m) {
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < m; i++) {
        w[i] = u[i] + v[i];
    }
}

/*
 * ベクトルの差を返す．
 *
 * @param u ベクトルu
 * @param v ベクトルv
 * @param w ベクトル u-v
 * @param m ベクトルuの要素数
 */
void sub(double *u, double *v, double *w, int m) {
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < m; i++) {
        w[i] = u[i] - v[i];
    }
}

/*
 * 行階段形である一般の行列のランクを返す．
 *
 * @param[in] r 行階段形である一般の行列R
 * @param[in] m 行列Rの行数
 * @param[in] n 行列Rの列数
 * @return 方程式が解をもつときtrue
 */
int rank_r(double *r, int m, int n) {
    int rank = 0;
    double eps = 1.0e-10;
    double norm_val = norm(r, m*n);
    for (int j = 0; j < n; j++) {
        int index = rank + j*m;
        if (index >= m*n) {
            return rank;
        }

        if (std::abs(r[index]) > eps*norm_val) {
            rank++;
        }
    }
    return rank;
}

/*
 * 一般の行列のランクを返す．
 *
 * @param[in] a 行階段形である一般の行列A
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @return 方程式が解をもつときtrue
 */
int rank(double *a, int m, int n) {
    double *r = new double[m*n];
    copy(a, r, m, n);
    qrdecomp(r, m, n);
    int rank_val = rank_r(r, m, n);
    delete[] r; r = nullptr;
    return rank_val;
}

/*
 * 行階段形である一般の行列を係数行列とした方程式が解をもつ場合trueを返す．
 *
 * @param[in] r 行階段形である一般の行列R
 * @param[in] b 右辺ベクトルb
 * @param[in] m 行列Rの行数
 * @param[in] n 行列Rの列数
 * @return 方程式が解をもつときtrue
 */
bool has_solution_r(double *r, double *b, int m, int n) {
    int rank = rank_r(r, m, n);

    double eps = 1.0e-10;
    double norm_val = norm(r, m*n);
    for (int i = rank; i < m; i++) {
        if (std::abs(b[i]) > eps*norm_val) {
            return false;
        }
    }

    return true;
}

/*
 * 一般の行列を係数行列とした方程式が解をもつ場合trueを返す．
 *
 * @param[in] a 一般の行列A
 * @param[in] b 右辺ベクトルb
 * @param[in] m 行列Rの行数
 * @param[in] n 行列Rの列数
 * @return 方程式が解をもつときtrue
 */
bool has_solution(double *a, double *b, int m, int n) {
    double *r = new double[m*n];
    copy(a, r, m, n);
    
    double *qtb = new double[m];
    copy(b, qtb, m);
    qrdecompb(r, qtb, m, n);
    bool has_sol =  has_solution_r(r, qtb, m, n);
    delete[] r; r = nullptr;
    delete[] qtb; qtb = nullptr;
    return has_sol;
}

/*
 * 行階段形である一般の行列を係数行列とした方程式の解の一つを返す．
 *
 * @param[in] r 行階段形である一般の行列R
 * @param[in] b 右辺ベクトルb
 * @param[out] x Rx=bの解のひとつ
 * @param[in] m 行列Rの行数
 * @param[in] n 行列Rの列数
 * @return 方程式が解をもつときtrue
 */
bool general_solve_r(double *r, double *b, double *x, int m, int n) {
#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    int minlen = (m < n) ? m : n;
    int *indices = new int[minlen];

    int rank = 0;
    double eps = 1.0e-10;
    double norm_val = norm(r, m*n);
    for (int j = 0; j < n; j++) {
        if (std::abs(r[rank + j*m]) > eps*norm_val) {
            indices[rank] = j;
            rank++;
        }
    }

    for (int i = rank; i < m; i++) {
        if (std::abs(b[i]) > eps) {
            delete[] indices; indices = nullptr;
            return false;
        }
    }

    double *small_r = new double[rank*rank];

    int j;
#ifdef _OPENMP
    #pragma omp parallel for private(j)
#endif // _OPENMP
    for (int i = 0; i < rank; i++) {
        int irank = i*rank;
        for (j = 0; j <= i; j++) {
            small_r[j + irank] = r[j + indices[i]*m];
        }

        for (j = i + 1; j < rank; j++) {
            small_r[j + irank] = 0.0;
        }
    }

    solve_r(small_r, b, rank);

#ifdef _OPENMP
    #pragma omp parallel for
#endif // _OPENMP
    for (int i = rank - 1; i >= 0; i--) {
        x[indices[i]] = b[i];
    }

    delete[] small_r; small_r = nullptr;
    delete[] indices; indices = nullptr;

    return true;
}

/*
 * 一般の行列を係数行列とした方程式の解の一つを返す．
 *
 * @param[in] A 一般の行列A
 * @param[in] b 右辺ベクトルb
 * @param[out] x Ax=bの解のひとつ
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @return 方程式が解をもつときtrue
 */
bool general_solve(double *a, double *b, double *x, int m, int n) {
    qrdecompb(a, b, m, n);
    return general_solve_r(a, b, x, m, n);
}

} // namespace matutil
