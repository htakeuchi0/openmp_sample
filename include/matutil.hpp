/**
 * @file matutil.hpp
 * @brief 行列演算の基本的機能を提供するソースファイルのヘッダファイル．
 */

#ifndef OPENMP_SAMPLE_MATUTIL_HPP_
#define OPENMP_SAMPLE_MATUTIL_HPP_

namespace matutil {

/**
 * 半正定値対称行列AのCholesky分解A=LL^Tを計算する．
 *
 * @param[in, out] a 半正定値対称行列Aを入力し，下三角部分にLを上書きして返す．
 * @param[in] m 行列の大きさ．
 * @return なし．
 */
void cholesky(double *a, int m);

/**
 * 与えられた正則な下三角行列Lとベクトルbについて，Lx=b の解を返す．
 *
 * @param[in] l 下三角行列．ただし，上三角部分に成分が入っていてもよい．
 * @param[in, out] b Lx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 下三角行列の大きさ．
 * @return なし．
 */
void solve_l(double *l, double *b, int m);

/**
 * 与えられた正則な下三角行列Lとベクトルbについて，Lx=b の解を返す．
 *
 * @param[in] l 下三角行列．ただし，上三角部分に成分が入っていてもよい．
 * @param[in, out] b Lx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 下三角行列の大きさ．
 * @param[in] start_idx ベクトルの参照開始インデックス
 * @return なし．
 */
void solve_l(double *l, double *b, int m, int start_idx);

/**
 * 与えられた正則な下三角行列Lとベクトルbについて，L^Tx=b の解を返す．
 *
 * @param[in] l 下三角行列．ただし，上三角部分に成分が入っていてもよい．
 * @param[in, out] b L^Tx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 下三角行列の大きさ．
 * @return なし．
 */
void solve_lt(double *l, double *b, int m);

/**
 * 与えられた正則な下三角行列Lとベクトルbについて，L^Tx=b の解を返す．
 *
 * @param[in] l 下三角行列．ただし，上三角部分に成分が入っていてもよい．
 * @param[in, out] b L^Tx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 下三角行列の大きさ．
 * @param[in] start_idx ベクトルの参照開始インデックス
 * @return なし．
 */
void solve_lt(double *l, double *b, int m, int start_idx);

/**
 * 与えられた正則な上三角行列Rとベクトルbについて，Rx=b の解を返す．
 *
 * @param[in] r 上三角行列．ただし，下三角部分に成分が入っていてもよい．
 * @param[in, out] b Rx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 上三角行列の大きさ．
 * @return なし．
 */
void solve_r(double *r, double *b, int m);

/**
 * 与えられた正則な上三角行列Rとベクトルbについて，Rx=b の解を返す．
 *
 * @param[in] r 上三角行列．ただし，下三角部分に成分が入っていてもよい．
 * @param[in, out] b Rx=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 上三角行列の大きさ．
 * @param[in] start_idx ベクトルの参照開始インデックス
 * @return なし．
 */
void solve_r(double *r, double *b, int m, int start_idx);

/**
 * 正定値対称行列Aとベクトルbについて，Ax=b の解を返す．
 *
 * @param[in] a 正定値対称行列．ただし，メソッド実行後，この行列の中身は破壊される．
 * @param[in, out] b Ax=bの右辺ベクトルbを入力し，解xを上書きして返す．
 * @param[in] m 下三角行列の大きさ．
 * @return なし．
 */
void solve(double *a, double *b, int m);

/**
 * 正定値対称行列Aと行列Bについて，AX=B の解を返す．
 *
 * @param[in] a 正定値対称行列．ただし，メソッド実行後，この行列の中身は破壊される．
 * @param[in, out] b AX=Bの右辺行列Bを入力し，解Xを上書きして返す．
 * @param[in] m 行列Aの大きさ．
 * @param[in] n 行列Bの列数．
 * @return なし．
 */
void solve(double *a, double *b, int m, int n);

/**
 * ベクトルの2ノルムを返す．
 *
 * @param[in] v ベクトル
 * @param[in] m ベクトルの大きさ
 * @return ベクトルの2ノルム
 */
double norm(double *v, int m);

/**
 * ベクトルを2ノルムで正規化する．
 *
 * @param[in, out] v ベクトル．正規化したベクトルで上書きして返す．
 * @param[in] m ベクトルの大きさ
 * @return 成功したときtrue
 */
bool normalize(double *v, int m);

/**
 * QR分解におけるベクトルをuをつくる．
 *
 * @param[in] a QR分解をする行列
 * @param[out] u ベクトルu
 * @param[in] m ベクトルの大きさ
 * @param[in] i 着目している列
 */
void qrdecomp_create_u(double *a, double *u, int m, int i);

/**
 * QR分解におけるベクトルuを更新する．
 *
 * @param[in, out] u ベクトルu. 更新したベクトルを上書きする．
 * @param[in] m ベクトルのサイズ
 * @param[in] entry uの先頭
 */
void qrdecomp_update_u(double *u, int m);

/**
 * QR分解における行列Aを更新する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[in] u ベクトルu
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] i 着目している列のインデックス
 */
void qrdecomp_update_a(double *a, double *u, int m, int n, int i);

/**
 * QR分解の第1列目に対する処理を実行する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[out] q 直交行列Q
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] u ベクトルu
 */
void qrdecomp_firstiter(double *a, double *q, int m, int n, double *u);

/**
 * QR分解の第2列目以降の処理を実行する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[in, out] q 直交行列Q．更新後の行列で上書きする．
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] u ベクトルu
 * @param[in] i 着目している列のインデックス
 */
void qrdecomp_iter(double *a, double *q, int m, int n, double *u, int i);

/**
 * QR分解の反復処理を直交行列Qを構成せずに実行する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] u ベクトルu
 * @param[in] i 着目している列のインデックス
 */
void qrdecomp_iter_noq(double *a, int m, int n, double *u, int i);

/**
 * 行列のQR分解をする．
 *
 * @param[in, out] a 行列A．上三角行列Rを上書きして返す．
 * @param[out] q 直交行列Q
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void qrdecomp(double *a, double *q, int m, int n);

/**
 * 行列のQR分解を直交行列Qを構成せずに行う．
 *
 * @param[in, out] a 行列A．上三角行列Rを上書きして返す．
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void qrdecomp(double *a, int m, int n);

/**
 * 与えたベクトルを更新する場合のQR分解の各列の処理を実行する．
 *
 * @param[in, out] a QR分解をする行列A．更新後の行列で上書きする．
 * @param[in, out] b ベクトルb. A=QR に対する Q^Tb で上書きして返す．
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @param[in] u ベクトルu
 */
void qrdecompb_iter(double *a, double *b, int m, int n, double *u, int i);

/**
 * 行列AのQR分解により，A=QR となる上三角行列Rを求め，ベクトルbに対してQ^Tbを返す．
 *
 * @param[in, out] a 行列a．上三角行列rを上書きして返す．
 * @param[in, out] b ベクトルb. A=QR に対する Q^Tb で上書きして返す．
 * @param[in] m 行列aの行数
 * @param[in] n 行列aの列数
 */
void qrdecompb(double *a, double *b, int m, int n);

/**
 * 行列積を返す．
 *
 * @param[in] a 行列A
 * @param[in] b 行列B
 * @param[out] c A, Bの積
 * @param[in] l 行列Aの行数
 * @param[in] m 行列Aの列数
 * @param[in] n 行列Bの列数
 */
void mmul(double *a, double *b, double *c, int l, int m, int n);

/**
 * 行列ベクトル積を返す．
 *
 * @param[in] a 行列A
 * @param[in] v ベクトルv
 * @param[out] b A, vの積
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void mmul(double *a, double *v, double *b, int m, int n);

/**
 * ベクトルをコピーする．
 *
 * @param[in] a コピー元ベクトル
 * @param[out] b コピー先ベクトル
 * @param[in] m ベクトルの大きさ
 */
void copy(double *a, double *b, int m);

/**
 * 行列をコピーする．
 *
 * @param[in] a コピー元行列A
 * @param[out] b コピー先行列B
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void copy(double *a, double *b, int m, int n);

/**
 * 行列を表示する．
 *
 * @param[in] a 行列
 * @param[in] m 行数
 * @param[in] n 列数
 */
void print(double *a, int m, int n);

/**
 * ベクトルを表示する．
 *
 * @param[in] v ベクトル
 * @param[in] m 要素数
 */
void print(double *v, int m);

/**
 * 行列を転置する．
 *
 * @param[in] a 行列A
 * @param[out] at 行列Aの転置行列
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 */
void transpose(double *a, double *at, int m, int n);

/**
 * ベクトルの符号を反転して返す．
 *
 * @param[in, out] b ベクトル．この符号を反転したベクトルを上書きして返す．
 * @param[in] m ベクトルの大きさ
 * @return なし．
 */
void minus(double *b, int m);

/**
 * ベクトルをスカラー倍して返す
 *
 * @param[in, out] b ベクトル．スカラー倍したベクトルを上書きして返す．
 * @param[in] a スカラー
 * @param[in] m ベクトルの大きさ
 * @return なし．
 */
void times(double *b, double a, int m);

/**
 * 行列の和を返す．
 *
 * @param a 行列A
 * @param b 行列B
 * @param c 行列 A+B
 * @param m 行列Aの行数
 * @param n 行列Bの列数
 */
void add(double *a, double *b, double *c, int m, int n);

/**
 * 行列の差を返す．
 *
 * @param a 行列A
 * @param b 行列B
 * @param c 行列 A-B
 * @param m 行列Aの行数
 * @param n 行列Aの列数
 */
void sub(double *a, double *b, double *c, int m, int n);

/**
 * ベクトルの和を返す．
 *
 * @param u ベクトルu
 * @param v ベクトルv
 * @param w ベクトル u+v
 * @param m ベクトルuの要素数
 */
void add(double *u, double *v, double *w, int m);

/**
 * ベクトルの差を返す．
 *
 * @param u ベクトルu
 * @param v ベクトルv
 * @param w ベクトル u-v
 * @param m ベクトルuの要素数
 */
void sub(double *u, double *v, double *w, int m);

/**
 * 行階段形である一般の行列のランクを返す．
 *
 * @param[in] r 行階段形である一般の行列R
 * @param[in] m 行列Rの行数
 * @param[in] n 行列Rの列数
 * @return 方程式が解をもつときtrue
 */
int rank_r(double *r, int m, int n);

/**
 * 一般の行列のランクを返す．
 *
 * @param[in] a 行階段形である一般の行列A
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @return 方程式が解をもつときtrue
 */
int rank(double *a, int m, int n);

/**
 * 行階段形である一般の行列を係数行列とした方程式が解をもつ場合trueを返す．
 *
 * @param[in] r 行階段形である一般の行列R
 * @param[in] b 右辺ベクトルb
 * @param[in] m 行列Rの行数
 * @param[in] n 行列Rの列数
 * @return 方程式が解をもつときtrue
 */
bool has_solution_r(double *r, double *b, int m, int n);

/**
 * 一般の行列を係数行列とした方程式が解をもつ場合trueを返す．
 *
 * @param[in] a 一般の行列A
 * @param[in] b 右辺ベクトルb
 * @param[in] m 行列Rの行数
 * @param[in] n 行列Rの列数
 * @return 方程式が解をもつときtrue
 */
bool has_solution(double *a, double *b, int m, int n);

/**
 * 行階段形である一般の行列を係数行列とした方程式の解の一つを返す．
 *
 * @param[in] r 行階段形である一般の行列R
 * @param[in] b 右辺ベクトルb
 * @param[out] x Rx=bの解のひとつ
 * @param[in] m 行列Rの行数
 * @param[in] n 行列Rの列数
 * @return 方程式が解をもつときtrue
 */
bool general_solve_r(double *r, double *b, double *x, int m, int n);

/**
 * 一般の行列を係数行列とした方程式の解の一つを返す．
 *
 * @param[in] A 一般の行列A
 * @param[in] b 右辺ベクトルb
 * @param[out] x Ax=bの解のひとつ
 * @param[in] m 行列Aの行数
 * @param[in] n 行列Aの列数
 * @return 方程式が解をもつときtrue
 */
bool general_solve(double *a, double *b, double *x, int m, int n);

} // namespace matutil

#endif // #ifndef OPENMP_SAMPLE_MATUTIL_HPP_
