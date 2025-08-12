/**
## 定理

对于线性方程：

$$
\begin{cases}
x \equiv a_1 \pmod {m_1} \\
x \equiv a_2 \pmod {m_2} \\
\cdots \\
x \equiv a_n \pmod {m_n} \\
\end{cases}
$$

如果 $a_i$ 两两互质，可以得到 $x$ 的解 $x\equiv L\pmod M$，其中 $M=\prod m_i$，而 $L$ 由下式给出：$$L = \left(\sum a_i m_i\times (\left(M/m_i\right)^{-1}\bmod m_i)\right)\bmod M$$
**/
#include "../header.cpp"
i64 A[MAXN], B[MAXN], M = 1;
i64 exgcd(i64 a, i64 b, i64 &x, i64 &y);
int main(){
  int n; cin >> n;
  for(int i = 1;i <= n;++ i){
    cin >> B[i] >> A[i];
    M = M * B[i];
  }
  i64 L = 0;
  for(int i = 1;i <= n;++ i){
    i64 m = M / B[i], b, k;
    exgcd(m, B[i], b, k);
    L = (L + (__int128)A[i] * m * b) % M;
  }
  L = (L % M + M) % M;
  cout << L << endl;
  return 0;
}