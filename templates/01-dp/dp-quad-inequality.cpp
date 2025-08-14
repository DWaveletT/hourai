#include "../header.cpp"
// 要求对于 $a \leq b \leq c \leq d$ 有 $w(b, c) \leq w(a, d)$ 且 $w(a,c)+w(b,d) \leq w(a,d) + w(b,c)$
int w(int l, int r);
int f[MAXN][MAXN], m[MAXN][MAXN], n;
int solve(){
  for(int len = 2; len <= n; ++len)
    for(int l = 1, r = len;r <= n;++l, ++r){
      f[l][r] = INF;
      for(int k = m[l][r - 1];k <= m[l + 1][r]; ++k){
        int u = f[l][k] + f[k + 1][r]+w(l, r);
        if(f[l][r] > u)
          f[l][r] = u, m[l][r] = k;
      }
    }
}
