```cpp
#include "../../header.cpp"
struct Mat{
  int n, m;
  int W[MAXN][MAXN];
  Mat(int _n = 0, int _m = 0){
    n = _n;
    m = _m;
    for(int i = 1;i <= n;++ i)
      for(int j = 1;j <= m;++ j)
        W[i][j] = 0;
  }
};
int power(int a, int b){
  int r = 1;
  while(b){
    if(b & 1) r = 1ll * r * a % MOD;
    b >>= 1,  a = 1ll * a * a % MOD;
  }
  return r;
}
int inv(int x){
  return power(x, MOD - 2);
}
int mat_rank(Mat &a){
  const int &n = a.n;
  const int &m = a.m;
  int cnt = 0;
  for(int i = 1;i <= m;++ i){
    int p = cnt + 1;
    int f = -1;
    for(int j = p;j <= n;++ j){
      if(a.W[j][i] != 0){
        f = j;
        break;
      }
    }
    if(f == -1)
      continue;
    if(f != p){
      for(int j = 1;j <= m;++ j)
        swap(a.W[p][j], a.W[f][j]);
    }
    ++ cnt;
    int invp = inv(a.W[p][i]);
    for(int j = p + 1;j <= n;++ j){
      int rate = 1ll * a.W[j][i] * invp % MOD;
      for(int k = 1;k <= m;++ k){
        a.W[j][k] = (a.W[j][k] - 1ll * rate * a.W[p][k] % MOD + MOD) % MOD;
      }
    }
  }
  return cnt;
}
int X[MAXN];
int main(){
  int n;
  cin >> n;
  Mat A(n, n);
  Mat T(n, n + 1);
  for(int i = 1;i <= n;++ i){
    for(int j = 1;j <= n;++ j)
      cin >> A.W[i][j];
    for(int j = 1;j <= n;++ j)
      T.W[i][j] = A.W[i][j];
    cin >> T.W[i][n + 1];
  }
  int res1 = mat_rank(A);
  int res2 = mat_rank(T);
  if(res1 != res2)
    cout << -1 << endl;
  else
  if(res2 <  n)
    cout << 0 << endl;
  else {
    for(int i = n;i >= 1;-- i){
      int invp = inv(T.W[i][i]);
      X[i] = 1ll * T.W[i][n + 1] * invp % MOD;
      for(int j = i - 1;j >= 1;-- j){
        int rate = 1ll * T.W[j][i] * invp % MOD;
        T.W[j][  i] = (T.W[j][  i] - 1ll * rate * T.W[i][  i] % MOD + MOD) % MOD;
        T.W[j][n + 1] = (T.W[j][n + 1] - 1ll * rate * T.W[i][n + 1] % MOD + MOD) % MOD;
      }
    }
    for(int i = 1;i <= n;++ i)
      cout << "x" << i << "=" << X[i] << endl;
  }
  return 0;
}
```
