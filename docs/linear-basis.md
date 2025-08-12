```cpp
#include "../header.cpp"
namespace LB{
  const int SIZ = 60 + 3;
  i64 W[SIZ], h = 60;
  void insert(i64 w){
    for(int i = h;i >= 0;-- i){
      if(w & (1ll << i)){
        if(!W[i]){
          W[i] = w;
          break;
        } else {
          w ^= W[i];
        }
      }
    }
  }
  i64 query(i64 x){
    for(int i = h;i >= 0;-- i){
      if(W[i]){
        x = max(x, x ^ W[i]);
      }
    }
    return x;
  }
}
namespace realLB{
  const int SIZ = 500 + 3;
  long double W[SIZ][SIZ];
  int n = 0;
  void init(int n0){
    n = n0;
  }
  bool zero(long double w){
    return fabs(w) < 1e-9;
  }
  bool insert(long double X[]){
    for(int i = 1; i <= n;++ i){
      if(!zero(X[i])){
        if(zero(W[i][i])){
          for(int j = 1;j <= n;++ j)
            W[i][j] = X[j];
          return true;
        } else {
          long double t = X[i] / W[i][i];
          for(int j = 1;j <= n;++ j)
            X[j] -= t * W[i][j];
        }
      }
    }
    return false;
  }
}
// ===== TEST =====
int qread();
const int MAXN = 500 + 3;
long double X[MAXN][MAXN], C[MAXN];
int I[MAXN];
bool cmp(int a, int b){
  return C[a] < C[b];
}
int main(){
  int n, m;
  cin >> n >> m;
  realLB :: init(m);
  for(int i = 1;i <= n;++ i){
    for(int j = 1;j <= m;++ j){
      cin >> X[i][j];
    }
  }
  for(int i = 1;i <= n;++ i){
    cin >> C[i];
    I[i] = i;
  }
  sort(I + 1, I + 1 + n, cmp);
  int ans = 0, cnt = 0;
  for(int i = 1;i <= n;++ i){
    int x = I[i];
    if(realLB :: insert(X[x]))
      ans += C[x],
      cnt += 1;
  }
  cout << cnt << " " << ans << endl;
  return 0;
}
```
