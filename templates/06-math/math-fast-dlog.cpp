/**
## 用法

给定原根 $g$ 以及模数 $\mathrm{M}$，$T$ 次询问 $x$ 的离散对数。

复杂度 $\mathcal O(\mathrm{M}^{2/3} + T \log \mathrm{M})$。
**/
#include "../header.cpp"
namespace BSGS {
  unordered_map <int, int> M;
  int B, U, P, g;
  void init(int g, int P0, int B0);
  int solve(int y);
}
const int MAXN = 1e5 + 3;
int H[MAXN], P[MAXN], H0, p, h, g, M;
bool V[MAXN];
int solve(int x){
  if(x <= h) return H[x];
  int v = M / x, r = M % x;
  if(r < x - r) return ((H0 + solve(r)) % (M - 1) - H[v] + M - 1) % (M - 1);
  else          return (solve(x - r) - H[v + 1] + M - 1) % (M - 1);
}
int main(){
  ios :: sync_with_stdio(false);
  cin.tie(nullptr);
  cin >> g >> M;
  h = sqrt(M) + 1;
  BSGS :: init(g, M, sqrt(1ll * M * sqrt(M) / log10(M)));
  H0 = BSGS :: solve(M - 1);
  H[1] = 0;
  for(int i = 2;i <= h;++ i){
    if(!V[i]){
      P[++ p] = i;
      H[i] = BSGS :: solve(i);
    }
    for(int j = 1;j <= p&&P[j] <= h / i;++ j){
      int &p = P[j];
      H[i * p] = (H[i] + H[p]) % (M - 1);
      V[i * p] = true;
      if(i % p == 0) break;
    }
  }
  int T; cin >> T;
  while(T --){
    int x; cin >> x;
    cout << solve(x) << "\n";
  }
  return 0;
}