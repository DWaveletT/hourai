#include "../../header.cpp"
const int MAXN = 1e5 + 3;
int X1[MAXN], Y1[MAXN];
int X2[MAXN], Y2[MAXN];
int n, h, H[MAXN * 2];
namespace Seg{
  #define lc(t) (t << 1)
  #define rc(t) (t << 1 | 1)
  const int SIZ = 8e5 + 3;
  int T[SIZ], S[SIZ], L[SIZ];
  void pushup(int t, int a, int b){
    S[t] = 0;
    if(a != b){
      S[t] = S[lc(t)] + S[rc(t)];
      L[t] = L[lc(t)] + L[rc(t)];
    }
    if(T[t]) S[t] = L[t];
  }
  void modify(int t, int a, int b, int l, int r, int w){
    if(l <= a && b <= r){
      T[t] += w, pushup(t, a, b);
    } else {
      int c = a + b >> 1;
      if(l <= c) modify(lc(t), a, c, l, r, w);
      if(r >  c) modify(rc(t), c + 1, b, l, r, w);
      pushup(t, a, b);
    }
  }
  void build(int t, int a, int b){
    if(a == b){
      L[t] = H[a] - H[a - 1];
    } else {
      int c = a + b >> 1;
      build(lc(t), a, c);
      build(rc(t), c + 1, b);
      pushup(t, a, b);
    }
  }
  int query(int t){
    return S[t];
  }
}
tuple <int, int, int> P[MAXN], Q[MAXN];
int main(){
  n = qread();
  for(int i = 1;i <= n;++ i){
    X1[i] = qread(), Y1[i] = qread();
    X2[i] = qread(), Y2[i] = qread();
    if(X1[i] > X2[i]) swap(X1[i], X2[i]);
    if(Y1[i] > Y2[i]) swap(Y1[i], Y2[i]);
    H[++ h] = Y1[i];
    H[++ h] = Y2[i];
    P[i] = make_tuple(X1[i], Y1[i], Y2[i]);
    Q[i] = make_tuple(X2[i], Y1[i], Y2[i]);
  }
  sort(H + 1, H + 1 + h);
  sort(P + 1, P + 1 + n);
  sort(Q + 1, Q + 1 + n);
  int o = unique(H + 1, H + 1 + h) - H - 1;
  Seg :: build(1, 1, o);
  i64 ans = 0, last = -1;
  int p = 1, q = 1;
  while(p <= n || q <= n){
    int x = INF;
    if(p <= n) x = min(x, get<0>(P[p]));
    if(q <= n) x = min(x, get<0>(Q[q]));

    if(last != -1){
      ans += 1ll * Seg :: query(1) * (x - last);
    }
    last = x;

    while(q <= n && get<0>(Q[q]) == x){
      auto [x, l, r] = Q[q]; ++ q;
      l = lower_bound(H + 1, H + 1 + o, l) - H + 1;
      r = lower_bound(H + 1, H + 1 + o, r) - H;
      Seg :: modify(1, 1, o, l, r, 1);
    }
    while(p <= n && get<0>(P[p]) == x){
      auto [x, l, r] = P[p]; ++ p;
      l = lower_bound(H + 1, H + 1 + o, l) - H + 1;
      r = lower_bound(H + 1, H + 1 + o, r) - H;
      Seg :: modify(1, 1, o, l, r, -1);
    }
  }
  printf("%lld\n", ans);
  return 0;
}