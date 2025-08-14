/**
## 例题

给定 $n$ 个物品挂在洞下，第 $i$ 个物品坐标 $(x_i, y_i)$ 重量为 $w_i$。询问平衡点。
**/
#include "../header.cpp"
const db T0 = 2e3, Tk = 1e-14, delta = 0.993, R = 1e-3;
mt19937 MT(114514);
db distance(db x, db y, db a, db b){
  return sqrt(pow(a - x, 2) + pow(b - y, 2));
}
const int MAXN = 1e3 + 3;
db X[MAXN], Y[MAXN], W[MAXN]; int n;
db calculate(db x, db y){
  db gx, gy, a;
  for(int i = 0;i < n; ++i){
    a = atan2(y - Y[i], x - X[i]);
    gx += cos(a) * W[i];
    gy += sin(a) * W[i];
  }
  return pow(gx, 2) + pow(gy, 2);
}
db ex, ey, eans = 1e18;
void SA(){
  db T = T0, x = 0, y = 0, ans = calculate(x, y);
  db ansx, ansy;
  uniform_real_distribution<db> U;
  while(T > Tk){
    db nx, ny, nans;
    nx = x + 2 * (U(MT) - .5) * T;
    ny = y + 2 * (U(MT) - .5) * T;
    if((nans = calculate(nx, ny)) < ans){
      ans = nans;
      ansx = x = nx;
      ansy = y = ny;
    } else if(exp(-distance(nx, ny, x, y) / T / R) > U(MT)){
      x = nx, y = ny;
    }
    T *= delta;
  }
  if(ans < eans) eans = ans, ex = ansx, ey = ansy;
}
