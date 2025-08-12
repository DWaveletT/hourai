#include "../header.cpp"
int D[MAXN], F[MAXN], P[MAXN];
vector<int> tree2prufer(int n){
  vector <int> P(n);
  for(int i = 1, j = 1;i <= n - 2;++ i, ++ j){
    while(D[j]) ++ j;
    P[i] = F[j];
    while(i <= n - 2 && !--D[P[i]] && P[i] < j)
      P[i + 1] = F[P[i]], i ++;
  }
  return P;
}
vector<int> prufer2tree(int n){
  vector <int> F(n);
  for(int i = 1, j = 1;i <= n - 1;++ i, ++ j){
    while(D[j]) ++ j;
    F[j] = P[i];
    while(i <= n - 1 && !--D[P[i]] && P[i] < j)
      F[P[i]] = P[i + 1], i ++;
  }
  return F;
}