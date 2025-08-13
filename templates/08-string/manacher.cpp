#include "../header.cpp"
char S[MAXN], T[MAXN]; int n, R[MAXN];
int main(){
  scanf("%s", S + 1);
  n = strlen(S + 1);
  for(int i = 1;i <= n;++ i){
    T[2 * i - 1] = S[i], T[2 * i] = '#';
  }
  T[0] = '#', n = 2 * n;
  int p = 0, x = 0, ans = 0;
  for(int i = 1;i <= n;++ i){
    if(i <= p)R[i] = min(R[2 * x - i], p - i);
    while(i - R[i] - 1 >= 0
      && T[i + R[i] + 1] == T[i - R[i] - 1])
      ++ R[i];
    if(i + R[i] > p){
      p = i + R[i];
      x = i;
    }
    ans = max(ans, R[i]);
  }
  printf("%d\n", ans);
  return 0;
}