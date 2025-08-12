#include "../../header.cpp"
const double EPS = 1e-9;

struct Mat{
  int n, m;
  double W[MAXN][MAXN];
  Mat(int _n = 0, int _m = 0){
    n = _n;
    m = _m;
    for(int i = 1;i <= n;++ i)
      for(int j = 1;j <= m;++ j)
        W[i][j] = 0;
  }
};

bool zero(double f){
  return fabs(f) < EPS;
}

int mat_rank(Mat &a){
  const int &n = a.n;
  const int &m = a.m;
  int cnt = 0;
  for(int i = 1;i <= m;++ i){
    int p = cnt + 1;
    int f = -1;
    for(int j = p;j <= n;++ j){
      if(!zero(a.W[j][i])){
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
    for(int j = p + 1;j <= n;++ j){
      double rate = a.W[j][i] / a.W[p][i];
      for(int k = 1;k <= m;++ k){
        a.W[j][k] -= rate * a.W[p][k];
      }
    }
  }
  return cnt;
}
double X[MAXN];
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
      X[i] = T.W[i][n + 1] / T.W[i][i];
      for(int j = i - 1;j >= 1;-- j){
        double rate = T.W[j][i] / T.W[i][i];
        T.W[j][  i] -= rate * T.W[i][  i];
        T.W[j][n + 1] -= rate * T.W[i][n + 1];
      }
    }
    for(int i = 1;i <= n;++ i)
      cout << "x" << i << "=" << fixed << setprecision(2) << X[i] << endl;
  }
  return 0;
}