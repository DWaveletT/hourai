#include "../header.cpp"

int getphi(int x);              // 求解 phi
vector <int> getprime(int x);   // 求解质因数
bool test(int g, int m, int mm,vector<int>&P){
  for(auto &p: P)
    if(power(g, mm / p, m) == 1)
      return false;
  return true;
}
int get_genshin(int m){
  int mm = getphi(m);
  vector <int> P = getprime(mm);
  for(int i = 1;;++ i)
    if(test(i, m, mm, P)) return i;
}
