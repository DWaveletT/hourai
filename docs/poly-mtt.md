```cpp
#include "poly-family.cpp"
const int BLOCK = 32768;
using cplx = complex<double>;
cplx A1[MAXN], A2[MAXN], B1[MAXN], B2[MAXN];
int n, m, L, mod;
cplx P[MAXN], Q[MAXN];
void FFTFFT(int L, cplx X[], cplx Y[]){
  for(int i = 0;i < L;++ i){
    P[i] = { X[i].real(), Y[i].imag() };
  }
  Poly :: FFT(L, P);
  for(int i = 0;i < L;++ i){
    Q[i] = (i == 0 ? P[0] : P[L - i]);
    Q[i].imag(-Q[i].imag());
  }
  for(int i = 0;i < L;++ i){
    X[i] = (P[i] + Q[i]);
    Y[i] = (Q[i] - P[i]) * cplx(0, 1);
    X[i] /= 2, Y[i] /= 2;
  }
}
int main(){
  ios :: sync_with_stdio(false);
  cin.tie(nullptr);
  cin >> n >> m >> mod;
  for(int i = 0;i <= n;++ i){
    int a; cin >> a; a %= mod;
    A1[i].real(a / BLOCK);
    A2[i].imag(a % BLOCK);
  }
  for(int i = 0;i <= m;++ i){
    int a; cin >> a; a %= mod;
    B1[i].real(a / BLOCK);
    B2[i].imag(a % BLOCK);
  }
  for(L = 1;L <= n + m;L <<= 1);
  FFTFFT(L, A1, A2), FFTFFT(L, B1, B2);
  for(int i = 0;i < L;++ i){
    P[i] = A1[i] * B1[i] + cplx(0, 1) * A2[i] * B1[i];
    Q[i] = A1[i] * B2[i] + cplx(0, 1) * A2[i] * B2[i];
  }
  Poly :: IFFT(L, P);
  Poly :: IFFT(L, Q);
  for(int i = 0;i < L;++ i){
    long long a1b1 = P[i].real() + 0.5;
    long long a2b1 = P[i].imag() + 0.5;
    long long a1b2 = Q[i].real() + 0.5;
    long long a2b2 = Q[i].imag() + 0.5;
    long long w = ((a1b1 % mod * (BLOCK * BLOCK % mod)) + ((a2b1 + a1b2) % mod) * BLOCK + a2b2) % mod;
    if(i <= n + m) cout << w << " ";
  }
  return 0;
}
```
