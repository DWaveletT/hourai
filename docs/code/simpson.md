## 例题

计算

$$\int_{0}^{+\infty} x^{(a/x) - x}$$

```cpp
#include<bits/stdc++.h>
using namespace std;
using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;
double simpson(double (*f)(double), double l, double r){
    double mid = (l + r) / 2;
    return (r - l) * (f(l) + 4 * f(mid) + f(r)) / 6.0;
}
double adapt_simpson(double (*f)(double), double l, double r, double EPS, int step){
    double mid = (l + r) / 2;
    double w0 = simpson(f, l, r);
    double w1 = simpson(f, l, mid);
    double w2 = simpson(f, mid, r);
    if(fabs(w0 - w1 - w2) < EPS && step < 0)
        return w1 + w2;
    else
        return adapt_simpson(f, l, mid, EPS, step - 1) + 
               adapt_simpson(f, mid, r, EPS, step - 1);
}
double a, l, r;
double fun(double x){
    return pow(x, a / x - x);
}
int main(){
    cin >> a;
    if(a < 0)
        cout << "orz" << endl;
    else {
        l = 1e-9;
        r = 150;
        cout << fixed << setprecision(5) << adapt_simpson(fun, l, r, 1e-9, 15);
    }
}
```
