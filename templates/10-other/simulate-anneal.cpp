#include<bits/stdc++.h>
using namespace std;

const double T0 = 2e3, Tk = 1e-14, delta = 0.993, R = 1e-3;
mt19937 MT(114514);
double distance(double x, double y, double a, double b){
    return sqrt(pow(a - x, 2) + pow(b - y, 2));
}
const int MAXN = 1e3 + 3;
double X[MAXN], Y[MAXN], W[MAXN]; int n;
double calculate(double x, double y){
    double gx, gy, a;
    for(int i = 0;i < n; ++i){
        a = atan2(y - Y[i], x - X[i]);
        gx += cos(a) * W[i];
        gy += sin(a) * W[i];
    }
    return pow(gx, 2) + pow(gy, 2);
}
double ex, ey, eans = 1e18;
void SA(){
    double T = T0, x = 0, y = 0, ans = calculate(x, y);
    double ansx, ansy;
    uniform_real_distribution<double> U;
    while(T > Tk){
        double nx, ny, nans;
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
int main(){
    cin >> n;
    for(int i = 0;i < n;++ i)
        cin >> X[i] >> Y[i] >> W[i];
    cout << fixed << setprecision(3);
    if(n == 1){
        cout << X[0] << " " << Y[0] << endl;
    } else {
        SA(), SA(), SA();
        cout << ex << " " << ey << endl;
    }
    return 0;
}