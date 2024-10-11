```cpp
#include<bits/stdc++.h>
using namespace std;

int power(int a, int b, int p){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % p;
        b >>= 1,  a = 1ll * a * a % p;
    }
    return r;
}

bool check(int x, int p){
    return power(x, (p - 1) / 2, p) == 1;
}

struct Node {
    int real, imag;
};
Node mul(const Node a, const Node b, int p, int v){
    int nreal = (1ll * a.real * b.real + 1ll * a.imag * b.imag % p * v) % p;
    int nimag = (1ll * a.real * b.imag + 1ll * a.imag * b.real) % p;
    return { (nreal), nimag };
}
Node power(Node a, int b, int p, int v){
    Node r = { 1, 0 };
    while(b){
        if(b & 1) r = mul(r, a, p, v);
        b >>= 1,  a = mul(a, a, p, v);
    }
    return r;
}

mt19937 MT;
void solve(int n, int p, int &x1, int &x2){
    if(n == 0){
        x1 = x2 = 0;
        return;
    }
    if(!check(n, p)){
        x1 = x2 = -1;
        return;
    }
    int a, t;
    do {
        a = MT() % p;
    }while(check(t = (1ll * a * a - n + p) % p, p));
    Node u = { a, 1 };
    x1 = power(u, (p + 1) / 2, p, t).real;
    x2 = (p - x1) % p;

    if(x1 > x2)
        swap(x1, x2);
}

int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);
    
    int T;
    cin >> T;
    while(T --){
        int n, p, x1, x2;
        cin >> n >> p;
        solve(n, p, x1, x2);
        if(x1 == -1){
            cout << "Hola!\n";
        } else {
            if(x1 == x2){
                cout << x1 << "\n";
            } else {
                cout << x1 << " " << x2 << "\n";
            }
        }
    }
    return 0;
}
```
