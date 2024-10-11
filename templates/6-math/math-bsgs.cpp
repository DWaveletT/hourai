#include<bits/stdc++.h>
using namespace std;

int power(int a, int b, int  p){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % p;
        b >>= 1,  a = 1ll * a * a % p;
    }
    return r;
}
namespace BSGS {
    unordered_map <int, int> M;
    int solve(int a, int y, int p){    // a ^ x = y (mod p)
        M.clear();
        int B = sqrt(p);
        int w1 = y, u1 = power(a, p - 2, p);
        int w2 = 1, u2 = power(a, B, p);
        for(int i = 0;i < B;++ i){
            M[w1] = i;
            w1 = 1ll * w1 * u1 % p;
        }
        for(int i = 0;i < p / B;++ i){
            if(M.count(w2)){
                return i * B + M[w2];
            }
            w2 = 1ll * w2 * u2 % p;
        }
        return -1;
    }
}

int main(){
    int p, b, n;
    cin >> p >> b >> n;

    int ans = BSGS :: solve(b, n, p);
    if(ans == -1){
        cout << "no solution\n";
    } else {
        cout << ans << "\n";
    }
    return 0;
}