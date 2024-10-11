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

int getphi(int x){
    int t = x, r = x;
    for(int i = 2;i <= x / i;++ i){
        if(t % i == 0){
            r = r / i * (i - 1);
            while(t % i == 0)
                t /= i;
        }
    }
    if(t != 1){
        r = r / t * (t - 1);
    }
    return r;
}
vector <int> getprime(int x){
    vector <int> p;
    int t = x;
    for(int i = 2;i <= x / i;++ i){
        if(t % i == 0){
            p.push_back(i);
            while(t % i == 0)
                t /= i;
        }
    }
    if(t != 1)
        p.push_back(x);
    return p;
}

// mm: Phi(m)
// P : prime of mm
bool test(int g, int m, int mm, vector<int> &P){
    for(auto &p: P){
        if(power(g, mm / p, m) == 1)
            return false;
    }
    return true;
}

// 获取最小原根
// m 可以不为质数
int get_genshin(int m){
    int mm = getphi(m);
    vector <int> P = getprime(mm);
    for(int i = 1;;++ i){
        if(test(i, m, mm, P))
            return i;
    }
}

int main(){
    cout << get_genshin(998244353) << endl;
    return 0;
}
/*
998244353: 3
1e9 + 7  : 5
*/