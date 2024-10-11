/* <guide> **

** <guide> */
#include<bits/stdc++.h>
using namespace std;

using u64 = unsigned long long;
const int MAXN = 1e6 + 3;

int power(int a, int b, int p){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % p;
        b >>= 1,  a = 1ll * a * a % p;
    }
    return r;
}

namespace ODT {
    // <pos_type, value_type>
    map <int, long long> M;

    // 分裂为 [1, p) 和 [p, +inf)，返回后者迭代器
    auto split(int p) {
        auto it = prev(M.upper_bound(p));
        return M.insert(
            it,
            make_pair(p, it -> second)
        );
    }

    // 区间赋值
    void assign(int l, int r, int v) {
        auto it = split(l);
        split(r + 1);
        while (it -> first != r + 1) {
            it = M.erase(it);
        }
        M[l] = v;
    }

    // // 执行操作
    // void perform(int l, int r) {
    //     auto it = split(l);
    //     split(r + 1);
    //     while (it -> first != r + 1) {
    //         // Do something...
    //         it = next(it);
    //     }
    // }

    void modify1(int l, int r, int w) {
        auto it = split(l);
        split(r + 1);
        while(it -> first != r + 1) {
            it -> second += w;
            it = next(it);
        }
    }
    void modify2(int l, int r, int w) {
        assign(l, r, w);
    }
    long long query1(int l, int r, int k) {
        auto it = split(l);
        split(r + 1);

        map <long long, int> T;
        while(it -> first != r + 1) {
            T[it -> second] += next(it) -> first - it -> first;
            it = next(it);
        }
        for(auto &[w, c]: T){
            if(c >= k)
                return w;
            k -= c;
        }
        return -1;
    }
    long long query2(int l, int r, int x, int y) {
        auto it = split(l);
        split(r + 1);

        int ans = 0;
        while(it -> first != r + 1) {
            int c = next(it) -> first - it -> first;
            long long a = it -> second;
            ans = (ans + 1ll * c * power(a % y, x, y)) % y;
            it = next(it);
        }
        return ans;
    }

};

const int MOD = 1e9 + 7;

int read(int &seed){
    int ret = seed;
    seed = (seed * 7ll + 13) % MOD;
    return ret;
}

int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, seed, vmax;
    cin >> n >> m >> seed >> vmax;

    ODT :: M[n + 1] = 0;
    for(int i = 1;i <= n;++ i){
        int a = read(seed) % vmax + 1;
        ODT :: M[i] = a;
    }
    for(int i = 1;i <= m;++ i){
        int op = read(seed) % 4 + 1;
        int l = read(seed) % n + 1;
        int r = read(seed) % n + 1;
        int x, y;
        if(l > r)
            swap(l, r);

        if(op == 3){
            x = (read(seed) % (r - l + 1)) + 1;
        } else 
            x = read(seed) % vmax + 1;
        
        if(op == 4)
            y = read(seed) % vmax + 1;
        

        if(op == 1){
            ODT :: modify1(l, r, x);
        } else 
        if(op == 2){
            ODT :: modify2(l, r, x);
        } else 
        if(op == 3){
            cout << ODT :: query1(l, r, x) << "\n";
        } else 
        if(op == 4){
            cout << ODT :: query2(l, r, x, y) << "\n";
        }
    }
    return 0;
}