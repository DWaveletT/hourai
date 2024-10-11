```cpp
#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

namespace BLOCK{
    const int SIZ = 1e6 + 1e5 + 3;
    const int BSZ = 2000;

    list <vector<int> > block;

    void build(int n, const int A[]){
        for(int l = 0, r = 0;r != n;){
            l = r;
            r = min(l + BSZ / 2, n);
            vector <int> V0(A + l, A + r);

            block.emplace_back(V0);
        }
    }
    int get_kth(int k){
        for(auto it = block.begin();it != block.end();++ it){
            if(it -> size() < k)
                k -= it -> size();
            else return it -> at(k - 1);
        }
        return -1;
    }
    int get_rank(int w){
        int ans = 0;
        for(auto it = block.begin();it != block.end();++ it){
            if(it -> back() < w)
                ans += it -> size();
            else {
                ans += lower_bound(it -> begin(), it -> end(), w) - it -> begin();
                break;
            }
        }
        return ans + 1;
    }
    // 插入到第 k 个位置
    void insert(int k, int w){
        for(auto it = block.begin();it != block.end();++ it){
            if(it -> size() < k)
                k -= it -> size();
            else{
                it -> insert(it -> begin() + k - 1, w);
                if(it -> size() > BSZ){
                    vector <int> V1(it -> begin(), it -> begin() + BSZ / 2);
                    vector <int> V2(it -> begin() + BSZ / 2, it -> end());

                    *it = V2;
                    block.insert(it, V1);
                }
                return;
            }
        }
    }
    // 删除第 k 个数
    void erase(int k){
        for(auto it = block.begin();it != block.end();++ it){
            if(it -> size() < k)
                k -= it -> size();
            else{
                it -> erase(it -> begin() + k - 1);
                if(it -> empty())
                    block.erase(it);
                return;
            }
        }
    }
}

int qread();

const int MAXN = 1e5 + 3;
int A[MAXN];

// ===== TEST =====

int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    
    cin >> n >> m;
    for(int i = 1;i <= n;++ i)
        cin >> A[i];
    sort(A + 1, A + 1 + n);
    A[n + 1] = INT_MAX;

    BLOCK :: build(n + 1, A + 1);

    int last = 0;
    int ans = 0;

    for(int i = 1;i <= m;++ i){
        int op;
        cin >> op;
        if(op == 1){
            int x; cin >> x; x ^= last;
            int k = BLOCK :: get_rank(x);
            BLOCK :: insert(k, x);
        } else 
        if(op == 2){
            int x; cin >> x; x ^= last;
            int k = BLOCK :: get_rank(x);
            BLOCK :: erase(k);
        } else 
        if(op == 3){
            int x; cin >> x; x ^= last;
            int k = BLOCK :: get_rank(x);
            last = k, ans ^= last;
        } else 
        if(op == 4){
            int x; cin >> x; x ^= last;
            int k = BLOCK :: get_kth (x);
            last = k, ans ^= last;
        } else 
        if(op == 5){
            int x; cin >> x; x ^= last;
            int k = BLOCK :: get_rank(x);
            last = BLOCK :: get_kth (k - 1), ans ^= last;
        } else 
        if(op == 6){
            int x; cin >> x; x ^= last;
            int k = BLOCK :: get_rank(x + 1);
            last = BLOCK :: get_kth (k), ans ^= last;
        }
    }
    cout << ans << endl;
    return 0;
}
```
