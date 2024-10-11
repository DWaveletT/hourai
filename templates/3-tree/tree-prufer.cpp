#include<bits/stdc++.h>
using namespace std;

const int MAXN = 5e6 + 3;
int D[MAXN], F[MAXN], P[MAXN];
vector<int> tree2prufer(int n){
    vector <int> P(n);
    for(int i = 1, j = 1;i <= n - 2;++ i, ++ j){
        while(D[j]) ++ j;
        P[i] = F[j];
        while(i <= n - 2 && !--D[P[i]] && P[i] < j)
            P[i + 1] = F[P[i]], i ++;
    }
    return P;
}
vector<int> prufer2tree(int n){
    vector <int> F(n);
    for(int i = 1, j = 1;i <= n - 1;++ i, ++ j){
        while(D[j]) ++ j;
        F[j] = P[i];
        while(i <= n - 1 && !--D[P[i]] && P[i] < j)
            F[P[i]] = P[i + 1], i ++;
    }
    return F;
}

int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector <int> ANS;
    if(m == 1){     // tree -> prufer
        for(int i = 1;i <= n - 1;++ i){
            cin >> F[i], D[F[i]] ++;
        }
        ANS = tree2prufer(n);
    } else {        // prufer -> tree
        for(int i = 1;i <= n - 2;++ i){
            cin >> P[i], D[P[i]] ++;
        }
        P[n - 1] = n;
        ANS = prufer2tree(n);
    }
    long long ans = 0, cnt = 0;
    for(int i = 1;i <= n - (m == 1 ? 2 : 1);++ i)
        ans ^= 1ll * ANS[i] * i;
    cout << ans << "\n";
    return 0;
}