#include "../header.cpp"
vector <int> G[MAXN];
bool V[MAXN];
int ML[MAXN], MR[MAXN];

bool kuhn(int u){
    V[u] = true;
    for(auto &v: G[u]) if(MR[v] == 0){
        ML[u] = v, MR[v] = u;
        return true;
    }
    for(auto &v: G[u]) if(!V[MR[v]] && kuhn(MR[v])){
        ML[u] = v, MR[v] = u;
        return true;
    }
    return false;
}
void solve(int L, int R){
    mt19937 MT;     // 需要打乱避免构造
    for(int i = 1;i <= L;++ i){
        shuffle(G[i].begin(), G[i].end(), MT);
    }
    while(1){
        bool ok = false;
        memset(V, false, sizeof(V));
        for(int i = 1;i <= L;++ i)
            ok |= !ML[i] && kuhn(i);
        if(!ok) break;
    }
}