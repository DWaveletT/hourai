#include<bits/stdc++.h>
using namespace std;

const int MAXN = 2e7 + 3;
unsigned A[MAXN];

int p, P[MAXN]; bool V[MAXN];
void solve(int n){
    for(int i = 2;i <= n;++ i){
        if(!V[i]){
            P[++ p] = i;
            for(int j = 1;j <= n / i;++ j){ // 前缀和
                A[j * i] += A[j];
            }
        }
        for(int j = 1;j <= p && P[j] <= n / i;++ j){
            V[i * P[j]] = true;
            if(i % P[j] == 0)
                break;
        }
    }
    
}

unsigned seed;
inline unsigned read(){
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}


int main(){
    int n;
    cin >> n >> seed;
    for(int i = 1;i <= n;++ i){
        A[i] = read();
    }
    solve(n);
    unsigned ans = 0;
    for(int i = 1;i <= n;++ i){
        ans ^= A[i];
    }
    cout << ans << endl;

    return 0;
}