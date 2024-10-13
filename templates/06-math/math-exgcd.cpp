/**
## 内容

给定 $a, b$，求出 $ax+by=\gcd(a, b)$ 的一组 $x, y$。
**/
int exgcd(int a, int b, int &x, int &y){
    if(a == 0){
        x = 0, y = 1; return b;
    } else {
        int x0 = 0, y0 = 0;
        int d = exgcd(b % a, a, x0, y0);
        x = y0 - (b / a) * x0;
        y = x0;
        return d;
    }
}
