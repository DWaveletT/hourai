/**
## 用法

调用 init 计算出 $S$ 和 $X$，得到计算 $\lfloor x / P \rfloor = (x\times X) / 2^{60 + S}$。从而计算 $x \bmod P = x - P \times \lfloor x / P \rfloor$。
**/
#include "../header.cpp"
i64 S = 0, X = 0;
void init(int MOD){
  while((1 << (S + 1)) < MOD) S ++;
  X = ((i80)1 << 60 + S) / MOD + !!(((i80)1 << 60 + S) % MOD);
  cerr << S << " " << X << endl;
}
int power(i64 x, int y, int MOD){
  i64 r = 1;
  while(y){
    if(y & 1){
      r = r * x;
      r = r - MOD * ((i80)r * X >> 60 + S);
    }
    x = x * x;
    x = x - MOD * ((i80)x * X >> 60 + S);
    y >>= 1;
  }
  return r;
}