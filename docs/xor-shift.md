```cpp
#include "../header.cpp"
u32 xorshift32(u32 &x){
  x ^= x << 13, x ^= x >> 17, x ^= x << 5;
  return x;
}
u64 xorshift64(u64 &x){
  x ^= x << 13, x ^= x >> 7, x ^= x << 17;
  return x;
}
```
