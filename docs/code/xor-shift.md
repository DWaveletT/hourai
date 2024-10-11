```cpp
#include<bits/stdc++.h>
using namespace std;

using u32 = uint32_t;
using u64 = uint64_t;

u32 xorshift32(u32 &x){
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}
u64 xorshift64(u64 &x){
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}
```
