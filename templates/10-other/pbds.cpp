#include "../header.cpp"
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>    // 树
#include <ext/pb_ds/priority_queue.hpp> // 堆
using namespace __gnu_pbds;

// insert, erase, order_of_key, find_by_order,
// [lower|upper]_bound, join, split, size
__gnu_pbds::tree<int, null_type, less<int>,
    rb_tree_tag,
    tree_order_statistics_node_update> T;

// push, pop, top, size, empty, modify, erase,
// join 其中 modify 修改寄存器，join 合并堆
// 还可以用 rc_binomial_heap_tag
__gnu_pbds::priority_queue<int, less<int>,
    pairing_heap_tag> Q1, Q2;
