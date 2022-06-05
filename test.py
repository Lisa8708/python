# -*- encoding:utf-8 -*-

import sys
print(sys.version)

# 快排
def sort(lst):
    left, right = 0, len(lst)-1
    while left < len(lst):
        while left < right:
            if lst[right] > lst[left]:
                right -= 1
            elif lst[right] < lst[left]:
                lst[left], lst[right] = lst[right], lst[left]
        right = len(lst)-1
        left += 1
    return lst

print(sort([1,3,2,4,5]))

# 归并
def merge(l1, l2):
    res = []
    i, j = 0, 0
    while i<len(l1) and j<len(l2):
        if l1[i] < l2[j]:
            res.append(l1[i])
            i+=1
        else:
            res.append(l2[j])
            j+=1
    if i<len(l1):
        res += l1[i:]
    if j<len(l2):
        res += l2[j:]
    return res

def sort2(lst):
    if len(lst) <= 1:
        return lst
    n = len(lst)//2
    left = sort2(lst[:n])
    right = sort2(lst[n:])
    return merge(left, right)

print(sort2([1,3,2,4,5]))
#print(sort2([1,3,2,4,5,2]))
