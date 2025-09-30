import math
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation

def levenshteindistance(a, b):
    m = len(a)
    n = len(b)
    dp = np.zeros((m+1, n+1), dtype = int)
    for i in range(m+1):
        dp[i, 0] = i
    for j in range(n+1):
        dp[0, j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                cost = 0
            else:
                cost = 1            
            dp[i, j] = min(
                dp[i-1, j] + 1,        
                dp[i, j-1] + 1,        
                dp[i-1, j-1] + cost    
            )
    return dp[m, n]

print(levenshteindistance("kitten", "sitting"))  
print(levenshteindistance("flaw", "lawn"))       

def levenshteinpath(a, b):
    m = len(a) 
    n = len(b)
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m+1):
        dp[i, 0] = i
    for j in range(n+1):
        dp[0, j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,
                dp[i, j-1] + 1,
                dp[i-1, j-1] + cost
            )
    x = []  
    i = m 
    j = n
    while i > 0 or j > 0:
        if i > 0 and dp[i, j] == dp[i-1, j] + 1:
            x.append(("delete", a[i-1], i-1, j-1))
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j-1] + 1:
            x.append(("insert", b[j-1], i-1, j-1))
            j -= 1
        else:
            if i > 0 and j > 0:
                if a[i-1] == b[j-1]:
                    x.append(("match", a[i-1], i-1, j-1))
                else:
                    x.append(("substitute", f"{a[i-1]}→{b[j-1]}", i-1, j-1))
                i -= 1
                j -= 1
            else:
                break
    x.reverse()
    return dp, x

dp, x = levenshteinpath("kitten", "sitting")
print("Distance:", dp[-1, -1])
for o in x:
    print(o)

def backtrack(dp: np.ndarray):
    i = dp.shape[0]-1
    j = dp.shape[1]-1
    x = [(i, j)]
    while i > 0 or j > 0:
        c = []
        if i > 0: c.append((dp[i-1, j], i-1, j))
        if j > 0: c.append((dp[i, j-1], i, j-1))
        if i > 0 and j > 0: c.append((dp[i-1, j-1], i-1, j-1))
        val = dp[i, j]
        prev = min(c, key = lambda x: x[0])
        pv, pi, pj = prev
        if pv + 1 == val or pv == val:
            i, j = pi, pj
        else:
            i, j = pi, pj
        x.append((i, j))
    return x

def showdp(a, b):
    dp, _ = levenshteinpath(a, b)
    x = set(backtrack(dp))
    fig, ax = plt.subplots(figsize = (6,4))
    im = ax.imshow(dp, origin="upper")
    ax.set_title(f"Levenshtein DP for '{a}' → '{b}' (distance={dp[-1,-1]})")
    ax.set_xlabel("j (len of b prefix)")
    ax.set_ylabel("i (len of a prefix)")

    for i in range(dp.shape[0]):
        for j in range(dp.shape[1]):
            txt = ax.text(j, i, dp[i, j], ha="center", va="center",
                          fontsize=8, color="black" if (i,j) not in x else "white")
    ys, xs = zip(*x)
    ax.scatter(xs, ys, s=20, marker="s", facecolors="none", edgecolors="yellow", linewidths=1.5)
    plt.show()

showdp("kitten", "sitting")

tests = [
    ("kitten", "sitting"),
    ("intention", "execution"),
    ("Saturday", "Sunday"),
    ("abcdefg", "abcdefg"),
    ("algorithm", "altruistic"),
]

def timerun(a, b, runs=3):
    best = math.inf
    for _ in range(runs):
        t0 = time.perf_counter()
        d = levenshteindistance(a, b)
        best = min(best, (time.perf_counter() - t0))
    return d, best*1000

rows = []
for (x, y) in tests:
    d, ms = timerun(x, y)
    rows.append((x, y, d, f"{ms:.2f} ms"))

for r in rows:
    print(r)
