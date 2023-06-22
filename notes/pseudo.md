Rationalization($y_{1...t}$, $y_{t+1}$)
    inference $p(y_{t+1}|y_{1...t})$
    Initialize importance scores $s_{1...t}$
    loop until StoppingCondition($y_{1...t}$, $y_{t+1}$, $s$)
        generate a set of position $R$ that need to be replaced 
        replace part of the sequence $y_{1...t}$ to $\hat{t_{1...t}}$ based on $R$
        inference $p(y_{t+1}|\hat{y_{1...t}})$
        compute $\delta p$ by $p(y_{t+1}|y_{1...t}) - p(y_{t+1}|\hat{y_{1...t}})$
        update importance scores $s_{1...t}$ by $\delta p$ and $R$
    return rational is tokens of top N $s$

StoppingCondition($y_{1...t}$, $y_{t+1}$, $s$)
    replace tokens in $y_{1...t}$ which dose not have top $N$ importance score to $y^{{e}}_{1...t}$
    inference $W$ which is the top K prediction of the next token of $y^{{e}}_{1...t}$
    return whether $y_{t+1}$ exist in the set $W$
