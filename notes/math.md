$$

% \bm{s}^{(1)}(\bm{y}_t, \bm{y}_{1\dots t}) = \argmax_{k\in t-1} p(\bm{y}_t|\bm{m}({y_{1\dots t-1},{}})) \\ 
% S^{(n+1)} = S^{(n)}\\

S^{(1)}(t) = \{t-1\} \\
S^{(n+1)}(t, \bm{y}_t, \bm{y}_{[1\dots t-1]}, S^{(n)}) = S^{(n)} \cup \argmax_{k\in [1\dots t-1] / S^{(n)}} p(\bm{y}_t|\bm{m}(\bm{y}_{[1\dots t-1]}, S^{(n)}) + \bm{m}(\bm{r}, \overline{S^{(n)}})) \quad \bm{r} \sim U(1, t) \\

$$

- $t$ is the token to evaluate its rationale
- $S^{n}$ is the set of all the token positions of rationale candidates at step $n$
- $p(y, \bm{x})$ is the likelihood of token $y$ followed by sequence $\bm{x}$ estimated by a model
- $\bm{m}(\bm{y}, S)$ is a masking function that masking elements $\bm{y}_i$ to $0$ in the sequence $\bm{y}$  where $i \in S$
- $\bm{r}$ is a sequence that sampled from uniform distribution $U(1, t)$

