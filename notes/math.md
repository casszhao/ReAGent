
## Overview

$$

\begin{align}

    p^{(o)}(y_t, \bm{y}_{[1\dots t-1]}) &= p(y_t|\bm{y}_{[1\dots t-1]}) \\
    p^{(r)}(y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}) &= p(y_t|(\bm{m}(\bm{y}_{[1\dots t-1]}, \overline{\mathcal{R}}) + \bm{m}(\bm{r}(\bm{y}_{[1\dots t-1]}), \mathcal{R}))) \\
    \Delta p(y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}) &= p^{(o)}(y_t, \bm{y}_{[1\dots t-1]}) - p^{(r)}(y_t, y_{[1\dots t-1]}, \mathcal{R}) \\
    % \Delta p^{(l)}(y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}) &= logit(\Delta p(y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R})) \\
    \Delta \bm{s}_n(y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}) &= \bm{m}(\Delta p(y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}), \mathcal{R}) + \bm{m}(-\Delta p(y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}), \overline{\mathcal{R}}) \\
    \bm{s}^{(l)}_n(\bm{s}^{(l)}_{n-1}, y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}) &= \bm{s}^{(l)}_{n-1} + \eta \text{logit}\left(\frac{\Delta \bm{s}_n(y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}) + 1}{2}\right) \\
    \bm{s}_n(\bm{s}^{(l)}_{n-1}, y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}) &= \text{softmax}(\bm{s}^{(l)}_n(\bm{s}^{(l)}_{n-1}, y_t, \bm{y}_{[1\dots t-1]}, \mathcal{R}))

\end{align}

$$

- $p(y_t|\bm{y}_{[1\dots t-1]})$: model that predict the likelihood of $y_t$ condition on $\bm{y}_{[1\dots t-1]}$
- $\mathcal{R}$: set of replacing positions
- $\bm{r}(\bm{y}_{[1\dots t-1]})$: generate replacement tokens from $\bm{y}_{[1\dots t-1]}$
- $\bm{m}(p, \mathcal{R})$:a masking function. return a tensor where position in the set $\mathcal{R}$ to $0$, otherwise $p$
- $\bm{s}^{(l)}_n$: importance score in logistic scale at step $n$
- $\bm{s}_n$: importance score at step $n$

$$
\mathcal{R} = [ i_1, i_2, \dots, i_n ] \sim U(1, t-1)
$$

- $n$ number of items to mask

## Uniform Replacing

$$

\bm{r}(\bm{y}_{[1\dots t-1]}) = [ \hat{y}_1, \hat{y}_2, \dots, \hat{y}_{t-1} ] \sim U(\mathcal{Y})

$$

## Inferential Replacing

$$
\bm{r}(\bm{y}_{[1\dots t-1]}) = \left[ y_{1}, \argmax_{\hat{y}_2} p(\hat{y}_2|y_{1}), \argmax_{\hat{y}_3} p(\hat{y}_{3}|\bm{y}_{[1\dots 2]}), \dots, \argmax_{\hat{y}_{t-1}} p(\hat{y}_{t-1}|\bm{y}_{[1\dots t-2]}) \right]
$$

## POStag Replacing

$$

\bm{r}(\bm{y}_{[1\dots t-1]}) = [ \hat{y}_1, \hat{y}_2, \dots \hat{y}_{t-1} ] \sim U(\mathcal{Y}^{(g)}([y_1, y_2, \dots, y_{t-1}]))

$$

- $\mathcal{Y}^{(g)}(y)$: is the set of tokens having the same POS tag of y
