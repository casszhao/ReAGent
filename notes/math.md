
## Overview

$$

\begin{align}
    p^{(o)}(y_t, y_{[1\dots t-1]}) &= p(\bm{y}_t|\bm{y}_{[1\dots t-1]}) \\
    p^{(r)}(y_t, y_{[1\dots t-1]}, \mathcal{R}) &= p(\bm{y}_t|(\bm{m}(\bm{y}_{[1\dots t-1]}, \overline{\mathcal{R}}) + \bm{m}(\bm{r}(\bm{y}_{[1\dots t-1]}), \mathcal{R}))) \\
    \Delta p(y_t, y_{[1\dots t-1]}, \mathcal{R}) &= p^{(o)}(y_t, y_{[1\dots t-1]}) - p^{(r)}(y_t, y_{[1\dots t-1]}, \mathcal{R}) \\
    \bm{s^{(l)}_n}(s^{(l)}_{n-1}, y_t, y_{[1\dots t-1]}, \mathcal{R}) &= m(s^{(l)}_{n-1} + \Delta p(y_t, y_{[1\dots t-1]}, \mathcal{R}), \mathcal{R}) + m(s^{(l)}_{n-1} - \Delta p(y_t, y_{[1\dots t-1]}, \mathcal{R}), \overline{\mathcal{R}}) \\
    s_n(s^{(l)}_{n-1}, y_t, y_{[1\dots t-1]}, \mathcal{R}) &= softmax(s^{(l)}_n(s^{(l)}_{n-1}, y_t, y_{[1\dots t-1]}, \mathcal{R}))

\end{align}

$$

- $p(\bm{y}_t|\bm{y}_{[1\dots t-1]})$: model that predict the likelihood of $\bm{y}_t$ condition on $\bm{y}_{[1\dots t-1]}$
- $\mathcal{R}$: set of replacing positions
- $\bm{r}(\bm{y}_{[1\dots t-1]})$: generate replacement tokens from $\bm{y}_{[1\dots t-1]}$
- $\bm{m}(\bm{y}_{[1\dots t-1]}, \mathcal{R})$:a masking function. set tokens which position is in the set $\mathcal{R}$ to $0$
- $s^{(l)}_n$: importance score in logistic scale at step $n$
- $s_n$: importance score at step $n$

$$
\mathcal{R} = [ i_1, i_2, \dots, i_n ] \sim U(1, t-1)
$$

- $n$ number of items to mask

## Uniform Replacing

$$

\bm{r}(\bm{y}_{[1\dots t-1]}) = [ \hat{\bm{y}}_1, \hat{\bm{y}}_2, \dots, \hat{\bm{y}}_{t-1} ] \sim U(\mathcal{Y})

$$

## Inferential Replacing

$$
\bm{r}(\bm{y}_{[1\dots t-1]}) = \left[ \bm{y}_{1}, \argmax_{\hat{\bm{y}}_2} p(\hat{\bm{y}}_2|\bm{y}_{1}), \argmax_{\hat{\bm{y}}_3} p(\hat{\bm{y}}_{3}|\bm{y}_{[1\dots 2]}), \dots, \argmax_{\hat{\bm{y}}_{t-1}} p(\hat{\bm{y}}_{t-1}|\bm{y}_{[1\dots t-2]}) \right]
$$

## POStag \mathcal{R}eplacing

$$

\bm{r}(\bm{y}_{[1\dots t-1]}) = [ \hat{\bm{y}}_1, \hat{\bm{y}}_2, \dots \hat{\bm{y}}_{t-1} ] \sim U(\mathcal{Y}^{(g)}([\bm{y}_1, \bm{y}_2, \dots, \bm{y}_{t-1}]))

$$

- $Y^{(g)}(y)$: is the set of tokens having the same POS tag of y