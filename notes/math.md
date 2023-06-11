
$$

\begin{align}
    p^{(o)}(y_t, y_{[1\dots t-1]}) &= p(\bm{y}_t|\bm{y}_{[1\dots t-1]}) \\
    p^{(r)}(y_t, y_{[1\dots t-1]}, R) &= p(\bm{y}_t|(\bm{m}(\bm{y}_{[1\dots t-1]}, \overline{R}) + \bm{m}(\bm{r}(\bm{y}_{[1\dots t-1]}), R))) \\
    \Delta p(y_t, y_{[1\dots t-1]}, R) &= p^{(o)}(y_t, y_{[1\dots t-1]}) - p^{(r)}(y_t, y_{[1\dots t-1]}, R) \\
    \bm{s^{(l)}_n}(s^{(l)}_{n-1}, y_t, y_{[1\dots t-1]}, R) &= m(s^{(l)}_{n-1} + \Delta p(y_t, y_{[1\dots t-1]}, R), R) + m(s^{(l)}_{n-1} - \Delta p(y_t, y_{[1\dots t-1]}, R), \overline{R}) \\
    s_n(s^{(l)}_{n-1}, y_t, y_{[1\dots t-1]}, R) &= softmax(s^{(l)}_n(s^{(l)}_{n-1}, y_t, y_{[1\dots t-1]}, R))

\end{align}

$$

- $p(\bm{y}_t|\bm{y}_{[1\dots t-1]})$: model that predict the likelihood of $\bm{y}_t$ condition on $\bm{y}_{[1\dots t-1]}$
- $R$: set of replacing positions
- $\bm{r}(\bm{y}_{[1\dots t-1]})$: generate replacement tokens from $\bm{y}_{[1\dots t-1]}$
- $\bm{m}(\bm{y}_{[1\dots t-1]}, R)$:a masking function. set tokens which position is in the set $R$ to $0$
- $s^{(l)}_n$: importance score in logistic scale at step $n$
- $s_n$: importance score at step $n$