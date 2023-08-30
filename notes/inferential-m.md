$$

\begin{align}
    

\bm{W} &= \begin{bmatrix}
    w_1, q_2, \dots, w_{t-1}
\end{bmatrix} \\

\bm{W}^\prime &= \begin{bmatrix}
    w_1^\prime &=& w_1 &, \\
    w_2^\prime &=& m^T(\argmax_{w^{(b)}_2}(f^{(b)}(w^{(b)}_2 | m([w_1])))) &, \\
    w_3^\prime &=& m^T(\argmax_{w^{(b)}_3}(f^{(b)}(w^{(b)}_3 | m([w_0, w_1])))) &, \\
    \vdots \\
    w_{t-1}^\prime &=& m^T(\argmax_{w^{(b)}_{t-1}}(f^{(b)}(w^{(b)}_{t-1} | m([w_0, w_1, \dots, w_{t-2}])))) \\

\end{bmatrix}^T

\end{align}
$$

- $m$: A transform function that can transform representations within domain of definition of $f^{(a)}$ to representations within domain of definition of $f^{(b)}$. During implementation, the domain of language (text) can act as an intermediate domain. (Note: this function will be a complex function)
- $m^T$: the inversion of $m$
- $f^{(a)}(w_t | [w_1, w_2, \dots, w_{t-1}])$: likelihood of $w_t$ conditional on $[w_1, w_2, \dots, w_{t-1}]$ in domain $a$.

$$
\begin{align}
    
p^{(o)}(w_t, \bm{W}) &= f^{(a)}( w_{t} | \bm{W} ) \\
p^{(r)}(w_t, \bm{W}, \bm{W}^\prime, \mathcal{R}) &= f^{(a)}( w_{t} | m(\bm{W}^\prime, \mathcal{R}) + m(\bm{W}, \mathcal{\overline{R}}))

\end{align}
$$

- $p^{(o)}$: likelihood of $w_t$ on original sequence
- $p^{(r)}$: likelihood of $w_t$ on replaced sequence

