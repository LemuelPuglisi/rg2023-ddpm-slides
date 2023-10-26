---
layout: center
transition: slide-left
---

# Appendix A
## Deriving the variational lower bound

---
layout: center
transition: slide-up
---

## Deriving the variational lower bound

As in VAEs, we can find a variational lower bound for the likelihood, or equivalently, a **variational upper bound** to minimize instead of the negative log-likelihood.

$$
-\log p_\theta(x_0) \le-\log p_\theta(x_0) + \underbrace{\color{yellow}D_{KL}[q(x_{1:T}\mid x_0) \mid \mid p_\theta(x_{1:T}\mid x_0)]}_{\ge 0}
$$

<v-click>

Define

$$
D_{KL}[q(x_{1:T}\mid x_0) \mid \mid p_\theta(x_{1:T}\mid x_0) ] = \log \frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{1:T}\mid x_0)}
$$

</v-click>

---
layout: center
transition: slide-up
---

Replace

$$
-\log p_\theta(x_0) \le-\log p_\theta(x_0) +\color{yellow}\log \frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{1:T}\mid x_0)}
$$

<v-click>

Focus on the denominator $p_\theta(x_{1:T}\mid x_0)$

$$
p_\theta(x_{1:T}\mid x_0) = \frac{p_\theta(x_0 \mid x_{1:T})p_\theta(x_{1:T})}{p_\theta(x_0)} = 
\frac{p_\theta(x_0, x_{1:T})}{p_\theta(x_0)} = \frac{p_\theta(x_{0:T})}{p_\theta(x_0)}
$$

</v-click>

<v-click>

Replace

$$
\color{yellow}\log \frac{q(x_{1:T} \mid x_0)}{\frac{p_\theta(x_{0:T})}{p_\theta(x_0)}}\color{white} = 
\log \frac{q(x_{1:T} \mid x_0)p_\theta(x_0)}{p_\theta(x_{0:T})} = 
\log \frac{q(x_{1:T} \mid x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0)
$$

</v-click>


---
layout: center
transition: slide-up
---

Replace in the upper bound

$$
-\log p_\theta(x_0) \le -\log p_\theta(x_0)\color{white} + \color{yellow} \log \frac{q(x_{1:T} \mid x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0) \color{white} = \log \frac{q(x_{1:T} \mid x_0)}{p_\theta(x_{0:T})}
$$

<v-click>

The variational upper bound to minimize is:

$$
\log \frac{q(x_{1:T} \mid x_0)}{p_\theta(x_{0:T})}
$$

</v-click>

<v-click>

But can we compute it?

</v-click>

---
layout: center
transition: slide-up
---

Recall that since the diffusion process is a Markov process, we can use the Markov property:

$$
q\left( x_{1:T} \mid x_0 \right) = \prod_{t=1}^T q\left(x_t \mid x_{t-1}\right) \hspace{.5cm}
$$

and 

$$
p_\theta\left( x_{0:T} \right) = p_\theta\left(x_T\right)\prod_{t=1}^T p_\theta\left(x_{t-1} \mid x_t\right)
$$

---
layout: center
transition: slide-up
---

Replace in the upper bound:

$$ 
\begin{split}

\log \frac{q(x_{1:T} \mid x_0)}{p_\theta(x_{0:T})} 
&= \log \frac {\prod_{t=1}^T q\left(x_t \mid x_{t-1}\right)}
{p_\theta\left(x_T\right)\prod_{t=1}^T p_\theta\left(x_{t-1} \mid x_t\right)} \\

&= -\log p_\theta\left(x_T\right) +  \log \prod_{t=1}^T \frac { q\left(x_t \mid x_{t-1}\right)}
{p_\theta\left(x_{t-1} \mid x_t\right)} \\

&= -\log p_\theta\left(x_T\right) + \sum_{t=1}^T \log\frac {q\left(x_t \mid x_{t-1}\right)}
{p_\theta\left(x_{t-1} \mid x_t\right)} \\

\end{split}
$$

---
layout: center
transition: slide-up
---

Move out iteration $t=1$ from the sum:

$$
\begin{split}

-\log p_\theta\left(x_T\right) &+ \sum_{t=1}^T \log\frac {q\left(x_t \mid x_{t-1}\right)} {p_\theta\left(x_{t-1} \mid x_t\right)} = \\

&= -\log p_\theta\left(x_T\right) + \sum_{t=2}^T \log\frac {q\left(x_t \mid x_{t-1}\right)} {p_\theta\left(x_{t-1} \mid x_t\right)} + 
\log\frac {q\left(x_1 \mid x_0\right)} {p_\theta\left(x_0 \mid x_1\right)}

\end{split}
$$

---
layout: center
transition: slide-up
---

Since the Markov property holds, we can write:

$$
q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})  
$$

<v-click>

By applying Bayes rule we have

$$
q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}
$$

</v-click>

---
layout: center
transition: slide-up
---

Replace in the numerator in the second term:

$$
\sum_{t=2}^T \log\frac {q\left( x_{t-1} \mid x_t, x_0 \right) q\left(x_t \mid x_0\right)} {p_\theta\left(x_{t-1} \mid x_t\right) q\left(x_{t-1} \mid x_0\right)}
$$

<v-click>

which becomes

$$
\sum_{t=2}^T \log\frac {q\left( x_{t-1} \mid x_t, x_0 \right)}{p_\theta\left(x_{t-1} \mid x_t\right)} + \sum_{t=2}^T \log\frac { q\left(x_t \mid x_0\right)} {q\left(x_{t-1} \mid x_0\right)}
$$

</v-click>

---
layout: center
transition: slide-up
---

Replace the second term in the loss

$$
\begin{split}

-\log p_\theta\left(x_T\right) +

\sum_{t=2}^T \log\frac {q\left( x_{t-1} \mid x_t, x_0 \right)}{p_\theta\left(x_{t-1} \mid x_t\right)}+ 

\color{yellow}\sum_{t=2}^T \log\frac { q\left(x_t \mid x_0\right)} {q\left(x_{t-1} \mid x_0\right)}\color{white} + 

\log\frac {q\left(x_1 \mid x_0\right)} {p_\theta\left(x_0 \mid x_1\right)}

\end{split}
$$

<v-click>

Notice that

$$
\begin{split}

\color{yellow}\sum_{t=2}^T \log\frac { q\left(x_t \mid x_0\right)} {q\left(x_{t-1} \mid x_0\right)}\color{white} &= \log \prod_{t=2}^T \frac { q\left(x_t \mid x_0\right)} {q\left(x_{t-1} \mid x_0\right)} \\
&= \log\left( \frac{q\left(x_2 \mid x_0\right)} {q\left(x_1 \mid x_0\right)}\frac{q\left(x_3 \mid x_0\right)} {q\left(x_2 \mid x_0\right)} \dots\frac{q\left(x_T \mid x_0\right)} {q\left(x_{T-1} \mid x_0\right)}\right) \\
&= \log \frac{q\left(x_T \mid x_0\right)} {q\left(x_1 \mid x_0\right)} \\

\end{split}
$$

</v-click>

---
layout: center
transition: slide-up
---

Replace the term:

$$
\begin{split}

-\log p_\theta\left(x_T\right) +

\sum_{t=2}^T \log\frac {q\left( x_{t-1} \mid x_t, x_0 \right)}{p_\theta\left(x_{t-1} \mid x_t\right)}+ 

\color{yellow}\log \frac{q\left(x_T \mid x_0\right)} {q\left(x_1 \mid x_0\right)} +

\log\frac {q\left(x_1 \mid x_0\right)} {p_\theta\left(x_0 \mid x_1\right)}

\end{split}
$$

<v-click>

Merge the last two logs and simplify the yellow terms

$$
\begin{split}

-\log p_\theta\left(x_T\right) +

\sum_{t=2}^T \log\frac {q\left( x_{t-1} \mid x_t, x_0 \right)}{p_\theta\left(x_{t-1} \mid x_t\right)}+ 

\log \frac{q\left(x_T \mid x_0\right)} {\color{yellow}q\left(x_1 \mid x_0\right)\color{white}} \frac {\color{yellow}q\left(x_1 \mid x_0\right)\color{white}} {p_\theta\left(x_0 \mid x_1\right)}

\end{split}
$$

</v-click>

---
layout: center
transition: slide-up
---

Decompose the log fraction

$$
\begin{split}

-\log p_\theta\left(x_T\right) +

\sum_{t=2}^T \log\frac {q\left( x_{t-1} \mid x_t, x_0 \right)}{p_\theta\left(x_{t-1} \mid x_t\right)}+ 

\color{yellow}\log \frac{q\left(x_T \mid x_0\right)}{p_\theta\left(x_0 \mid x_1\right)}

\end{split}
$$

<v-click>

Merge the two yellow terms 

$$
\begin{split}

\color{yellow}-\log p_\theta\left(x_T\right)\color{white} +

\sum_{t=2}^T \log\frac {q\left( x_{t-1} \mid x_t, x_0 \right)}{p_\theta\left(x_{t-1} \mid x_t\right)} + 

\color{yellow}\log q\left(x_T \mid x_0\right)\color{white} - \log {p_\theta\left(x_0 \mid x_1\right)}

\end{split}
$$

</v-click>

<v-click>

And finally we derived the variational upper bound:

$$
\begin{split}
\underbrace{\log \frac{q\left(x_T \mid x_0\right)}{p_\theta\left(x_T\right)}}_{L_T} + \underbrace{\sum_{t=2}^T \log\frac {q\left( x_{t-1} \mid x_t, x_0 \right)}{p_\theta\left(x_{t-1} \mid x_t\right)}}_{L_{t-1}} \underbrace{- \log p_\theta\left(x_0 \mid x_1\right)}_{L_0} 

\end{split}
$$

</v-click>

---
layout: center
transition: slide-up
---

Notice that we can rewrite $L_T$ and $L_{t-1}$ as KL divergences:

$$
\begin{split}
L = D_{KL}\left( q\left(x_T\mid x_0\right) \mid \mid p_\theta\left(x_T\right) \right) &+ \sum_{t=2}^T D_{KL}\left( q\left(x_{t-1}\mid x_t, x_0\right) \mid \mid p_\theta\left(x_{t-1} \mid x_t\right) \right) \\ 
&- \log p_\theta\left(x_0 \mid x^{(1)}\right)
\end{split}
$$





