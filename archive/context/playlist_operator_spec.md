# Playlist Operator: Cyclic Linear Dynamics Spec

## Idea

Given a looping sequence of vectors $x_1, \dots, x_N \in \mathbb{R}^D$ (e.g. 20 songs as raw audio),
find a matrix $A$ such that

$$A x_n = x_{n+1}, \qquad A x_N = x_1.$$

One matrix multiply advances the playlist; iterating $A$ plays it forever.

## Theory

**Matrix form.** With $X = [x_1 \cdots x_N]$ ($D \times N$) and $S$ the $N \times N$ cyclic
shift permutation ($Se_j = e_{j+1}$, subdiagonal ones plus a top-right corner one):

$$A X = X S.$$

**Fourier diagonalization.** $S = F \Lambda F^*$ with $\Lambda = \mathrm{diag}(\omega^k)$,
$\omega = e^{2\pi i/N}$. Writing the temporal DFT modes $\hat{x}_k = \sum_n x_n \omega^{-nk}$
(each a $D$-vector), the condition becomes an eigenvector statement:

$$A \hat{x}_k = \omega^k \hat{x}_k \quad \text{for every active frequency } k.$$

**Existence.** An exact $A$ exists **iff** the nonzero modes $\{\hat{x}_k\}$ are linearly
independent. Equivalently: $\#\{\text{active frequencies}\} = \mathrm{rank}(X) \le D$.
Failure modes are both "one input, two outputs" violations:
inconsistent repetition ($x_1 = x_3$ but $x_2 \neq x_4$) and frequency collision
($\hat{x}_k \parallel \hat{x}_j$, $k \neq j$). Consistent sub-periodicity is fine.
If existence fails, delay embedding lifts to a dimension where it holds.

**Spectral pinning.** $A^N = I$ on $\mathrm{span}(X)$, so there the eigenvalues are exactly
$N$-th roots of unity, no Jordan blocks: pure rotation, no growth/decay. This is DMD
degenerating to the temporal DFT (Chen–Tu–Rowley); Koopman spectrum of a periodic orbit.

**Solution family.** All exact solutions:

$$A = \underbrace{X_{\text{next}}\, G^{-1} X^\top}_{A_0,\ \text{min-norm, rank } N}
    + Z\, P_\perp, \qquad
G = X^\top X,\quad P_\perp = I - X G^{-1} X^\top,\quad Z \text{ arbitrary}.$$

$P_\perp$ annihilates every $x_n$, so $Z$ (a $D(D{-}r)$-dim family) is invisible to the
playlist and free for regularization: contraction off-span, circulant filtering, etc.
$A_0$ reads as a linear heteroassociative memory: correlate ($X^\top$), unmix ($G^{-1}$),
emit successors ($X_{\text{next}}$).

## Practicalities

- **Never materialize $A$** at audio scale ($D \sim 4\text{M} \Rightarrow$ tens of TB dense).
  Store $X$, $X_{\text{next}}$, and the $N \times N$ matrix $G^{-1}$; apply in $O(DN)$:

  ```python
  Xnext = X[:, (np.arange(N) + 1) % N]
  Ginv  = np.linalg.inv(X.T @ X)              # N x N
  A0    = lambda v: Xnext @ (Ginv @ (X.T @ v))
  ```

- **Numerics.** Conditioning of $G$ = degree of independence of the songs; near-duplicates
  are the finite-precision shadow of the existence condition. Keep $\|Z\|$ moderate:
  it multiplies the floating-point residual of $P_\perp x_n$.

- **Rendering/inspection.** Any linear or polynomial statistic of $A$ streams through the
  factors: $RAC = (RU)(C^\top V)^\top$; block energy via $N \times N$ cross-Grams.
  Cost scales with output pixels, not $D^2$.

## Limits

- **Stability.** The song span is an attracting invariant subspace (tunable via $Z$), but
  within it the playlist orbit is only neutrally stable: attraction would need $|\lambda| < 1$
  on a data mode, contradicting the root-of-unity pinning. Mixtures rotate forever; noise is
  ignored only insofar as it is orthogonal to the span.
- **True attractor** (error correction, mixture purification) requires nonlinearity, minimally
  a softmax sharpening of the song coefficients between steps — the step from linear
  associative memory to Hopfield/attention-style recall.
- **Wrong parameterizations.** $A$ cannot be a convolution/circulant: per-frequency scalar
  action cannot map distinct spectra to each other. Template-to-template recall is rank
  structure, not shift-equivariant structure; spend Fourier priors on the $Z$ part instead.
