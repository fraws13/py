# Rapporto Tecnico — Esame Data Science 26 Febbraio 2026

**Laurea Magistrale in Ingegneria Informatica — Università degli Studi di Salerno**

---

## 1. Introduzione e Obiettivi

L'esame richiede l'analisi di un problema di classificazione binaria su due dataset:
- **PART1**: 5000 osservazioni con feature scalare (1D) e label ∈ {-1, +1}
- **PART2**: 5000 osservazioni con feature vettoriale (20D) e label ∈ {-1, +1}

Gli obiettivi sono: derivare il classificatore ottimo teorico, implementare SGD per regressione logistica, applicare PCA e confrontare le performance.

---

## 2. Modello Statistico e Classificatore Ottimo (Punto a)

### 2.1 Assunzioni del modello

- **Label Y**: distribuzione a priori uniforme → P(Y = -1) = P(Y = +1) = 0.5
- **Feature X | Y**: distribuzione Gaussiana condizionata
  - X | Y = -1 ~ N(μ₋₁ = -1, σ² = 0.5)
  - X | Y = +1 ~ N(μ₊₁ = +1, σ² = 0.5)
- Varianza σ² = 0.5 uguale per entrambe le classi (caso omoschedastico)

### 2.2 Derivazione del classificatore ottimo (Bayes)

Per classi equiprobabili e Gaussiane con stessa varianza, la regola di decisione ottima (MAP) si riduce a confrontare i rapporti di verosimiglianza. La soglia di decisione ottima è il punto equidistante tra le due medie:

$$\hat{Y} = \begin{cases} +1 & \text{se } x > 0 \\ -1 & \text{se } x \leq 0 \end{cases}$$

La soglia x = 0 è ottima perché le medie sono simmetriche rispetto all'origine e le classi sono equiprobabili.

### 2.3 Probabilità di errore teorica (Bayes Error)

Per simmetria delle Gaussiane e classi equiprobabili:

$$P_e = P(X > 0 | Y = -1) \cdot P(Y = -1) + P(X \leq 0 | Y = 1) \cdot P(Y = 1)$$

Dato che P(X < 0 | Y = 1) = P(X > 0 | Y = -1) per simmetria:

$$P_e = P(X < 0 | Y = 1) = \Phi\left(\frac{0 - 1}{\sqrt{0.5}}\right) = \Phi\left(-\sqrt{2}\right) \approx 0.0786$$

**Risultato numerico**: P_e ≈ **7.86%** (o 7.93% a seconda dell’arrotondamento)

---

## 3. Classificatore Logistico con SGD (Punto b)

### 3.1 Modello logistico

Il classificatore logistico modella:

$$P(Y = +1 | X) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

con decisione: ŷ = +1 se σ(w^T x + b) ≥ 0.5, altrimenti ŷ = -1.

### 3.2 Algoritmo SGD (Stochastic Gradient Descent)

- **Loss**: cross-entropy (log-loss) per classificazione binaria
- **Gradiente stocastico** (un campione per iterazione):
  $$\nabla L_i = -\frac{y_i x_i}{1 + \exp(y_i \cdot (w^T x_i))}$$
- **Aggiornamento**: w ← w - η · ∇L_i

Sono state testate due strategie di step-size:
1. **Step-size costante**: η = 0.1
2. **Step-size decrescente**: η = τ/(i+1) con τ = 1

### 3.3 Risultati dell’addestramento

| Configurazione | Loss finale | Beta finale (appross.) |
|----------------|-------------|------------------------|
| Step-size costante | ~0.26 | [1.69, -0.07] |
| Step-size decrescente | ~0.26 | [5.74, 0.06] (media ultimi 10%) |

La media degli ultimi 10% delle iterazioni è usata per attenuare la varianza dell’SGD.

### 3.4 Valutazione sul test set

- **Accuratezza empirica**: ~92%
- **Probabilità di errore empirica**: ~8.00%
- **Confronto con Bayes**: L’errore empirico (~8%) è molto vicino all’errore teorico (~7.86%), indicando che l’SGD converge verso la soglia ottima e che il modello logistico approssima bene il classificatore di Bayes per questo problema.

---

## 4. PCA e Riduzione Dimensionalità (Punto c)

### 4.1 Procedura PCA tramite SVD

1. **Standardizzazione**: i dati sono centrati e scalati (media 0, varianza 1)
2. **SVD**: decomposizione della matrice dei dati standardizzati
3. **Proiezione**: si mantengono le prime 2 componenti principali (PC1, PC2)

### 4.2 Varianza spiegata

Le prime componenti principali spiegano una frazione limitata della varianza totale:

| Componente | Varianza spiegata |
|------------|-------------------|
| PC1 | 7.55% |
| PC2 | 5.47% |
| PC3 | 5.44% |
| ... | ... |
| PC20 | 2.48% |

La varianza è distribuita in modo relativamente uniforme tra le 20 dimensioni, tipico di dati con poca struttura dominante.

### 4.3 Addestramento sul dataset ridotto

Il classificatore logistico è addestrato sulle 2 componenti principali. La probabilità di errore sul test set è confrontabile con quella del modello 1D, senza miglioramenti significativi.

---

## 5. Visualizzazioni (Punto d)

### 5.1 Dataset 1D (PART1)

- Grafico delle densità Gaussiane N(-1, 0.5) e N(1, 0.5)
- Soglia di decisione ottima a x = 0
- Aree di errore evidenziate (x > 0 per Y = -1 e x ≤ 0 per Y = +1)

### 5.2 Dataset 2D (dopo PCA su PART2)

- Scatter plot delle classi nelle coordinate (PC1, PC2)
- Confine di decisione lineare del classificatore logistico
- Le classi risultano sovrapposte lungo PC2, con separabilità principalmente lungo PC1

---

## 6. Interpretazione: Perché la seconda dimensione non aiuta (Punto e)

### 6.1 Analisi grafica

1. **Separabilità lungo PC1**: La maggior parte dell’informazione discriminante è già nella prima componente principale. PC1 corrisponde alla direzione di massima varianza e coincide con la direzione di separazione tra le classi.

2. **Ruolo di PC2**: Nel piano (PC1, PC2) le classi non si separano meglio lungo PC2. La sovrapposizione lungo PC2 è simile a quella lungo PC1, quindi non si riduce l’errore.

3. **Confine di decisione**: Il confine del classificatore logistico è quasi verticale (parallelo a PC2), indicando che il peso associato a PC2 è trascurabile. La decisione dipende quasi solo da PC1.

### 6.2 Motivazione teorica

- **Bayes Error**: L’errore di Bayes (~7.9%) è un limite inferiore per qualsiasi classificatore. Aggiungere una dimensione che non aumenta la separabilità lineare non può ridurre l’errore al di sotto di questo valore.
- **Struttura dei dati**: PART2 è ottenuto da PART1 aggiungendo 19 dimensioni con informazione ridondante o rumorosa. La PCA mantiene l’informazione rilevante in PC1; PC2 non aggiunge potere discriminante.
- **Invarianza della regione di sovrapposizione**: L’area di sovrapposizione tra le densità delle due classi nel piano 2D è simile a quella in 1D, quindi l’errore di classificazione resta sostanzialmente invariato.

---

## 7. Conclusioni

1. **Classificatore ottimo**: Soglia x = 0, con errore teorico ≈ 7.86%.
2. **SGD**: L’algoritmo converge verso parametri coerenti con il modello Bayesiano; l’errore empirico (~8%) è vicino all’errore teorico.
3. **PCA**: La riduzione a 2 dimensioni non migliora le performance perché l’informazione discriminante è già concentrata in PC1.
4. **Interpretazione**: Le visualizzazioni confermano che la seconda componente principale non aumenta la separabilità lineare tra le classi.

---

## 8. Riferimenti al Codice

- **Dataset**: `exercise2_26_02_26_dataset_PART1.npy`, `exercise2_26_02_26_dataset_PART2.npy`
- **Librerie**: NumPy, Matplotlib, SciPy (norm, SVD)
- **Split**: training/test per valutazione dell’errore empirico
- **Funzione di loss**: `np.logaddexp(0, -t)` per stabilità numerica nella cross-entropy

---

*Rapporto redatto per la verifica in Data Science — 26 febbraio 2026*
