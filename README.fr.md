# FR Projet d’affectation optimale — *Méthode Hongroise (Hungarian Algorithm)*

**Auteur : Julien Gimenez**  
**Date : 2025**  
**Langage : Python (pandas, networkx)**  

![Logo Student-Project Assignment Using the Kuhn–Munkres - Hungarian - Algorithm](docs/img/hugarian-method.png)

---

## Objectif

Ce projet implémente un système complet d'**affectation optimale** entre étudiants et projets à partir de préférences exprimées sous forme :

- **Ordonnée** : liste de projets par rang (`P1;P2;P3`)
- **Pondérée** : liste de projets avec poids (`P1:0.1;P2:0.3;P3:10`)

L'objectif est de minimiser le **coût global de satisfaction** selon les choix individuels ou de groupe, en utilisant :

> ⚙️ **L'algorithme hongrois (Hungarian / Kuhn-Munkres Algorithm)**  
> combiné à un modèle de **flot à coût minimum** pour gérer les capacités multiples.

---

## Modélisation mathématique

### Représentation du problème

On cherche à attribuer chaque étudiant $e_i$ à un projet $p_j$ en minimisant un **coût total** $C_{\text{total}}$.

$\min_{\pi} \sum_{i=1}^{N} c_{i,\pi(i)}$

où :
- $N$ = nombre d’étudiants (ou de groupes),
- $\pi(i)$ = projet attribué à l’étudiant $i$,
- $c_{ij}$ = coût associé au couple $(e_i, p_j)$.

---

### Cas **non pondéré**

Le coût dépend uniquement du **rang** $r_{ij}$ du projet dans la liste des préférences :

$c_{ij} =
\begin{cases}
r_{ij} - 1 & \text{si } p_j \text{ est listé} \\
\text{penalty} & \text{sinon.}
\end{cases}$

---

### Cas **pondéré**

Le coût est directement le **poids déclaré** par l’étudiant :

$c_{ij} = w_{ij}$

avec $w_{ij} \in [0.1, 10]$, où une valeur faible traduit une forte préférence.

---

### Objectif global (modèle de flot à coût minimum)

Le problème est exprimé sous forme de graphe orienté $G = (V, E)$, où :

$s \rightarrow e_i \rightarrow p_j \rightarrow t$

et où l’on cherche à minimiser :

$\min \sum_{(u,v) \in E} c_{uv} \cdot f_{uv}$

Sous contraintes :

$\begin{cases}
\sum_{v} f_{uv} - \sum_{v} f_{vu} = b_u, & \forall u \in V \\
0 \le f_{uv} \le \text{cap}_{uv}
\end{cases}$

- $f_{uv}$ : flux (nombre d’affectations)  
- $b_u$ : demande (source ou puits)  
- $c_{uv}$ : coût de l’arc  
- $\text{cap}_{uv}$ : capacité (souvent 1)

---

## Algorithme Hongrois — *Principe et Intuition*

L'**algorithme hongrois**, aussi appelé **algorithme de Kuhn-Munkres**, résout le **problème d'affectation** en temps polynomial $O(n^3)$.  
Il garantit la **solution optimale** pour des coûts réels $c_{ij}$.

### Étapes :

1. **Réduction par ligne :**  
   $c'_{ij} = c_{ij} - \min_j(c_{ij})$
2. **Réduction par colonne :**  
   $c''_{ij} = c'_{ij} - \min_i(c'_{ij})$
3. **Marquage des zéros** : on cherche à couvrir tous les zéros de la matrice par un nombre minimal de lignes et colonnes.
4. **Ajustement de la matrice** :  
   - Si le nombre de lignes couvrantes = $N$, une solution parfaite est trouvée.  
   - Sinon, on soustrait le plus petit élément non couvert et on recommence.

---

## Échelle des pondérations

| Intention | Poids conseillé | Interprétation |
|------------|----------------:|----------------|
| ❤️ Premier choix | **0.1** | Très fort désir |
| 💚 Très bon choix | **0.15 – 0.25** | Fort désir |
| 💛 Bon choix | **0.25 – 0.35** | Préférence positive |
| 😐 Neutre | **0.5** | Indifférent |
| 😒 À éviter | **1 – 3** | Préférence négative |
| 😖 Peu apprécié | **6 – 9** | Très peu souhaité |
| 💀 Détesté | **10** | Forte pénalité |

Formule linéaire :  
$w(r) = 0.1 + \frac{0.4}{R - 1} \times (r - 1)$  
avec $R$ = nombre de projets listés.

---

## Exécution


### Mode notebook
```bash
jupyter notebook src/Assignment-Project_Hungarian-Method.ipynb
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## 🧾 Licence

Licence libre **BSD 3-Clause**  
© 2025 — Julien Gimenez  

> ✨ *“L'élégance d'une affectation optimale se mesure à la satisfaction totale.”*
