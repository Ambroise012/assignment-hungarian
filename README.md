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

> **L'algorithme hongrois (Hungarian / Kuhn-Munkres Algorithm)**  
> combiné à un modèle de **flot à coût minimum** pour gérer les capacités multiples.

---
# Résultats:

![Results1](docs/img/results1.png)

![Results2](docs/img/results2.png)

![Results3](docs/img/results3.png)
---
Fonctionnement

# Algorithme de couplage hongrois (flot à coût minimal) – Explication et exemple 3×3

Ce programme implémente une **affectation optimale** entre un ensemble d’étudiants et un ensemble de projets, 
en cherchant à minimiser un **coût global** associé à la qualité des correspondances entre les deux groupes.  
Il repose sur un **modèle de flot à coût minimal**, équivalent à l’**algorithme hongrois** lorsque chaque étudiant 
doit être affecté à un projet unique de capacité $1$.

---

## 1. Principe général

On dispose de deux fichiers d’entrée :

- **`projects.csv`** : liste des projets, avec :
  - `id` : identifiant unique (ex. `A`, `B`, `C`) ;
  - `label` : libellé (facultatif) ;
  - `capacity` : nombre maximal d’étudiants assignables (défaut = 1).

- **`student-choices.csv`** : liste des étudiants, avec :
  - `student` : nom de l’étudiant (ex. `S1`, `S2`, …) ;
  - `prefs` : préférences, soit **ordonnées** (`A;B;C`), soit **pondérées** (`A:0;B:1.5;C:3`) ;
  - `weight` : poids (défaut = 1, utile pour des groupes) ;
  - `names` : noms explicites pour chaque copie (optionnel).

Deux modes sont possibles :

1. **Ordonné** → les projets sont classés par ordre de préférence.  
   Le coût associé au rang $r$ est donné par :
   $$
   c_r = r - 1
   $$
   Exemple : 1er vœu $=0$, 2e vœu $=1$, 3e vœu $=2$.

2. **Pondéré** → chaque étudiant indique directement un **coût** numérique (plus petit = meilleur).  
   Exemple : `A:0;B:1;C:3`.

Tout projet non cité reçoit un **coût de pénalité** élevé (par défaut $10$) pour éviter des affectations non souhaitées.

---

## 2. Modélisation mathématique

On construit un **graphe orienté** :

- un **nœud source** $s$ ;
- un **nœud pour chaque étudiant** $e_i$ ;
- un **nœud pour chaque projet** $p_j$ ;
- un **nœud puits** $t$.

Les arcs sont définis comme suit :

- $s \to e_i$ avec capacité $1$ et coût $0$ ;
- $e_i \to p_j$ avec capacité $1$ et coût $c_{ij}$ (selon les préférences) ;
- $p_j \to t$ avec capacité = capacité du projet et coût $0$.

On cherche le **flot de coût minimal** qui envoie tout le flot des étudiants vers les projets :

$$
\min \sum_{i,j} c_{ij} \, x_{ij}
$$

sous les contraintes :
- chaque étudiant est affecté à **au plus un** projet ;
- la somme des flots vers chaque projet ne dépasse pas sa **capacité** ;
- les flots $x_{ij}$ sont entiers (0 ou 1 dans notre cas).

---

## 3. Application numérique : 3 étudiants × 3 projets

Nous avons trois étudiants : $S_1$, $S_2$, $S_3$  
et trois projets : $A$, $B$, $C$ (chacun de capacité $1$).

### Préférences ordonnées

| Étudiant | 1er vœu | 2e vœu | 3e vœu |
|:--:|:--:|:--:|:--:|
| S1 | A | B | C |
| S2 | B | C | A |
| S3 | B | A | C |

Les coûts de rang sont donc :

|     | A | B | C |
|:---:|:---:|:---:|:---:|
| S1  | 0 | 1 | 2 |
| S2  | 2 | 0 | 1 |
| S3  | 1 | 0 | 2 |

---

### Construction du graphe

- Arcs **étudiant → projet** avec ces coûts ;
- Arcs **projet → puits** de capacité $1$ ;
- Arcs **source → étudiant** de capacité $1$.

Le flot total à envoyer vaut :
$$
F = 3
$$

puisqu’il y a trois étudiants à affecter.

---

### Calcul des affectations possibles

Nous devons affecter chaque étudiant à un projet distinct.  
Voici quelques combinaisons avec leurs coûts totaux :

1. $(S_1 \to A,\, S_2 \to B,\, S_3 \to C)$  
   $0 + 0 + 2 = 2$

2. $(S_1 \to A,\, S_2 \to C,\, S_3 \to B)$  
   $0 + 1 + 0 = 1$ ✅ (meilleur)

3. $(S_1 \to B,\, S_2 \to C,\, S_3 \to A)$  
   $1 + 1 + 1 = 3$

4. $(S_1 \to C,\, S_2 \to A,\, S_3 \to B)$  
   $2 + 2 + 0 = 4$

Le **coût minimal total** est donc :
$$
C_{\min} = 1
$$

---

### 🔧 Résultat optimal

L’affectation optimale est :

| Étudiant | Projet attribué | Rang | Coût |
|:--:|:--:|:--:|:--:|
| S1 | A | 1 | 0 |
| S2 | C | 2 | 1 |
| S3 | B | 1 | 0 |

---

### 📈 Indicateurs de satisfaction

- Nombre d’étudiants affectés : $3 / 3 = 100\%$  
- Nombre de non-affectés : $0$  
- Rang médian : $\tilde{r} = 1$  
- Taux de 1er vœu : $p_{top1} = \frac{2}{3} \approx 66{,}7\%$  
- Taux top-3 : $p_{top3} = 1.0 = 100\%$  

---

## 🧮 4. Variante pondérée

Si les préférences sont exprimées en **coûts explicites** (mode pondéré) :

| Étudiant | A | B | C |
|:--:|:--:|:--:|:--:|
| S1 | 0 | 1 | 3 |
| S2 | 3 | 0 | 1 |
| S3 | 2 | 0 | 3 |

Les arcs portent désormais ces coûts exacts (et une pénalité $10$ pour tout projet absent).  
Le problème reste identique : minimiser la

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

---

## Exécution


### Mode notebook
```bash
jupyter notebook src/Assignment-Project_Hungarian-Method.ipynb
```

[Assignment-Project_Hungarian-Method.ipynb](src/Assignment-Project_Hungarian-Method.ipynb)


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
