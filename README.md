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
