# picturesRC

Redressement automatique de l'horizon et rognage de photos. Application de bureau
**Windows 11 / Debian**, **100 % hors ligne** (aucun modèle téléchargé, aucun accès réseau).

## Installation (développement)

Python 3.11 ou plus récent.

**Windows 11** (PowerShell, à la racine du projet) :

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

**Debian** :

```bash
sudo apt install python3-venv libxcb-cursor0   # libxcb-cursor0 : requis par Qt 6 sous X11
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Dans PyCharm, sélectionnez `.venv` comme interpréteur du projet.

> N'installez **pas** `opencv-python` en plus de `opencv-python-headless` : le Qt
> embarqué par `opencv-python` entre en conflit avec PySide6.

## Lancement

```bash
picturesrc              # interface graphique
python -m picturesrc    # équivalent
picturesrc-cli --help   # traitement par lot en ligne de commande
```

## Utilisation de l'interface

1. **Ajouter un dossier** (Ctrl+O), ou glisser-déposer photos ou dossiers dans la fenêtre.
   L'analyse se fait en tâche de fond.
2. Chaque miniature porte une pastille de confiance :
   🟢 élevée · 🟠 moyenne · 🔴 faible. Les 🟢 sont **validées d'office** (option de la barre d'outils).
3. Ne revoir que le reste :
   - **tracer à la souris** une ligne le long de l'horizon ou d'une verticale : l'angle est corrigé aussitôt ;
   - ou ajuster au clavier.
4. Si besoin, **ajuster le cadre de rognage** (voir plus bas).
5. **Enregistrer les validées** (Ctrl+E), après confirmation : chaque photo corrigée remplace
   l'original dans son dossier, sous le même nom, et l'original est **déplacé** dans le dossier des
   originaux. Les photos sans correction ne sont ni modifiées ni déplacées.

| Touche | Action |
|---|---|
| ← / → | ±0,1° |
| Maj + ← / → | ±1° |
| Ctrl + ← / → | 90° à gauche / à droite (boutons ↺ 90° et ↻ 90°) |
| ↑ / ↓ | photo précédente / suivante |
| Entrée | valider et passer à la suivante non validée |
| Espace | valider / dévalider |
| A | revenir à l'angle automatique |
| R | remettre à 0° |
| O | afficher l'original (avant / après) |
| G | afficher / masquer la grille |
| C | revenir au cadre de rognage automatique |
| Échap | annuler le cadre en deux clics commencé |

Modes de rognage (cadre proposé automatiquement) :
- **autour de l'objet** : par défaut ; détecte l'objet posé au centre (scellé, appareil…) et retire le
  fond autour (bureau, table), avec une petite marge (voir plus bas) ;
- **surface maximale** : garde le plus de pixels, les proportions peuvent changer ;
- **proportions d'origine** : une photo 3:2 reste en 3:2 ;
- **sans rognage** : les coins pivotés restent noirs (le cadre n'est alors ni affiché ni modifiable).

### Rognage autour de l'objet

Pensé pour les photos d'un objet posé au centre sur un fond assez uniforme. Le code est dans
`src/picturesrc/core/subject.py` :

1. La couleur du fond est apprise sur une bande le long des bords de la photo, redressée.
2. Les pixels qui s'en écartent forment l'objet. Les trous sont bouchés (zones transparentes d'un sachet).
3. On garde la zone la plus grande et la plus centrale : un passe-câble ou un objet au bord est ignoré.
4. Les excroissances sont retirées, par exemple un reflet sur le bureau collé au sachet.
5. Le cadre entoure l'objet redressé, avec 2 % de marge, sans jamais de coin vide.

**Redressement d'après l'objet.** Dans ce mode, l'angle est mesuré sur les **bords de l'objet**
plutôt que sur les lignes de toute la photo. Un appareil posé de travers dans un sachet, ou une
étiquette collée en biais, ne fausse donc plus le redressement. Chaque côté de l'objet est relevé,
puis ajusté par une droite robuste qui ignore les plis et bosses du plastique. On retient les côtés
qui concordent à 1° près. Si le contour n'est pas exploitable (pas d'objet net, objet coupé par le
bord de la photo…), l'estimation par les lignes reprend la main. L'interface indique la méthode
utilisée (« auto (contour de l'objet) » ou « auto (lignes) »). Changer de mode de rognage
recalcule l'angle automatique, sauf pour les photos retouchées à la main.

Une confiance est calculée. Elle baisse si l'objet touche le bord de la photo, s'il est minuscule
ou s'il occupe toute l'image, si sa forme est peu compacte ou si le contraste avec le fond est faible.
La pastille de la miniature affiche la plus faible des deux confiances (angle et objet).
Seules les photos sûres pour les deux sont validées d'office. Si aucun objet n'est trouvé,
le cadre « surface maximale » est utilisé et la photo est signalée à revoir.

Dans ce mode, une photo est rognée même à 0° : elle est donc réencodée à l'enregistrement.
En ligne de commande : `--crop subject` (défaut).

### Ajuster le cadre de rognage

Le cadre automatique (blanc) peut être modifié photo par photo ; il devient alors orange,
et la miniature porte le signe ✂.

| Geste (bouton gauche) | Action |
|---|---|
| **deux clics** (sans glisser) | 1er clic : coin supérieur gauche · 2e clic : coin inférieur droit ; le cadre s'applique aussitôt |
| glisser un bord ou un coin du cadre | redimensionner |
| Ctrl + glisser dans le cadre | déplacer |
| Maj + glisser | tracer un nouveau cadre |
| glisser ailleurs | tracer l'horizon (inchangé) |

Cadre en deux clics : après le premier clic, une cible marque le coin posé et un cadre en
pointillés suit la souris. **Échap** annule. Les coins peuvent aussi être posés dans l'ordre
inverse. Deux clics trop proches (moins de 15 px à l'écran) sont ignorés.

Le cadre ne peut jamais englober un coin vide : il bute contre le bord de la zone utile
de l'image pivotée et glisse le long de celui-ci. Les proportions sont libres. Si l'on change
ensuite l'angle, un cadre devenu trop grand est réduit au plus près de celui choisi.
Un cadre ajusté s'applique aussi à une photo laissée à 0° (qui est alors réencodée).

## Enregistrement et traçabilité

- Les photos corrigées restent dans leur **dossier source**, sous leur nom d'origine.
- Les **originaux** de ces photos sont **déplacés**, sans aucune modification :
  - par défaut dans un sous-dossier `originaux/` du dossier de chaque photo ;
  - ou dans un dossier unique choisi avec « Dossier des originaux… » (`--originals` en ligne de commande).
  Si un nom y existe déjà, un suffixe `_1`, `_2`… est ajouté : rien n'est jamais écrasé.
- Les photos sans correction (0°, pas de cadre, pas de quart de tour) restent telles quelles.
- Ordre des opérations, sûr en cas d'erreur (fichier ouvert dans une autre application, disque plein…) :
  1. la correction est écrite dans un fichier temporaire du dossier source ;
  2. l'original est déplacé (simple renommage sur le même disque ; sur un autre disque,
     copie contrôlée par SHA-256 puis suppression) ;
  3. le fichier temporaire prend le nom de l'original.
  En cas d'échec, l'original est remis à sa place et le fichier temporaire supprimé.
- Ré-enregistrer une photo déjà traitée dans la même session recalcule la correction **depuis
  l'original déplacé** (jamais de double correction). Si la photo n'a plus de correction,
  l'original est remis à sa place.
- L'orientation EXIF est appliquée. Les métadonnées EXIF (Orientation remise à 1) et le profil ICC sont conservés.
- **Optionnel** (case « Fichier JSON de traçabilité », ou `--no-json` en ligne de commande pour le désactiver) :
  un fichier `<nom>.picturesrc.json` est écrit **à côté de l'original déplacé** (le dossier
  source ne contient que les photos). Il contient :
  - les chemins et SHA-256 de l'original (déplacé) et de la photo corrigée ;
  - le quart de tour éventuel (90° à gauche, à droite, 180°), l'angle appliqué, le mode de rognage (`objet`, `manuel` si le cadre a été ajusté…) et la boîte de rognage ;
  - en mode objet, la confiance de la détection ;
  - l'estimation automatique, sa méthode (`lignes` ou `contour`) et sa confiance ;
  - un indicateur de correction manuelle.

## Comment fonctionne la détection

Le code est dans `src/picturesrc/core/straighten.py`. En mode « autour de l'objet », c'est le
contour de l'objet qui sert (voir plus haut).

1. L'image est réduite à environ 1000 px et légèrement floutée.
2. Les contours sont extraits par Canny, avec des seuils dérivés d'Otsu.
3. Les segments de droite sont détectés par une transformée de Hough probabiliste.
4. Chaque segment est **réajusté au sous-pixel** sur le maximum du gradient. Cette étape supprime le biais vers 0° des petites inclinaisons.
5. Seuls les segments proches de l'horizontale ou de la verticale sont gardés (±15°). Chacun vote, pondéré par sa longueur.
6. Le pic d'une densité à noyau donne l'angle. La part des votes concordants donne la confiance.

Sur les scènes de test, l'erreur est inférieure à 0,1°. Le calcul prend environ 60 ms par photo.

Convention d'angle : rotation à appliquer, en degrés, **positive = antihoraire**
(même convention que `PIL.Image.rotate` et `cv2.getRotationMatrix2D`).

## Tests

```bash
pytest
```

## Application autonome (sans Python installé)

```bash
python packaging/build.py
```

Le résultat se trouve dans `dist/picturesRC/`. Le paquet se construit **sur chaque OS cible**
(PyInstaller ne fait pas de compilation croisée).

## Limites connues (v0.1)

- Les images 16 bits sont converties en 8 bits. Les formats RAW et HEIC ne sont pas pris en charge.
- Photos sans aucune ligne droite (paysage naturel sans horizon net) : la confiance est faible et la correction est manuelle (un tracé suffit).
- Les verticales fortement convergentes (bâtiment pris en contre-plongée) peuvent perturber l'estimation.
- Rognage autour de l'objet : un fond très chargé ou de la même couleur que l'objet le met en échec. La confiance est alors faible et la photo est à revoir.
