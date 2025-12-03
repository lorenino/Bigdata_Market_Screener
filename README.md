# 📂 Dossier de Projet : "Market Screener" Big Data

## 1. Contexte & Objectif (Scénario "Market Screener")
**Le Pitch** : J'agis en tant qu'analyste pour un fonds d'investissement. Face à l'impossibilité de surveiller manuellement 500+ actifs simultanément, j'ai développé un outil de **screening automatisé** pour détecter les opportunités d'achat et gérer le risque.

**Objectif Technique** : Concevoir une architecture capable d'ingérer, traiter et visualiser un grand volume de données financières (S&P 500, CAC 40, Cryptos) en temps réel, là où les outils classiques (Excel, Power BI) montrent leurs limites.

## 2. Architecture Technique "Full Code" (Le Choix du Big Data)
J'ai opté pour une stack 100% Python, open-source et scalable.

### Pourquoi pas Power BI ?
*   **Volume & Vélocité** : Power BI est excellent pour l'agrégation simple, mais très lent pour les calculs financiers complexes sur des séries temporelles (Time-Series) de millions de lignes.
*   **Complexité des Calculs** : Calculer une Moyenne Mobile Exponentielle ou une Volatilité Glissante en langage DAX est lourd et peu performant.
*   **Scalabilité** : Mon script Python est conçu pour passer de 50 à 5000 actifs sans changer une ligne de code, en se connectant potentiellement à des clusters de calcul (Spark).

### Le Pipeline de Données (ETL)
1.  **Extract (Source)** : API `yfinance` (Yahoo Finance). Données brutes de marché (Open, High, Low, Close, Volume).
2.  **Transform (Python/Pandas)** :
    *   Nettoyage des données (valeurs manquantes, jours fériés).
    *   **Feature Engineering** : Calcul vectorisé des indicateurs techniques (SMA 50/200, Volatilité Annualisée, Rendements Logarithmiques).
3.  **Load & Viz (Streamlit)** : Interface Web interactive pour l'utilisateur final.

## 3. Méthodologie Statistique
Pour garantir la pertinence de l'analyse, j'ai corrigé certains biais statistiques fréquents.

### A. La Volatilité (Risque Réel)
*   *Avant* : Écart-type du Prix. (Biais : Une action à 2000$ semble plus risquée qu'une à 10$).
*   *Après* : **Volatilité Annualisée des Rendements**. (Permet de comparer le risque du Bitcoin vs Coca-Cola sur une échelle commune en %).

### B. Les Corrélations (Diversification)
*   *Avant* : Corrélation des Prix. (Biais : "Spurious Correlation" - tout monte sur 5 ans, donc tout semble corrélé).
*   *Après* : **Corrélation des Rendements Quotidiens**. (Mesure la vraie contagion du risque : si A chute aujourd'hui, est-ce que B chute aussi ?).

## 4. Fonctionnalités du Dashboard

### 🔍 Vue Micro : Analyse Technique
Pour le trader qui doit valider un point d'entrée.
*   **Golden Cross** : Visualisation des croisements de Moyennes Mobiles (SMA 50 vs SMA 200) pour identifier les changements de tendance.
*   **Chandelier Japonais** : Graphique de précision pour voir la psychologie du marché (Open/Close).

### 🌍 Vue Macro : Matrice "Risk / Reward"
Pour le gestionnaire de portefeuille qui alloue le capital.
*   **Concept** : Un Scatter Plot dynamique comparant tous les actifs.
    *   *Axe Y (Rendement)* vs *Axe X (Risque)*.
    *   *Objectif* : Identifier les actifs présentant le meilleur ratio rendement/risque (Haut-Gauche) et ceux à éviter (Bas-Droite).
*   **Heatmap de Corrélation** : Outil de gestion du risque systémique (éviter d'avoir un portefeuille où tout s'effondre en même temps).

### 🔮 Prédictif : Simulation Monte Carlo
Pour le Risk Manager qui doit anticiper le futur.
*   **Méthode** : Simulation stochastique (Mouvement Brownien Géométrique).
*   **Résultat** : Génération de **50 scénarios futurs** possibles sur 6 mois.
*   **Apport** : Visualise le "Cône d'Incertitude". Le passé ne prédit pas le futur, mais la volatilité passée permet de modéliser l'étendue des risques futurs.
