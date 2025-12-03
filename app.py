import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Projet Big Data Finance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre Principal
st.title("📊 Dashboard Financier & Big Data")
st.markdown("""
*Ce tableau de bord analyse la performance, la volatilité et les corrélations d'actifs financiers 
en utilisant une architecture Python 'Full Code' pour traiter les données volumineuses.*
""")
st.markdown("---")

# --- 2. CHARGEMENT DES DONNÉES (ETL) ---
@st.cache_data # Optimisation : garde les données en mémoire (cache)
def load_data():
    try:
        # On charge le CSV généré par le script de récupération
        df = pd.read_csv('market_data_bigdata.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("⚠️ ERREUR : Le fichier 'market_data_bigdata.csv' est introuvable.")
    st.info("Veuillez lancer le script de récupération des données (step 1) avant de lancer l'application.")
    st.stop()

# --- 3. BARRE LATÉRALE (Filtres Interactifs) ---
st.sidebar.header("Paramètres d'Analyse")

# Liste complète des tickers
liste_tickers = df['Ticker'].unique()

# Sélecteur d'actif unique (pour la vue Micro)
choix_ticker = st.sidebar.selectbox("🔎 Focus sur un Actif", liste_tickers)

# Sélecteur de période (Slider de dates)
date_min = df['Date'].min()
date_max = df['Date'].max()
dates = st.sidebar.slider(
    "Période temporelle",
    min_value=date_min.date(),
    max_value=date_max.date(),
    value=(date_min.date(), date_max.date())
)

# Création du masque de filtrage
mask_ticker = (df['Ticker'] == choix_ticker) & \
              (df['Date'] >= pd.to_datetime(dates[0])) & \
              (df['Date'] <= pd.to_datetime(dates[1]))
df_filtered = df.loc[mask_ticker]

# --- 4. SECTION 1 : VUE "MICRO" (Focus Actif) ---
st.header(f"1. Analyse Technique : {choix_ticker}")

if not df_filtered.empty:
    # Calcul des KPIs dynamiques sur la période filtrée
    last_close = df_filtered.iloc[-1]['Close']
    prev_close = df_filtered.iloc[-2]['Close'] if len(df_filtered) > 1 else last_close
    variation = ((last_close - prev_close) / prev_close) * 100
    
    avg_volatility = df_filtered['Volatilite'].mean()
    total_volume = df_filtered['Volume'].sum()

    # Affichage des KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Dernier Cours", f"{last_close:.2f} $", f"{variation:.2f} %")
    kpi2.metric("Volatilité Annualisée (Risque)", f"{avg_volatility*100:.2f} %")
    kpi3.metric("Volume Total (Liquidité)", f"{total_volume:,.0f}")

    # --- GRAPHIQUE PRINCIPAL (Bougies + Moyennes Mobiles) ---
    fig = go.Figure()

    # Bougies (Candlestick)
    fig.add_trace(go.Candlestick(
        x=df_filtered['Date'],
        open=df_filtered['Open'], high=df_filtered['High'],
        low=df_filtered['Low'], close=df_filtered['Close'],
        name='Prix'
    ))

    # Moyennes Mobiles (Feature Engineering)
    fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['SMA_50'], 
                             line=dict(color='orange', width=1.5), name='Moyenne Mobile 50j'))
    fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['SMA_200'], 
                             line=dict(color='#00CC96', width=1.5), name='Moyenne Mobile 200j'))

    fig.update_layout(
        title=f"Évolution du cours - {choix_ticker}",
        yaxis_title="Prix ($)",
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False 
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- GRAPHIQUES SECONDAIRES ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volumes Échangés")
        fig_vol = px.bar(df_filtered, x='Date', y='Volume', template="plotly_dark", 
                         color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col2:
        st.subheader("Distribution des Rendements")
        fig_hist = px.histogram(df_filtered, x="Daily_Return", nbins=40, 
                                title="Fréquence des gains/pertes",
                                template="plotly_dark", color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.warning("Données insuffisantes pour la période sélectionnée.")

st.markdown("---")

# --- 5. SECTION 2 : VUE "MACRO" BIG DATA (Innovation) ---
st.header("2. Analyse de Marché Globale (Vision Big Data)")
st.info("💡 Cette section agrège l'ensemble des données historiques pour comparer tous les actifs simultanément.")

# Préparation des données agrégées pour la matrice
summary_data = []

# Filtrage global sur la plage de date (mais sur TOUS les tickers)
df_period = df[(df['Date'] >= pd.to_datetime(dates[0])) & (df['Date'] <= pd.to_datetime(dates[1]))]

for ticker in liste_tickers:
    subset = df_period[df_period['Ticker'] == ticker]
    if not subset.empty:
        # Performance globale sur la période
        start = subset.iloc[0]['Close']
        end = subset.iloc[-1]['Close']
        perf = ((end - start) / start) * 100
        
        # Moyennes
        vol = subset['Volatilite'].mean()
        vol_moyen = subset['Volume'].mean()
        
        # Catégorisation simple pour la couleur
        if "-USD" in ticker: cat = "Crypto"
        elif ".PA" in ticker: cat = "Europe"
        else: cat = "US Tech/Indus"
        
        summary_data.append({
            'Ticker': ticker,
            'Performance (%)': perf,
            'Risque (Volatilité)': vol,
            'Volume Moyen': vol_moyen,
            'Catégorie': cat
        })

df_summary = pd.DataFrame(summary_data)

# COLONNES POUR LA VISUALISATION AVANCÉE
col_matrix, col_corr = st.columns([1.5, 1]) # La matrice prend plus de place

with col_matrix:
    st.subheader("🔮 Matrice Risque / Rendement")
    st.markdown("Objectif : Identifier les actifs à **Fort Rendement** et **Faible Risque** (Haut-Gauche).")
    
    if not df_summary.empty:
        fig_bubble = px.scatter(
            df_summary,
            x="Risque (Volatilité)",
            y="Performance (%)",
            size="Volume Moyen",
            color="Catégorie",
            hover_name="Ticker",
            text="Ticker",
            template="plotly_dark",
            size_max=50,
            height=550,
            labels={"Risque (Volatilité)": "Volatilité Annualisée (Risque)"}
        )
        # Lignes médianes pour faire un cadran
        fig_bubble.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig_bubble.add_vline(x=df_summary['Risque (Volatilité)'].mean(), line_dash="dash", line_color="white", opacity=0.3)
        
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.write("Pas de données suffisantes pour la matrice.")

with col_corr:
    st.subheader("🔗 Corrélations")
    st.markdown("Analyse des liens entre actifs (Diversification).")
    
    # Pivot pour corrélation
    if not df_period.empty:
        # On limite aux 15 premiers tickers pour la lisibilité du heatmap si trop de données
        top_tickers = liste_tickers[:15] 
        # CORRECTION MAJEURE : Corrélation sur les RENDEMENTS (Daily_Return) et non les PRIX (Close)
        df_pivot = df_period[df_period['Ticker'].isin(top_tickers)].pivot_table(index='Date', columns='Ticker', values='Daily_Return')
        
        corr_matrix = df_pivot.corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=False, # Mettre True si on veut voir les chiffres
            aspect="auto",
            color_continuous_scale='RdBu_r',
            template="plotly_dark",
            height=550
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# --- 6. DONNÉES BRUTES (Preuve de rigueur) ---
with st.expander("📂 Voir l'extrait des données brutes (Dataframe)"):
    st.dataframe(df_filtered.head(500))

# --- SECTION 3 : SIMULATION MONTE CARLO (Prédictif) ---
import numpy as np # Nécessaire pour les calculs mathématiques

st.markdown("---")
st.header("3. Simulation Prédictive (Monte Carlo)")
st.info("🤖 Cette section simule 50 scénarios futurs possibles basés sur la volatilité historique (Méthode stochastique).")

if not df_filtered.empty:
    # Paramètres de la simulation
    days_forecast = 180  # Prévision sur 6 mois
    num_simulations = 50 # Nombre de scénarios
    
    last_close = df_filtered.iloc[-1]['Close']
    # Calcul des rendements quotidiens logarithmiques (plus précis pour la finance)
    log_returns = np.log(df_filtered['Close'] / df_filtered['Close'].shift(1)).dropna()
    
    u = log_returns.mean() # Moyenne
    var = log_returns.var() # Variance
    drift = u - (0.5 * var) # Dérive
    stdev = log_returns.std() # Écart-type
    
    # Génération des simulations
    simulations = np.zeros((days_forecast, num_simulations))
    
    for i in range(num_simulations):
        # Génère des chocs aléatoires
        random_shocks = np.random.normal(0, 1, days_forecast) 
        daily_returns = np.exp(drift + stdev * random_shocks)
        
        price_path = np.zeros(days_forecast)
        price_path[0] = last_close
        
        for t in range(1, days_forecast):
            price_path[t] = price_path[t-1] * daily_returns[t]
            
        simulations[:, i] = price_path

    # Création du graphique Spaghetti
    fig_monte = go.Figure()
    
    # Axe des dates futures
    future_dates = pd.date_range(start=df_filtered['Date'].max(), periods=days_forecast)
    
    for i in range(num_simulations):
        fig_monte.add_trace(go.Scatter(
            x=future_dates, 
            y=simulations[:, i],
            mode='lines',
            line=dict(width=1),
            opacity=0.3, # Transparence pour voir la densité
            showlegend=False,
            name=f"Simu {i}"
        ))

    # Ajout du prix actuel en point de départ
    fig_monte.add_trace(go.Scatter(
        x=[df_filtered['Date'].max()], 
        y=[last_close],
        mode='markers',
        marker=dict(color='white', size=10),
        name='Prix Actuel'
    ))

    fig_monte.update_layout(
        title=f"Projection des 6 prochains mois pour {choix_ticker}",
        template="plotly_dark",
        yaxis_title="Prix Projeté ($)",
        height=500
    )
    
    st.plotly_chart(fig_monte, use_container_width=True)
    st.caption("Note : Plus les lignes s'écartent, plus l'incertitude (le risque) est grande.")