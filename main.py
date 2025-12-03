import yfinance as yf
import pandas as pd
import datetime

# 1. LISTE DES ACTIFS (Mix Actions & Crypto pour la variété)
# Pour le "Big Data", on en met une liste représentative.
tickers_list = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', # Tech
    'LVMH.PA', 'OR.PA', 'TTE.PA', 'BNP.PA', 'AIR.PA', # CAC 40 (France)
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', # Crypto
    'JPM', 'V', 'JNJ', 'WMT', 'PG' # Finance & Retail US
]

print(f"Récupération des données pour {len(tickers_list)} actifs...")

appended_data = []

for symbol in tickers_list:
    try:
        # 2. TÉLÉCHARGEMENT (5 ans d'historique)
        df = yf.download(symbol, period="5y", interval="1d", progress=False)
        
        if len(df) > 0:
            # Nettoyage des MultiIndex (souvent un problème avec yfinance)
            df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
            
            # 3. FEATURE ENGINEERING (Création des KPIs en Python pour soulager Power BI)
            # Calcul des Moyennes Mobiles (Tendances)
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Calcul du Rendement Journalier (%)
            df['Daily_Return'] = df['Close'].pct_change()
            
            # Calcul de la Volatilité (Écart type annualisé des rendements)
            # On multiplie par racine de 252 (jours de bourse) pour annualiser
            df['Volatilite'] = df['Daily_Return'].rolling(window=20).std() * (252 ** 0.5)
            
            # Ajout de la colonne Ticker (essentiel pour Power BI)
            df['Ticker'] = symbol
            
            # On garde la date comme colonne normale
            df.reset_index(inplace=True)
            
            appended_data.append(df)
            print(f"Chargé : {symbol}")
            
    except Exception as e:
        print(f"Erreur sur {symbol}: {e}")

# 4. CONSOLIDATION & EXPORT
if appended_data:
    final_df = pd.concat(appended_data)
    
    # Nettoyage final (suppression des lignes vides dues aux calculs de moyennes mobiles)
    final_df.dropna(inplace=True)
    
    # Export vers CSV pour Power BI
    output_file = 'market_data_bigdata.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nSuccès ! Fichier '{output_file}' généré avec {len(final_df)} lignes.")
else:
    print("Aucune donnée récupérée.")