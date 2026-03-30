import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time
from math import pi
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="PFE - Segmentation Client | EST Fès",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #3498db;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: #7f8c8d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('clients_shopping_dataset.csv')
    return df

df = load_data()

# Charger vos vrais clusters
try:
    df_final = pd.read_csv('listes_marketing/resultat_final_pfe.csv')
    if 'Cluster' in df_final.columns:
        df['Cluster'] = df_final['Cluster']
except:
    df_encoded = df.copy()
    df_encoded['Gender'] = df_encoded['Gender'].map({'Female': 0, 'Male': 1})
    df_encoded['Loyalty_Program'] = df_encoded['Loyalty_Program'].map({'No': 0, 'Yes': 1})
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    features = ['Age', 'Annual_Income_k$', 'Spending_Score', 'Online_Shopping_Ratio']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded[features])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

# Noms des clusters
cluster_names = {
    0: "ÉCONOMES PRAGMATIQUES",
    1: "ACHETEURS WEB",
    2: "VIP TRADITIONNELS",
    3: "FIDÈLES DE PROXIMITÉ"
}

cluster_colors = {
    0: "#FF6B6B",
    1: "#4ECDC4",
    2: "#45B7D1",
    3: "#96CEB4"
}

# Statistiques par cluster
stats_clusters = {}
for i in range(4):
    data = df[df['Cluster'] == i]
    stats_clusters[i] = {
        'age': data['Age'].mean(),
        'revenu': data['Annual_Income_k$'].mean(),
        'score': data['Spending_Score'].mean(),
        'ratio': data['Online_Shopping_Ratio'].mean(),
        'effectif': len(data),
        'pourcentage': len(data)/len(df)*100
    }

# Stratégies marketing
cluster_strategies = {
    0: {
        "strategie": "Conversion",
        "actions": "Coupons de réduction, offres promotionnelles, emailings",
        "canaux": "Email, SMS, notifications push",
        "offres": ["-20% sur première commande", "Cadeau de bienvenue", "Programme de fidélité"],
        "timing": "Dans les 24h suivant l'inscription"
    },
    1: {
        "strategie": "Campagnes digitales",
        "actions": "Publicité ciblée, retargeting, notifications push",
        "canaux": "Instagram, TikTok, Facebook, Google Ads",
        "offres": ["Livraison gratuite", "Code promo 15%", "Exclusivités web"],
        "timing": "Campagne de retargeting J+3"
    },
    2: {
        "strategie": "Rétention premium",
        "actions": "Ventes privées, accès VIP, service conciergerie",
        "canaux": "Email personnalisé, appel téléphonique, événements exclusifs",
        "offres": ["Accès VIP", "Avant-premières", "Service client dédié"],
        "timing": "Programme de fidélité annuel"
    },
    3: {
        "strategie": "Prescription",
        "actions": "Programme de parrainage, événements en boutique",
        "canaux": "Bouche-à-oreille, WhatsApp, événements physiques",
        "offres": ["Parrainage : 20€ offerts", "Événements privés", "Cadeaux exclusifs"],
        "timing": "Invitation au prochain événement"
    }
}

# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.markdown("### 🎓 EST Fès")
st.sidebar.markdown("---")

theme = st.sidebar.selectbox("🎨 Thème", ["Clair", "Sombre"])

if theme == "Sombre":
    st.markdown("""
    <style>
        .stApp { background-color: #1e1e2f; }
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #ffffff; }
        .stMetric label, .stMetric .stMetric-value, .stMetric .stMetric-delta { color: #ffffff; }
        .st-emotion-cache-1y4p8pa { background-color: #2d2d3a; }
        .st-emotion-cache-16idsys p { color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Accueil", "🔍 Exploration", "📈 Régression", "🎯 Clustering",
     "💼 Personas", "🤖 Random Forest", "🎯 Simulateur", "📊 Comparaison",
     "💬 Chatbot", "ℹ️ À propos", "📊 Export"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Auteurs :**\n"
    "- Hatim LAHMAR\n"
    "- Moumen YOUSSEF\n\n"
    "**Encadrant :**\n"
    "- Pr. Faiq GMIRA\n\n"
    "**Filière :** Ingénierie des Données"
)

# ==========================================
# PAGE 1 : ACCUEIL
# ==========================================
if menu == "🏠 Accueil":
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 2rem;">
        <h1 style="color: white;">📊 Segmentation Clientèle</h1>
        <p style="color: white; font-size: 1.2rem;">Par l'intelligence artificielle et le Machine Learning</p>
        <p style="color: white;">Projet de Fin d'Études - EST Fès</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Clients", f"{len(df):,}")
    with col2:
        st.metric("📊 Segments", "4")
    with col3:
        st.metric("🎯 Précision RF", "78.7%")
    with col4:
        st.metric("⭐ Score silhouette", "0.145")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Objectif du projet")
        st.info("Aider l'équipe marketing à personnaliser ses campagnes et optimiser son retour sur investissement.")
        
        st.markdown("### 📊 Résultats obtenus")
        st.success("""
        - 4 segments clients identifiés
        - Score de silhouette: 0.1452
        - Random Forest: 78.7% de précision
        - Gain de +1.7 points par rapport à la régression logistique
        """)
    
    with col2:
        st.markdown("### 🚀 Valeur ajoutée")
        st.success("""
        - Marketing ciblé par segment
        - Réduction du gaspillage budgétaire
        - Augmentation du taux de conversion
        - Interface interactive pour l'équipe marketing
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://github.com/youssefmn06/Segmentation-Client-PFE" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white">
        </a>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 2 : EXPLORATION
# ==========================================
elif menu == "🔍 Exploration":
    st.markdown('<p class="main-header">🔍 Analyse Exploratoire des Données</p>', unsafe_allow_html=True)
    
    # Filtres dynamiques
    st.markdown('<p class="sub-header">🔍 Filtres dynamiques</p>', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        age_min, age_max = st.slider("Tranche d'âge", 18, 69, (25, 55))
    with col_f2:
        revenu_min, revenu_max = st.slider("Revenu (k$)", 10, 150, (30, 100))
    with col_f3:
        cluster_filtre = st.multiselect("Cluster", list(cluster_names.values()))
    
    df_filtre = df[(df['Age'] >= age_min) & (df['Age'] <= age_max) &
                   (df['Annual_Income_k$'] >= revenu_min) & (df['Annual_Income_k$'] <= revenu_max)]
    
    if cluster_filtre:
        df_filtre = df_filtre[df_filtre['Cluster'].map(cluster_names).isin(cluster_filtre)]
    
    st.write(f"**{len(df_filtre)}** clients correspondent aux critères")
    
    tab1, tab2, tab3 = st.tabs(["📦 Boxplots", "🔗 Matrice de corrélation", "📊 Graphiques interactifs"])
    
    with tab1:
        st.markdown('<p class="sub-header">Figure 1 : Boxplots des variables numériques</p>', unsafe_allow_html=True)
        try:
            st.image('figures/figure1_boxplots.png', use_container_width=True)
        except:
            st.warning("Figure non trouvée")
    
    with tab2:
        st.markdown('<p class="sub-header">Figure 2 : Matrice de corrélation</p>', unsafe_allow_html=True)
        try:
            st.image('figures/figure3_matrice_correlation.png', use_container_width=True)
        except:
            st.warning("Figure non trouvée")
        st.info("🔍 **Observation :** Aucune corrélation ne dépasse 0.1")
    
    with tab3:
        st.markdown('<p class="sub-header">Graphiques interactifs Plotly</p>', unsafe_allow_html=True)
        var_x = st.selectbox("Axe X", ['Age', 'Annual_Income_k$', 'Spending_Score', 'Online_Shopping_Ratio'])
        var_y = st.selectbox("Axe Y", ['Age', 'Annual_Income_k$', 'Spending_Score', 'Online_Shopping_Ratio'])
        
        fig = px.scatter(df_filtre, x=var_x, y=var_y, 
                         color=df_filtre['Cluster'].map(cluster_names),
                         color_discrete_sequence=list(cluster_colors.values()),
                         title=f"{var_x} vs {var_y}")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 3 : RÉGRESSION
# ==========================================
elif menu == "📈 Régression":
    st.markdown('<p class="main-header">📈 Régression Linéaire (Revenu vs Score)</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R²", "0.0000", delta="Aucune relation linéaire")
    with col2:
        st.metric("RMSE", "28.17", delta="Erreur élevée")
    
    try:
        st.image('figures/figure2_regression_lineaire.png', use_container_width=True)
    except:
        st.warning("Figure non trouvée")
    
    st.warning("⚠️ **Conclusion :** R² = 0.00, absence totale de relation linéaire entre le revenu et le score de dépense.")

# ==========================================
# PAGE 4 : CLUSTERING
# ==========================================
elif menu == "🎯 Clustering":
    st.markdown('<p class="main-header">🎯 Segmentation par Clustering K-Means</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score de silhouette", "0.1452", delta="K=4 optimal")
    with col2:
        st.metric("Nombre de clusters", "4")
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(
        df[df['Cluster']==0]['Age'],
        df[df['Cluster']==1]['Age'],
        df[df['Cluster']==2]['Age'],
        df[df['Cluster']==3]['Age']
    )
    st.metric("ANOVA - Différence d'âge entre clusters", f"p-value = {p_value:.4f}", 
              delta="Significatif" if p_value < 0.05 else "Non significatif")
    
    tab1, tab2, tab3 = st.tabs(["📊 Graphiques", "📋 Profils", "🎨 Visualisation 3D"])
    
    with tab1:
        try:
            st.image('figures/figure5_elbow_method.png', use_container_width=True)
            st.image('figures/figure6_silhouette.png', use_container_width=True)
            st.image('figures/figure7_clusters_scatter.png', use_container_width=True)
            st.image('figures/figure8_radar_chart.png', use_container_width=True)
        except:
            st.warning("Figures non trouvées")
    
    with tab2:
        profile_df = pd.DataFrame({
            'Cluster': [0, 1, 2, 3],
            'Nom': [cluster_names[i] for i in range(4)],
            'Âge': [stats_clusters[i]['age'] for i in range(4)],
            'Revenu (k$)': [stats_clusters[i]['revenu'] for i in range(4)],
            'Score': [stats_clusters[i]['score'] for i in range(4)],
            'Ratio Online': [stats_clusters[i]['ratio'] for i in range(4)],
            'Effectif': [stats_clusters[i]['effectif'] for i in range(4)],
            '%': [stats_clusters[i]['pourcentage'] for i in range(4)]
        })
        st.dataframe(profile_df.round(2), use_container_width=True)
    
    with tab3:
        fig = px.scatter_3d(df, x='Age', y='Annual_Income_k$', z='Spending_Score',
                            color=df['Cluster'].map(cluster_names),
                            color_discrete_sequence=list(cluster_colors.values()),
                            title="Visualisation 3D des clusters")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 5 : PERSONAS
# ==========================================
elif menu == "💼 Personas":
    st.markdown('<p class="main-header">💼 Analyse Métier et Personas</p>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            st.markdown(f"""
            <div style="background-color: {cluster_colors[i]}20; padding: 1rem; border-radius: 10px; border-left: 4px solid {cluster_colors[i]}; margin-bottom: 1rem;">
                <h4 style="color: {cluster_colors[i]};">{cluster_names[i]}</h4>
                <p><strong>Effectif :</strong> {stats_clusters[i]['effectif']:.0f} clients ({stats_clusters[i]['pourcentage']:.1f}%)</p>
                <p><strong>Âge :</strong> {stats_clusters[i]['age']:.0f} ans</p>
                <p><strong>Revenu :</strong> {stats_clusters[i]['revenu']:.0f} k$</p>
                <p><strong>Score :</strong> {stats_clusters[i]['score']:.0f}</p>
                <p><strong>Stratégie :</strong> {cluster_strategies[i]['strategie']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    try:
        st.image('figures/figure9_distributions_clusters.png', use_container_width=True)
    except:
        st.warning("Figure non trouvée")

# ==========================================
# PAGE 6 : RANDOM FOREST
# ==========================================
elif menu == "🤖 Random Forest":
    st.markdown('<p class="main-header">🤖 Classification Prédictive - Random Forest</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Random Forest", "78.7%", delta="+1.7%", delta_color="normal")
    with col2:
        st.metric("Régression Logistique", "77.0%")
    with col3:
        st.metric("Gain", "+1.7 points", delta="Amélioration")
    
    try:
        st.image('figures/figure10_confusion_rf.png', use_container_width=True)
        st.image('figures/figure11_importance_rf.png', use_container_width=True)
        st.image('figures/figure12_comparaison.png', use_container_width=True)
    except:
        st.warning("Figures non trouvées")
    
    st.success("✅ **Conclusion :** Le Random Forest surpasse la régression logistique, justifiant son choix pour le modèle final.")

# ==========================================
# PAGE 7 : SIMULATEUR IA
# ==========================================
elif menu == "🎯 Simulateur":
    st.markdown('<p class="main-header">🎯 Simulateur IA - Prédiction de Segment</p>', unsafe_allow_html=True)
    
    # Préparation du modèle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    df_train = df.copy()
    df_train['Gender'] = df_train['Gender'].map({'Female': 0, 'Male': 1})
    df_train['Loyalty_Program'] = df_train['Loyalty_Program'].map({'No': 0, 'Yes': 1})
    
    features_model = ['Age', 'Annual_Income_k$', 'Online_Shopping_Ratio', 'Gender', 
                      'Loyalty_Program', 'Visit_Frequency_per_Month', 
                      'Time_Spent_per_Visit_min', 'Days_Since_Last_Visit']
    
    scaler_model = StandardScaler()
    X_model = scaler_model.fit_transform(df_train[features_model])
    y_model = df['Cluster']
    
    rf_model_final = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf_model_final.fit(X_model, y_model)
    
    if 'historique' not in st.session_state:
        st.session_state.historique = []
    
    # Mode démonstration
    mode_demo = st.checkbox("🎬 Mode démonstration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Âge", 18, 100, 35)
        revenu = st.number_input("Revenu (k$)", 10, 200, 60)
        score = st.number_input("Score de dépense", 1, 100, 50)
        ratio_online = st.slider("Ratio achats en ligne", 0.0, 1.0, 0.5, 0.05)
    
    with col2:
        genre = st.selectbox("Genre", ["Femme", "Homme"])
        fidelite = st.selectbox("Fidélité", ["Non", "Oui"])
        frequence = st.number_input("Visites/mois", 1, 30, 10)
        temps = st.number_input("Temps passé (min)", 5, 180, 60)
        jours = st.number_input("Jours dernière visite", 0, 365, 30)
    
    if mode_demo:
        if st.button("🎲 Générer un client aléatoire"):
            age = np.random.randint(18, 70)
            revenu = np.random.randint(20, 150)
            score = np.random.randint(10, 100)
            ratio_online = np.random.uniform(0, 1)
            genre = np.random.choice(["Femme", "Homme"])
            fidelite = np.random.choice(["Non", "Oui"])
            frequence = np.random.randint(1, 30)
            temps = np.random.randint(10, 180)
            jours = np.random.randint(0, 200)
            st.success(f"Client généré : Âge={age}, Revenu={revenu}k$, Score={score}")
            st.rerun()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predire = st.button("🔮 Prédire", type="primary", use_container_width=True)
    with col_btn2:
        reset = st.button("🗑️ Réinitialiser", use_container_width=True)
    
    if reset:
        st.session_state.historique = []
        st.rerun()
    
    if predire:
        with st.spinner('🔮 Analyse en cours...'):
            time.sleep(0.5)
            
            genre_val = 1 if genre == "Homme" else 0
            fidelite_val = 1 if fidelite == "Oui" else 0
            
            nouveau_client = pd.DataFrame({
                'Age': [age], 'Annual_Income_k$': [revenu], 'Online_Shopping_Ratio': [ratio_online],
                'Gender': [genre_val], 'Loyalty_Program': [fidelite_val],
                'Visit_Frequency_per_Month': [frequence],
                'Time_Spent_per_Visit_min': [temps], 'Days_Since_Last_Visit': [jours]
            })
            
            nouveau_scaled = scaler_model.transform(nouveau_client[features_model])
            prediction = rf_model_final.predict(nouveau_scaled)[0]
            probas = rf_model_final.predict_proba(nouveau_scaled)[0]
            
            st.session_state.historique.append({
                'Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                'Âge': age, 'Revenu': revenu, 'Score': score,
                'Segment': cluster_names[prediction]
            })
            
            st.balloons()
            st.toast(f"Segment prédit : {cluster_names[prediction]}", icon="🎯")
            
            st.markdown(f"""
            <div style="background-color: {cluster_colors[prediction]}20; padding: 2rem; border-radius: 15px; text-align: center; border: 2px solid {cluster_colors[prediction]}; margin: 1rem 0;">
                <h2 style="color: {cluster_colors[prediction]};">{cluster_names[prediction]}</h2>
                <p style="font-size: 1.2rem;"><strong>Stratégie :</strong> {cluster_strategies[prediction]['strategie']}</p>
                <p><strong>Actions :</strong> {cluster_strategies[prediction]['actions']}</p>
                <p><strong>Canaux :</strong> {cluster_strategies[prediction]['canaux']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Probabilités par segment")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh([cluster_names[i] for i in range(4)], probas, color=[cluster_colors[i] for i in range(4)])
            ax.set_xlabel('Probabilité')
            ax.set_xlim(0, 1)
            st.pyplot(fig)
            
            st.markdown(f"""
            ### 🎁 Recommandations personnalisées
            
            | Catégorie | Recommandation |
            |-----------|----------------|
            | **Offres** | {cluster_strategies[prediction]['offres'][0]}<br>{cluster_strategies[prediction]['offres'][1]}<br>{cluster_strategies[prediction]['offres'][2]} |
            | **Canaux** | {cluster_strategies[prediction]['canaux']} |
            | **Timing** | {cluster_strategies[prediction]['timing']} |
            """)
            
            resultat = pd.DataFrame([{
                'Âge': age, 'Revenu': revenu, 'Score': score,
                'Ratio_online': ratio_online, 'Genre': genre, 'Fidélité': fidelite,
                'Segment': cluster_names[prediction],
                'Stratégie': cluster_strategies[prediction]['strategie']
            }])
            csv = resultat.to_csv(index=False)
            st.download_button("📥 Télécharger la prédiction", csv, f"prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
    
    if st.session_state.historique:
        with st.expander("📜 Historique des prédictions"):
            st.dataframe(pd.DataFrame(st.session_state.historique), use_container_width=True)

# ==========================================
# PAGE 8 : COMPARAISON
# ==========================================
elif menu == "📊 Comparaison":
    st.markdown('<p class="main-header">📊 Comparaison des Modèles</p>', unsafe_allow_html=True)
    
    comparaison = pd.DataFrame({
        'Critère': ['Précision', 'Interprétabilité', 'Complexité', 'Temps d\'entraînement', 'Gestion non-linéarité'],
        'Régression Logistique': ['77.0%', '⭐⭐⭐⭐⭐', '⭐⭐', '⭐⭐⭐⭐⭐', '⭐'],
        'Random Forest': ['78.7%', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐']
    })
    st.dataframe(comparaison, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ['Régression Logistique', 'Random Forest']
    scores = [77.0, 78.7]
    bars = ax.bar(models, scores, color=['#ff9999', '#66b3ff'])
    ax.set_ylabel('Précision (%)')
    ax.set_ylim(70, 85)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{score}%', ha='center')
    st.pyplot(fig)

# ==========================================
# PAGE 9 : CHATBOT
# ==========================================
elif menu == "💬 Chatbot":
    st.markdown('<p class="main-header">💬 Assistant Marketing</p>', unsafe_allow_html=True)
    st.markdown("Posez une question sur les segments clients")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Ex: Quelle stratégie pour le cluster 0 ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        prompt_lower = prompt.lower()
        if "cluster 0" in prompt_lower or "économes" in prompt_lower:
            reponse = f"**{cluster_names[0]}** : Stratégie de conversion. Actions recommandées : {cluster_strategies[0]['actions']}"
        elif "cluster 1" in prompt_lower or "acheteurs web" in prompt_lower:
            reponse = f"**{cluster_names[1]}** : Stratégie digitale. Actions recommandées : {cluster_strategies[1]['actions']}"
        elif "cluster 2" in prompt_lower or "vip" in prompt_lower:
            reponse = f"**{cluster_names[2]}** : Stratégie de rétention premium. Actions recommandées : {cluster_strategies[2]['actions']}"
        elif "cluster 3" in prompt_lower or "fidèles" in prompt_lower:
            reponse = f"**{cluster_names[3]}** : Stratégie de prescription. Actions recommandées : {cluster_strategies[3]['actions']}"
        elif "score" in prompt_lower:
            reponse = "Le score de silhouette est de 0.1452, indiquant une qualité de clustering acceptable."
        elif "precision" in prompt_lower or "performance" in prompt_lower:
            reponse = "Random Forest : 78.7% | Régression Logistique : 77.0% | Gain : +1.7 points"
        else:
            reponse = "Je peux vous aider avec : les stratégies des clusters, les performances des modèles, ou les caractéristiques des segments."
        
        st.session_state.messages.append({"role": "assistant", "content": reponse})
        st.chat_message("assistant").write(reponse)

# ==========================================
# PAGE 10 : À PROPOS
# ==========================================
elif menu == "ℹ️ À propos":
    st.markdown('<p class="main-header">ℹ️ À propos du projet</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎓 Projet de Fin d'Études
        **Titre :** Segmentation Clientèle par Clustering
        **Établissement :** EST Fès
        **Filière :** Ingénierie des Données
        **Année :** 2026
        """)
    
    with col2:
        st.markdown("""
        ### 👥 Équipe
        | Rôle | Nom |
        |------|-----|
        | **Auteur** | Hatim LAHMAR |
        | **Auteur** | Moumen YOUSSEF |
        | **Encadrant** | Pr. Faiq GMIRA |
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Technologies utilisées
    
    | Technologie | Utilisation |
    |-------------|-------------|
    | Python 3.10 | Langage principal |
    | Pandas | Manipulation des données |
    | Scikit-learn | Machine Learning |
    | Matplotlib / Seaborn | Visualisation |
    | Plotly | Graphiques interactifs |
    | Streamlit | Interface web |
    | GitHub | Versionnement |
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Objectif du projet
    Développer un système de segmentation client basé sur l'apprentissage automatique pour aider l'équipe marketing à personnaliser ses campagnes.
    
    ### 📈 Résultats clés
    - **4 segments clients** identifiés
    - **Score de silhouette : 0.1452**
    - **Random Forest : 78.7%** de précision
    - **Gain : +1.7 points** par rapport à la régression logistique
    """)
    
    st.info("Le code source complet est disponible sur GitHub : [Segmentation-Client-PFE](https://github.com/youssefmn06/Segmentation-Client-PFE)")

# ==========================================
# PAGE 11 : EXPORT
# ==========================================
elif menu == "📊 Export":
    st.markdown('<p class="main-header">📊 Export des Résultats</p>', unsafe_allow_html=True)
    
    st.markdown("### 📁 Télécharger les listes marketing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment = st.selectbox("Choisir un segment", list(cluster_names.keys()), 
                               format_func=lambda x: f"Cluster {x} : {cluster_names[x]}")
        
        if st.button("📥 Télécharger la liste"):
            clients_segment = df[df['Cluster'] == segment][['ClientID', 'Age', 'Annual_Income_k$', 'Spending_Score']]
            csv = clients_segment.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"clients_cluster_{segment}.csv",
                mime="text/csv"
            )
    
    with col2:
        rapport = f"""
========================================
RAPPORT SEGMENTATION CLIENT
========================================

Date: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
Total clients: {len(df)}

========================================
SEGMENTS IDENTIFIÉS
========================================

"""
        for i in range(4):
            rapport += f"""
Cluster {i} - {cluster_names[i]}:
  - Effectif: {stats_clusters[i]['effectif']:.0f} clients ({stats_clusters[i]['pourcentage']:.1f}%)
  - Âge moyen: {stats_clusters[i]['age']:.0f} ans
  - Revenu moyen: {stats_clusters[i]['revenu']:.0f} k$
  - Score moyen: {stats_clusters[i]['score']:.0f}
  - Stratégie: {cluster_strategies[i]['strategie']}
"""
        
        rapport += """
========================================
PERFORMANCES DES MODÈLES
========================================
- Régression Linéaire (R²): 0.0000
- Régression Logistique: 77.0%
- Random Forest: 78.7%
- Score de silhouette: 0.1452
"""
        
        st.download_button(
            label="📥 Télécharger rapport complet",
            data=rapport,
            file_name=f"rapport_segmentation_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    st.markdown("---")
    st.markdown("""
    ### 📁 Fichiers disponibles
    
    | Dossier | Contenu |
    |---------|---------|
    | `figures/` | 13 figures PNG |
    | `tableaux/` | 11 tableaux CSV |
    | `listes_marketing/` | 4 listes clients |
    """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <p>Projet de Fin d'Études - EST Fès | Segmentation Clientèle par Clustering | 2026</p>
</div>
""", unsafe_allow_html=True)