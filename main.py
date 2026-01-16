import streamlit as st
import pandas as pd
import pickle

# ======================================================
# CONFIGURATION PAGE
# ======================================================
st.set_page_config(
    page_title="Estimation Immobilière",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Estimation du prix immobilier")
st.markdown(
    """
    Cette application permet d'estimer le **prix d’un bien immobilier**
    à partir d’un modèle de **Random Forest entraîné** sur des données historiques.
    """
)

st.divider()

# ======================================================
# CHARGEMENT DES OBJETS
# ======================================================
@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        return pickle.load(f)

#@st.cache_resource
#def load_prix_m2_cp():
#    with open("prix_m2_cp.pkl", "rb") as f:
#        return pickle.load(f)

saved = load_model()
prix_m2_cp = {}#load_prix_m2_cp()

rf_model = saved["model"]
scaler = saved["scaler"]
numeric_cols = saved["numeric_cols"]

# ======================================================
# FORMULAIRE UTILISATEUR
# ======================================================
st.subheader("📋 Caractéristiques du bien")

with st.form("estimation_form"):

    surface = st.slider("Surface totale (m²)", 10, 300, 80)
    nb_pieces = st.slider("Nombre de pièces", 1, 10, 3)
    nb_chambres = st.slider("Nombre de chambres", 0, 8, 2)

    col1, col2 = st.columns(2)
    with col1:
        nb_sdb = st.slider("Salles de bain", 0, 4, 1)
    with col2:
        nb_wc = st.slider("WC", 0, 3, 1)

    charges = st.number_input("Charges annuelles (€)", 0, 30000, 1200)

    col3, col4 = st.columns(2)
    with col3:
        cave = st.checkbox("Cave")
    with col4:
        parking = st.checkbox("Parking")

    st.subheader("⚡ Performance énergétique")

    classe_energie = st.selectbox("Classe énergétique", [1, 2, 3, 4, 5, 6, 7])
    conso = st.slider("Consommation (kWh/m²/an)", 50, 500, 150)

    classe_co2 = st.selectbox("Classe émission CO₂", [1, 2, 3, 4, 5, 6, 7])
    emission_co2 = st.slider("Émissions CO₂ (kg eq/m²/an)", 5, 150, 30)

    type_maison = st.checkbox("Maison individuelle")

    st.subheader("📍 Localisation")
    code_postal = st.text_input("Code postal", "75015")

    submitted = st.form_submit_button("🔍 Estimer le prix")

# ======================================================
# PRÉDICTION
# ======================================================
if submitted:

    if code_postal not in prix_m2_cp:
        st.error(
            "❌ Code postal inconnu.\n\n"
            "L’estimation est limitée aux zones présentes dans le jeu de données."
        )
        st.stop()

    prix_m2_zone = prix_m2_cp[code_postal]

    bien = {
        "Surface_totale": surface,
        "Nb_pieces": nb_pieces,
        "Nb_chambres": nb_chambres,
        "Nb_salles_bain": nb_sdb,
        "Nb_wc": nb_wc,
        "Charges_annuelles": charges,
        "Cave": int(cave),
        "Parking": int(parking),
        "Classe_energie": classe_energie,
        "Conso_energie_kWh_m2_an": conso,
        "Classe_emission_CO2": classe_co2,
        "Emission_CO2_kgeq_m2_an": emission_co2,
        "Type_Maison": type_maison,
        "Prix_m2_moyenne_codepostal": prix_m2_zone
    }

    bien_df = pd.DataFrame([bien])
    bien_df[numeric_cols] = scaler.transform(bien_df[numeric_cols])

    prix_estime = rf_model.predict(bien_df)[0]

    # ======================================================
    # AFFICHAGE RÉSULTAT
    # ======================================================
    st.divider()
    st.subheader("💰 Résultat de l'estimation")

    st.metric(
        label="Prix estimé du bien",
        value=f"{prix_estime:,.0f} €"
    )

    st.caption(
        f"Prix moyen observé dans la zone : {prix_m2_zone:,.0f} €/m²"
    )

    st.info(
        f"Intervalle indicatif : "
        f"{prix_estime * 0.9:,.0f} € – {prix_estime * 1.1:,.0f} €"
    )
