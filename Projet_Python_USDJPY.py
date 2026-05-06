"""
===================================================================
USD/JPY FX Options — Dashboard Streamlit
Stratégie Short Vol · Élection Takaichi · Fév. 2026
===================================================================

Lancement : streamlit run usdjpy_vol_dashboard.py

Dépendances :
    pip install streamlit numpy scipy pandas matplotlib plotly requests

Partie API :
    - Yahoo Finance (yfinance) : spot USD/JPY en temps réel
    - FRED API (requests)      : taux Fed Funds Rate
    - BoJ                      : taux fixe (0.1%, non publié en API libre)
    - Vol GARCH                : estimée sur données historiques yfinance
===================================================================
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── tentative d'import des libs API (optionnelles) ────────────────
try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False


# ===================================================================
# CONFIG PAGE
# ===================================================================
st.set_page_config(
    page_title="USD/JPY FX Options · Short Vol",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS minimal pour un look terminal finance
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 12px; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] { color: #e6edf3; font-size: 22px; }
    .stSidebar { background-color: #161b22; }
    h1, h2, h3 { color: #e6edf3; }
    .badge-live   { background:#1D9E7520; color:#1D9E75; padding:3px 10px; border-radius:6px; font-size:12px; }
    .badge-manual { background:#EF9F2720; color:#EF9F27; padding:3px 10px; border-radius:6px; font-size:12px; }
    .badge-error  { background:#D85A3020; color:#D85A30; padding:3px 10px; border-radius:6px; font-size:12px; }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# 1. MODÈLE GARMAN-KOHLHAGEN
# ===================================================================

def N(x):
    return norm.cdf(x)

def n(x):
    return norm.pdf(x)

def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return intrinsic, 0.0, 0.0
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * np.exp(-r_f * T) * N(d1) - K * np.exp(-r_d * T) * N(d2)
    else:
        price = K * np.exp(-r_d * T) * N(-d2) - S * np.exp(-r_f * T) * N(-d1)
    return price, d1, d2


# ===================================================================
# 2. GREEKS
# ===================================================================

def compute_greeks(S, K, T, r_d, r_f, sigma, option_type='call'):
    _, d1, d2 = garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type)
    if T <= 0 or sigma <= 0:
        return {'Delta': 0.0, 'Gamma': 0.0, 'Vega': 0.0, 'Theta': 0.0}
    gamma = (np.exp(-r_f * T) * n(d1)) / (S * sigma * np.sqrt(T))
    vega  = S * np.exp(-r_f * T) * n(d1) * np.sqrt(T) * 0.01
    if option_type == 'call':
        delta = np.exp(-r_f * T) * N(d1)
        theta = (
            -(S * sigma * np.exp(-r_f * T) * n(d1)) / (2 * np.sqrt(T))
            - r_d * K * np.exp(-r_d * T) * N(d2)
            + r_f * S * np.exp(-r_f * T) * N(d1)
        ) / 365
    else:
        delta = -np.exp(-r_f * T) * N(-d1)
        theta = (
            -(S * sigma * np.exp(-r_f * T) * n(d1)) / (2 * np.sqrt(T))
            + r_d * K * np.exp(-r_d * T) * N(-d2)
            - r_f * S * np.exp(-r_f * T) * N(-d1)
        ) / 365
    return {
        'Delta': round(delta, 4),
        'Gamma': round(gamma, 6),
        'Vega':  round(vega,  4),
        'Theta': round(theta, 4),
    }


# ===================================================================
# 3. VOLATILITÉ IMPLICITE (BRENT)
# ===================================================================

def implied_volatility(market_price, S, K, T, r_d, r_f, option_type='call'):
    def objective(sigma):
        price, _, _ = garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type)
        return price - market_price
    try:
        return brentq(objective, 1e-4, 5.0, xtol=1e-6)
    except ValueError:
        return np.nan


# ===================================================================
# 4. PARTIE API — RÉCUPÉRATION DES DONNÉES DE MARCHÉ
# ===================================================================

FRED_API_KEY = "REMPLACEZ_PAR_VOTRE_CLE_FRED"
# Clé gratuite sur : https://fred.stlouisfed.org/docs/api/api_key.html

@st.cache_data(ttl=300)   # cache 5 minutes
def fetch_spot_usdjpy():
    """Récupère le spot USD/JPY via yfinance (Yahoo Finance)."""
    if not YFINANCE_OK:
        return None, "yfinance non installé (pip install yfinance)"
    try:
        ticker = yf.Ticker("USDJPY=X")
        hist   = ticker.history(period="5d", interval="1d")
        if hist.empty:
            return None, "Aucune donnée Yahoo Finance"
        spot   = float(hist['Close'].iloc[-1])
        chg    = float(hist['Close'].pct_change().iloc[-1]) * 100
        return {"spot": round(spot, 3), "chg_pct": round(chg, 3)}, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=3600)  # cache 1 heure
def fetch_fed_rate():
    """Récupère le taux Fed Funds Rate via l'API FRED."""
    if not REQUESTS_OK:
        return None, "requests non installé"
    if FRED_API_KEY == "REMPLACEZ_PAR_VOTRE_CLE_FRED":
        return None, "Clé FRED manquante (voir FRED_API_KEY dans le script)"
    try:
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=FEDFUNDS&api_key={FRED_API_KEY}"
            "&sort_order=desc&limit=1&file_type=json"
        )
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        val = float(r.json()['observations'][0]['value'])
        return {"rate": round(val / 100, 4)}, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=86400) # cache 24 heures
def fetch_historical_vol():
    """
    Calcule la volatilité historique annualisée (proxy GARCH)
    sur 60 jours de données USD/JPY via yfinance.
    Fenêtre 20 jours pour coller à un GARCH(1,1) de court terme.
    """
    if not YFINANCE_OK:
        return None, "yfinance non installé"
    try:
        ticker = yf.Ticker("USDJPY=X")
        hist   = ticker.history(period="3mo", interval="1d")
        if len(hist) < 20:
            return None, "Pas assez de données historiques"
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        vol_20d = float(log_ret.rolling(20).std().iloc[-1]) * np.sqrt(252)
        vol_60d = float(log_ret.std()) * np.sqrt(252)
        return {
            "vol_20d": round(vol_20d, 4),
            "vol_60d": round(vol_60d, 4),
        }, None
    except Exception as e:
        return None, str(e)


# ===================================================================
# 5. SIDEBAR — PARAMÈTRES & DONNÉES API
# ===================================================================

with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")

    # --- Bouton fetch API ---
    st.markdown("### 🌐 Données de marché")
    use_api = st.toggle("Utiliser les données en direct", value=True)

    api_status = {}

    if use_api:
        with st.spinner("Chargement des données..."):
            spot_data, spot_err = fetch_spot_usdjpy()
            fed_data,  fed_err  = fetch_fed_rate()
            vol_data,  vol_err  = fetch_historical_vol()

        # Spot
        if spot_data:
            default_S = spot_data['spot']
            api_status['spot'] = f"✅ {default_S} ({spot_data['chg_pct']:+.2f}%)"
        else:
            default_S = 149.50
            api_status['spot'] = f"⚠️ Fallback 149.50 — {spot_err}"

        # Fed
        if fed_data:
            default_rd = fed_data['rate']
            api_status['fed'] = f"✅ {default_rd*100:.2f}%"
        else:
            default_rd = 0.045
            api_status['fed'] = f"⚠️ Fallback 4.50% — {fed_err}"

        # Vol historique
        if vol_data:
            default_vg = vol_data['vol_20d']
            api_status['vol'] = f"✅ 20j={vol_data['vol_20d']*100:.1f}% / 60j={vol_data['vol_60d']*100:.1f}%"
        else:
            default_vg = 0.085
            api_status['vol'] = f"⚠️ Fallback 8.5% — {vol_err}"
    else:
        default_S  = 149.50
        default_rd = 0.045
        default_vg = 0.085

    # Status recap
    if use_api:
        with st.expander("📡 Statut API", expanded=False):
            for k, v in api_status.items():
                st.caption(f"**{k.upper()}** : {v}")
        if st.button("🔄 Rafraîchir les données"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### 📐 Paramètres du pricer")

    S = st.slider("Spot S (USD/JPY)", 130.0, 170.0, float(round(default_S, 1)), 0.1)
    K = st.slider("Strike K", 130.0, 170.0, float(round(default_S, 1)), 0.1)
    T_days = st.slider("Maturité (jours)", 1, 90, 7, 1)
    T = T_days / 365

    st.markdown("---")
    r_d = st.slider("Taux Fed r_d (%)", 0.5, 8.0, float(round(default_rd * 100, 2)), 0.05) / 100
    r_f = st.slider("Taux BoJ r_f (%)", 0.0, 2.0, 0.10, 0.05) / 100

    st.markdown("---")
    sigma_garch = st.slider("Vol GARCH σ_g (%)", 2.0, 25.0, float(round(default_vg * 100, 1)), 0.1) / 100
    sigma_impl  = st.slider("Vol Implicite σ_i (%)", 2.0, 30.0, 12.0, 0.1) / 100
    sigma_post  = st.slider("Vol post-event σ_p (%)", 2.0, 25.0, 8.5, 0.1) / 100

    st.markdown("---")
    st.caption("📚 Modèle : Garman-Kohlhagen (1983)")
    st.caption("📡 Data : Yahoo Finance · FRED · BoJ (fixe)")


# ===================================================================
# 6. CALCULS PRINCIPAUX
# ===================================================================

call_garch, d1_g, d2_g = garman_kohlhagen(S, K, T, r_d, r_f, sigma_garch, 'call')
put_garch,  _,    _    = garman_kohlhagen(S, K, T, r_d, r_f, sigma_garch, 'put')
call_impl,  d1_i, d2_i = garman_kohlhagen(S, K, T, r_d, r_f, sigma_impl,  'call')
put_impl,   _,    _    = garman_kohlhagen(S, K, T, r_d, r_f, sigma_impl,  'put')

straddle_garch = call_garch + put_garch
straddle_impl  = call_impl  + put_impl
event_premium  = straddle_impl - straddle_garch

greeks_call = compute_greeks(S, K, T, r_d, r_f, sigma_impl, 'call')
greeks_put  = compute_greeks(S, K, T, r_d, r_f, sigma_impl, 'put')

# Parité put-call
parity_lhs = call_garch - put_garch
parity_rhs = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)
parity_err = abs(parity_lhs - parity_rhs)


# ===================================================================
# 7. HEADER
# ===================================================================

col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("# 📊 USD/JPY FX Options — Stratégie Short Vol")
    st.caption("Modèle Garman-Kohlhagen · Greeks · Vol Implicite · Takaichi Fév. 2026")
with col_badge:
    if use_api and spot_data:
        st.markdown('<span class="badge-live">● LIVE</span>', unsafe_allow_html=True)
    elif use_api:
        st.markdown('<span class="badge-error">⚠ PARTIAL</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-manual">✎ MANUEL</span>', unsafe_allow_html=True)

st.markdown("---")


# ===================================================================
# 8. MÉTRIQUES HEADER
# ===================================================================

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("USD/JPY Spot",   f"{S:.2f}",         delta=f"{spot_data['chg_pct']:+.2f}%" if (use_api and spot_data) else None)
c2.metric("Taux Fed (r_d)", f"{r_d*100:.2f}%")
c3.metric("Taux BoJ (r_f)", f"{r_f*100:.2f}%")
c4.metric("Vol GARCH",      f"{sigma_garch*100:.2f}%")
c5.metric("Vol Impl. ATM",  f"{sigma_impl*100:.2f}%", delta=f"+{(sigma_impl-sigma_garch)*100:.1f}pp vs GARCH")
c6.metric("Event Premium",  f"{event_premium:+.3f} ¥")

st.markdown("---")


# ===================================================================
# 9. PRICING + GREEKS
# ===================================================================

col_pricing, col_greeks = st.columns(2)

with col_pricing:
    st.markdown("### 💰 Pricing pré-événement")

    df_price = pd.DataFrame({
        'Instrument'    : ['Call ATM', 'Put ATM', 'Straddle'],
        'Vol GARCH (¥)' : [f"{call_garch:.3f}", f"{put_garch:.3f}", f"{straddle_garch:.3f}"],
        'Vol Impl. (¥)' : [f"{call_impl:.3f}",  f"{put_impl:.3f}",  f"{straddle_impl:.3f}"],
        'Event Premium' : [
            f"+{call_impl-call_garch:.3f}",
            f"+{put_impl-put_garch:.3f}",
            f"+{event_premium:.3f}",
        ],
    })
    st.dataframe(df_price, hide_index=True, use_container_width=True)

    surv = (sigma_impl - sigma_garch) * 100
    st.info(f"📌 Surcoût de vol : **{surv:+.1f} pts de vol** ({sigma_garch*100:.1f}% → {sigma_impl*100:.1f}%)")

    # Parité put-call
    parity_ok = "✅" if parity_err < 1e-6 else "❌"
    st.caption(f"{parity_ok} Parité put-call : erreur = {parity_err:.2e}")

with col_greeks:
    st.markdown("### 🔢 Greeks du straddle short")

    delta_straddle = greeks_call['Delta'] + greeks_put['Delta']
    short_greeks = {
        g: -(greeks_call[g] + greeks_put[g])
        for g in ['Delta', 'Gamma', 'Vega', 'Theta']
    }
    notes = {
        'Delta' : "≈ 0 ATM → hedge minimal",
        'Gamma' : "< 0 short → risque si marché bouge",
        'Vega'  : "< 0 short → on gagne si vol ↓",
        'Theta' : "> 0 short → on gagne avec le temps",
    }
    units = {'Delta':'', 'Gamma':'', 'Vega':'¥/1%vol', 'Theta':'¥/j'}

    df_greeks = pd.DataFrame([{
        'Greek'     : g,
        'Call'      : f"{greeks_call[g]:.4f}",
        'Put'       : f"{greeks_put[g]:.4f}",
        'Straddle ⊖': f"{short_greeks[g]:.4f}",
        'Unité'     : units[g],
        'Note'      : notes[g],
    } for g in ['Delta', 'Gamma', 'Vega', 'Theta']])

    st.dataframe(df_greeks, hide_index=True, use_container_width=True)
    st.caption(f"🎯 Delta-hedge : {'acheter' if delta_straddle < 0 else 'vendre'} **{abs(delta_straddle):.4f}** USD/JPY spot")


st.markdown("---")


# ===================================================================
# 10. GRAPHIQUES PLOTLY
# ===================================================================

BLUE  = '#378ADD'
TEAL  = '#1D9E75'
CORAL = '#D85A30'
AMBER = '#EF9F27'
GRAY  = '#8b949e'
BG    = '#0d1117'
PAPER = '#161b22'

plotly_layout = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor=BG,
    font=dict(color='#e6edf3', size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor='#30363d', linecolor='#30363d'),
    yaxis=dict(gridcolor='#30363d', linecolor='#30363d'),
)

col_g1, col_g2 = st.columns(2)

# --- Graphique 1 : P&L short straddle ---
with col_g1:
    st.markdown("### 📉 P&L short straddle — vol crush post-élection")
    spots_range = np.linspace(S - 12, S + 12, 300)
    premium_recu = straddle_impl

    pnl_crush, pnl_stable = [], []
    for Sf in spots_range:
        c1_, _, _ = garman_kohlhagen(Sf, K, T * 0.3, r_d, r_f, sigma_post, 'call')
        p1_, _, _ = garman_kohlhagen(Sf, K, T * 0.3, r_d, r_f, sigma_post, 'put')
        pnl_crush.append(premium_recu - c1_ - p1_)

        c2_, _, _ = garman_kohlhagen(Sf, K, T * 0.3, r_d, r_f, sigma_impl, 'call')
        p2_, _, _ = garman_kohlhagen(Sf, K, T * 0.3, r_d, r_f, sigma_impl, 'put')
        pnl_stable.append(premium_recu - c2_ - p2_)

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=spots_range, y=pnl_crush, name=f'Vol crush → {sigma_post*100:.1f}%',
        line=dict(color=TEAL, width=2),
        fill='tozeroy', fillcolor='rgba(29,158,117,0.08)',
    ))
    fig_pnl.add_trace(go.Scatter(
        x=spots_range, y=pnl_stable, name=f'Vol stable {sigma_impl*100:.1f}%',
        line=dict(color=CORAL, width=2, dash='dot'),
    ))
    fig_pnl.add_hline(y=0, line_color=GRAY, line_width=0.8)
    fig_pnl.add_vline(x=K, line_color=GRAY, line_dash='dot', line_width=1)
    fig_pnl.update_layout(
        **plotly_layout,
        xaxis_title="USD/JPY spot final",
        yaxis_title="P&L (¥)",
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)'),
        height=320,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

# --- Graphique 2 : Smile de vol implicite ---
with col_g2:
    st.markdown("### 📈 Smile de volatilité implicite — 1-week")
    strikes = np.arange(S - 8, S + 8.5, 0.5)
    smile_vols = sigma_impl + 0.002 * ((strikes - K) / K * 100)**2
    smile_vols += np.where(strikes > K, 0.003, -0.001)

    market_prices_call = []
    for Ki, vi in zip(strikes, smile_vols):
        p, _, _ = garman_kohlhagen(S, Ki, T, r_d, r_f, vi, 'call')
        market_prices_call.append(p)

    implied_vols = []
    for Ki, pi in zip(strikes, market_prices_call):
        iv = implied_volatility(pi, S, Ki, T, r_d, r_f, 'call')
        implied_vols.append(iv * 100 if not np.isnan(iv) else np.nan)

    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(
        x=strikes, y=implied_vols, name='Vol implicite',
        line=dict(color=CORAL, width=2), mode='lines+markers',
        marker=dict(size=4),
    ))
    fig_smile.add_hline(
        y=sigma_garch * 100, line_color=TEAL, line_dash='dash', line_width=1.5,
        annotation_text=f"Vol GARCH {sigma_garch*100:.1f}%", annotation_font_color=TEAL,
    )
    fig_smile.add_vline(x=K, line_color=GRAY, line_dash='dot', line_width=1)
    fig_smile.update_layout(
        **plotly_layout,
        xaxis_title="Strike USD/JPY",
        yaxis_title="Vol implicite (%)",
        height=320,
    )
    st.plotly_chart(fig_smile, use_container_width=True)


col_g3, col_g4 = st.columns(2)

# --- Graphique 3 : Greeks vs Spot ---
with col_g3:
    st.markdown("### 🎯 Delta Call & Put — vs Spot")
    spots_g = np.linspace(S - 15, S + 15, 200)
    d_call = [compute_greeks(s, K, T, r_d, r_f, sigma_impl, 'call')['Delta'] for s in spots_g]
    d_put  = [compute_greeks(s, K, T, r_d, r_f, sigma_impl, 'put')['Delta']  for s in spots_g]

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=spots_g, y=d_call, name='Δ Call', line=dict(color=BLUE, width=2)))
    fig_delta.add_trace(go.Scatter(x=spots_g, y=d_put,  name='Δ Put',  line=dict(color=CORAL, width=2)))
    fig_delta.add_hline(y=0, line_color=GRAY, line_width=0.5)
    fig_delta.add_vline(x=S, line_color=GRAY, line_dash='dot', line_width=1)
    fig_delta.update_layout(
        **plotly_layout,
        xaxis_title="USD/JPY spot",
        yaxis_title="Delta",
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)'),
        height=280,
    )
    st.plotly_chart(fig_delta, use_container_width=True)

# --- Graphique 4 : Vega vs Maturité ---
with col_g4:
    st.markdown("### ⏱ Vega — décroissance temporelle")
    maturities = np.linspace(1, 90, 200) / 365
    vegas = [compute_greeks(S, K, Ti, r_d, r_f, sigma_impl, 'call')['Vega'] for Ti in maturities]

    fig_vega = go.Figure()
    fig_vega.add_trace(go.Scatter(
        x=np.linspace(1, 90, 200), y=vegas,
        name='Vega', line=dict(color=AMBER, width=2), fill='tozeroy',
        fillcolor='rgba(239,159,39,0.08)',
    ))
    fig_vega.add_vline(x=T_days, line_color=GRAY, line_dash='dash', line_width=1.5,
                       annotation_text=f"T={T_days}j", annotation_font_color=GRAY)
    fig_vega.update_layout(
        **plotly_layout,
        xaxis_title="Maturité (jours)",
        yaxis_title="Vega (¥ par 1% de vol)",
        height=280,
    )
    st.plotly_chart(fig_vega, use_container_width=True)


st.markdown("---")


# ===================================================================
# 11. SCÉNARIOS P&L POST-ÉVÉNEMENT
# ===================================================================

st.markdown("### 🎲 Backtest simplifié — Scénarios post-élection")

scenarios = [
    ("Vol crush seul (S stable)",  S,       sigma_post),
    ("Vol crush + USD/JPY +1 ¥",  S + 1.0, sigma_post),
    ("Vol crush + USD/JPY +2 ¥",  S + 2.0, max(sigma_post, 0.09)),
    ("Choc haussier USD +3 ¥",    S + 3.0, max(sigma_post, 0.095)),
    ("Choc baissier USD -2 ¥",    S - 2.0, max(sigma_post, 0.095)),
    ("Panique — vol reste haute",  S - 1.5, sigma_impl),
]

rows = []
for label, Sf, sv in scenarios:
    c_new, _, _ = garman_kohlhagen(Sf, K, T * 0.3, r_d, r_f, sv, 'call')
    p_new, _, _ = garman_kohlhagen(Sf, K, T * 0.3, r_d, r_f, sv, 'put')
    rachat = c_new + p_new
    pnl    = premium_recu - rachat
    rows.append({
        'Scénario'  : label,
        'S final'   : f"{Sf:.2f}",
        'σ final'   : f"{sv*100:.1f}%",
        'Premium reçu': f"{premium_recu:.3f} ¥",
        'Rachat'    : f"{rachat:.3f} ¥",
        'P&L'       : f"{pnl:+.3f} ¥",
        'Résultat'  : "✅ Gain" if pnl > 0 else "❌ Perte",
    })

df_scen = pd.DataFrame(rows)
st.dataframe(df_scen, hide_index=True, use_container_width=True)


# ===================================================================
# 12. EVENT PREMIUM PAR STRIKE (BAR CHART)
# ===================================================================

st.markdown("### 📊 Event premium par strike — surplus payé vs GARCH")

ep_strikes = np.arange(S - 7, S + 7.5, 0.5)
ep_values  = []
for Ki in ep_strikes:
    c_i, _, _ = garman_kohlhagen(S, Ki, T, r_d, r_f, sigma_impl,  'call')
    c_g, _, _ = garman_kohlhagen(S, Ki, T, r_d, r_f, sigma_garch, 'call')
    ep_values.append(c_i - c_g)

fig_ep = go.Figure(go.Bar(
    x=ep_strikes,
    y=ep_values,
    marker_color=[TEAL if v >= 0 else CORAL for v in ep_values],
    marker_line_width=0,
    opacity=0.85,
))
fig_ep.add_vline(x=K, line_color=GRAY, line_dash='dot', line_width=1)
fig_ep.update_layout(
    **plotly_layout,
    xaxis_title="Strike USD/JPY",
    yaxis_title="Event premium call (¥)",
    height=260,
    bargap=0.15,
)
st.plotly_chart(fig_ep, use_container_width=True)


st.markdown("---")
st.caption("🔧 Modèle : Garman-Kohlhagen (1983) · Data : Yahoo Finance · FRED · BoJ · Vol GARCH proxy · © Dashboard Streamlit")
