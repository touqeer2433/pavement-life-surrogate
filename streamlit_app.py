import os
import joblib
import numpy as np
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pavement_surrogate.joblib")

st.set_page_config(
    page_title="Pavement Life Predictor",
    page_icon="🛣️",
    layout="wide",
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "pavement_surrogate.joblib was not found. "
        "Upload it to the same GitHub repository as this app."
    )
    st.stop()

B = joblib.load(MODEL_PATH)
FE = B["features"]
DEF = {f: v[2] for f, v in B["feature_ranges"].items()}
k = B["transfer"]["fatigue"]
c = B["transfer"]["rutting"]

SD_LOG_NF = B["models"]["et_BCAC"]["resid_sd_log"] * abs(k["k2"])
SD_LOG_NR = B["models"]["ec_SGF"]["resid_sd_log"] * abs(c["c2"])

UI = [
    "t_baseAC",
    "tyrePressure",
    "t_baseAgg",
    "elasticModulus_baseAC",
    "elasticModulus_subgradeFill",
    "t_subgradeFill",
]
UI = [f for f in UI if f in FE]

UNIT_BY_PREFIX = {
    "t_": "mm",
    "density_": "tonne/mm³",
    "elasticModulus_": "MPa",
    "poissonsRatio_": "–",
    "tyrePressure": "MPa",
}

LAYER_NAME = {
    "wearingAC": "wearing course",
    "baseAC": "asphalt base",
    "baseAgg": "aggregate base",
    "subbase": "subbase",
    "subgradeFill": "subgrade fill",
    "naturalSubgrade": "natural subgrade",
}

QUANTITY_NAME = {
    "t": "Thickness",
    "density": "Density",
    "elasticModulus": "Elastic modulus",
    "poissonsRatio": "Poisson's ratio",
}


def unit_of(feature):
    for prefix, unit in UNIT_BY_PREFIX.items():
        if feature.startswith(prefix) or feature == prefix:
            return unit
    return "–"


def label_of(feature):
    unit = unit_of(feature)
    if feature == "tyrePressure":
        return f"Tyre pressure [{unit}]"
    if "_" in feature:
        qty, layer = feature.split("_", 1)
        qty = QUANTITY_NAME.get(qty, qty)
        layer = LAYER_NAME.get(layer, layer)
        return f"{qty}, {layer} [{unit}]"
    return f"{feature} [{unit}]"


def predict_life(slider_vals, esal):
    row = dict(DEF)
    row.update(dict(zip(UI, slider_vals)))
    x = np.array([[row[f] for f in FE]], dtype=float)

    log_strain = {}
    for src in ("et_BCAC", "ec_SGF"):
        m = B["models"][src]
        if m["name"] in B["scaled_models"]:
            xu = B["scaler"].transform(x)
        else:
            xu = x
        log_strain[src] = float(
            m["model"].predict(xu)[0] * m["y_sd"] + m["y_mean"]
        )

    et = 10 ** log_strain["et_BCAC"] * B["et_scale"]
    ev = 10 ** log_strain["ec_SGF"]

    E_MPa = row["elasticModulus_baseAC"]
    E_psi = E_MPa * B["transfer"]["mpa_to_psi"]

    Nf = k["k1"] * et ** k["k2"] * E_psi ** k["k3"]
    Nr = c["c1"] * ev ** c["c2"]
    Nd = min(Nf, Nr)

    Df = esal / Nf
    Dr = esal / Nr
    D = max(Df, Dr)
    mode = "Rutting" if Dr > Df else "Fatigue"
    verdict = "FAIL" if D >= 1 else ("MARGINAL" if D >= 0.7 else "PASS")

    f_fac = 10 ** (1.96 * SD_LOG_NF)
    r_fac = 10 ** (1.96 * SD_LOG_NR)

    return {
        "et": et,
        "ev": ev,
        "Nf": Nf,
        "Nr": Nr,
        "Nd": Nd,
        "D": D,
        "Df": Df,
        "Dr": Dr,
        "mode": mode,
        "verdict": verdict,
        "f_low": Nf / f_fac,
        "f_high": Nf * f_fac,
        "r_low": Nr / r_fac,
        "r_high": Nr * r_fac,
        "f_fac": f_fac,
        "r_fac": r_fac,
    }


st.title("🛣️ Pavement Fatigue and Rutting Life Predictor")
st.caption(
    "FEM surrogate model. Strain models are fitted in log₁₀ space. "
    "Life is calculated using the Asphalt Institute transfer functions. "
    f"{len(FE) - len(UI)} model inputs not shown are held at their dataset medians."
)

with st.sidebar:
    st.header("Design Inputs")
    values = []

    for f in UI:
        lo, hi, mid = B["feature_ranges"][f]
        values.append(
            st.slider(
                label_of(f),
                min_value=float(lo),
                max_value=float(hi),
                value=float(mid),
                step=float((hi - lo) / 100) if hi > lo else 0.01,
                help=f"Sampled range: {lo:.6g} to {hi:.6g} {B['units'][f]}",
            )
        )

    esal = st.slider(
        "Design traffic, n [ESALs]",
        min_value=100000.0,
        max_value=100000000.0,
        value=1000000.0,
        step=100000.0,
        help="Equivalent single axle loads over the design period.",
    )

    predict = st.button("Predict", type="primary", use_container_width=True)

if predict:
    result = predict_life(values, esal)
else:
    result = predict_life(
        [DEF[f] for f in UI],
        1_000_000.0,
    )

st.subheader("Predicted Response")

c1, c2, c3 = st.columns(3)
c1.metric("Tensile strain", f"{result['et'] * 1e6:.1f} µε")
c2.metric("Vertical compressive strain", f"{result['ev'] * 1e6:.1f} µε")
c3.metric("Governing design life", f"{result['Nd']:.3e} ESALs")

c4, c5, c6 = st.columns(3)
c4.metric("Fatigue life, Nf", f"{result['Nf']:.3e} ESALs")
c5.metric("Rutting life, Nr", f"{result['Nr']:.3e} ESALs")
c6.metric("Damage ratio, D", f"{result['D']:.4f}")

st.divider()

left, right = st.columns(2)

with left:
    st.write(f"**Critical distress mode:** {result['mode']}")
    st.write(f"**Fatigue damage, Df:** {result['Df']:.4f}")
    st.write(f"**Rutting damage, Dr:** {result['Dr']:.4f}")
    st.write(
        f"**Fatigue 95% CI:** {result['f_low']:.2e} to "
        f"{result['f_high']:.2e} ESALs (factor {result['f_fac']:.2f})"
    )
    st.write(
        f"**Rutting 95% CI:** {result['r_low']:.2e} to "
        f"{result['r_high']:.2e} ESALs (factor {result['r_fac']:.2f})"
    )

with right:
    verdict = result["verdict"]
    if verdict == "PASS":
        st.success(f"### {verdict}")
    elif verdict == "MARGINAL":
        st.warning(f"### {verdict}")
    else:
        st.error(f"### {verdict}")

    remaining = max(0.0, 1 - result["D"]) * 100
    st.write(f"Remaining life: **{remaining:.0f}%**")
    st.caption(
        "A damage ratio D ≥ 1.0 is treated as failure; "
        "0.7 ≤ D < 1.0 is treated as marginal."
    )

with st.expander("Model information"):
    st.write(f"Features in surrogate model: {len(FE)}")
    st.write(f"Fatigue strain model: {B['models']['et_BCAC']['name']}")
    st.write(f"Rutting strain model: {B['models']['ec_SGF']['name']}")
    st.write(f"Fatigue strain test R²: {B['models']['et_BCAC']['test_r2']:.4f}")
    st.write(f"Rutting strain test R²: {B['models']['ec_SGF']['test_r2']:.4f}")
    st.write(
        "The remaining model inputs are fixed at their dataset medians, "
        "as in the original deployment design."
    )
