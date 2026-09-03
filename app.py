import os
import joblib
import numpy as np
import gradio as gr

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pavement_surrogate.joblib')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        'pavement_surrogate.joblib not found. Run the deployment cell in the original '
        'Colab notebook and place the generated file beside app.py.'
    )

B = joblib.load(MODEL_PATH)
FE = B['features']
DEF = {f: v[2] for f, v in B['feature_ranges'].items()}
k = B['transfer']['fatigue']
c = B['transfer']['rutting']

SD_LOG_NF = B['models']['et_BCAC']['resid_sd_log'] * abs(k['k2'])
SD_LOG_NR = B['models']['ec_SGF']['resid_sd_log'] * abs(c['c2'])

UI = [
    't_baseAC', 'tyrePressure', 't_baseAgg',
    'elasticModulus_baseAC', 'elasticModulus_subgradeFill',
    't_subgradeFill'
]
UI = [f for f in UI if f in FE]

UNIT_BY_PREFIX = {
    't_': 'mm',
    'density_': 'tonne/mm³',
    'elasticModulus_': 'MPa',
    'poissonsRatio_': '–',
    'tyrePressure': 'MPa',
}

LAYER_NAME = {
    'wearingAC': 'wearing course',
    'baseAC': 'asphalt base',
    'baseAgg': 'aggregate base',
    'subbase': 'subbase',
    'subgradeFill': 'subgrade fill',
    'naturalSubgrade': 'natural subgrade',
}

QUANTITY_NAME = {
    't': 'Thickness',
    'density': 'Density',
    'elasticModulus': 'Elastic modulus',
    'poissonsRatio': "Poisson's ratio",
}


def unit_of(feature):
    for prefix, unit in UNIT_BY_PREFIX.items():
        if feature.startswith(prefix) or feature == prefix:
            return unit
    return '–'


def label_of(feature):
    unit = unit_of(feature)
    if feature == 'tyrePressure':
        return f'Tyre pressure  [{unit}]'
    if '_' in feature:
        qty, layer = feature.split('_', 1)
        qty = QUANTITY_NAME.get(qty, qty)
        layer = LAYER_NAME.get(layer, layer)
        return f'{qty}, {layer}  [{unit}]'
    return f'{feature}  [{unit}]'


def predict_life(*vals):
    *slider_vals, esal = vals
    row = dict(DEF)
    row.update(dict(zip(UI, slider_vals)))
    x = np.array([[row[f] for f in FE]], dtype=float)

    log_strain = {}
    for src in ('et_BCAC', 'ec_SGF'):
        m = B['models'][src]
        xu = B['scaler'].transform(x) if m['name'] in B['scaled_models'] else x
        log_strain[src] = float(
            m['model'].predict(xu)[0] * m['y_sd'] + m['y_mean']
        )

    et = 10 ** log_strain['et_BCAC'] * B['et_scale']
    ev = 10 ** log_strain['ec_SGF']
    E_psi = row['elasticModulus_baseAC'] * B['transfer']['mpa_to_psi']

    Nf = k['k1'] * et ** k['k2'] * E_psi ** k['k3']
    Nr = c['c1'] * ev ** c['c2']
    Nd = min(Nf, Nr)

    Df, Dr = esal / Nf, esal / Nr
    D = max(Df, Dr)
    mode = 'Rutting' if Dr > Df else 'Fatigue'
    verdict = 'FAIL' if D >= 1 else ('MARGINAL' if D >= 0.7 else 'PASS')

    f_fac = 10 ** (1.96 * SD_LOG_NF)
    r_fac = 10 ** (1.96 * SD_LOG_NR)

    return (
        f'{et*1e6:.1f} µε        ({et:.3e} mm/mm)',
        f'{ev*1e6:.1f} µε        ({ev:.3e} mm/mm)',
        f'{Nf:.3e} ESALs     95% CI  {Nf/f_fac:.2e} to {Nf*f_fac:.2e} ESALs   (factor {f_fac:.2f})',
        f'{Nr:.3e} ESALs     95% CI  {Nr/r_fac:.2e} to {Nr*r_fac:.2e} ESALs   (factor {r_fac:.2f})',
        f'{Nd:.3e} ESALs        ({np.log10(Nd):.2f} log₁₀ ESALs)',
        f'{D:.4f}  [–]        failure at D = 1.0',
        f'{mode}        (Df = {Df:.4f},  Dr = {Dr:.4f})',
        f'{verdict}        remaining life {max(0.0, 1-D)*100:.0f} %',
    )


with gr.Blocks(title='Pavement Life Predictor') as app:
    gr.Markdown('# Pavement Fatigue and Rutting Life Predictor')
    gr.Markdown(
        'FEM surrogate model. Strain models are fitted in log₁₀ space; '
        'pavement life is calculated using the Asphalt Institute transfer functions. '
        f'The {len(FE)-len(UI)} model inputs not shown are held at their dataset medians. '
        'Units are shown in square brackets.'
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### Design Inputs')
            sliders = []
            for f in UI:
                lo, hi, mid = B['feature_ranges'][f]
                step = (hi - lo) / 100 if hi > lo else 0.001
                sliders.append(
                    gr.Slider(
                        minimum=lo,
                        maximum=hi,
                        value=mid,
                        step=step,
                        label=B['labels'].get(f, label_of(f)),
                        info=f'sampled range {lo:.4g} to {hi:.4g} {B["units"].get(f, unit_of(f))}',
                    )
                )

            esal_in = gr.Slider(
                minimum=1e5,
                maximum=1e8,
                value=1e6,
                step=1e5,
                label='Design traffic, n  [ESALs]',
                info='Equivalent single axle loads over the design period',
            )
            go = gr.Button('Predict', variant='primary')

        with gr.Column(scale=1):
            gr.Markdown('### Predicted Response')
            o_et = gr.Textbox(label='Tensile strain, base of asphalt  [µε]')
            o_ev = gr.Textbox(label='Vertical compressive strain, subgrade  [µε]')
            o_nf = gr.Textbox(label='Fatigue life, Nf  [ESALs]')
            o_nr = gr.Textbox(label='Rutting life, Nr  [ESALs]')
            o_nd = gr.Textbox(label='Governing design life, N_design  [ESALs]')
            o_d = gr.Textbox(label='Damage ratio, D  [dimensionless]')
            o_m = gr.Textbox(label='Critical distress mode')
            o_v = gr.Textbox(label='Verdict')

    go.click(
        predict_life,
        inputs=sliders + [esal_in],
        outputs=[o_et, o_ev, o_nf, o_nr, o_nd, o_d, o_m, o_v],
    )

if __name__ == '__main__':
    app.launch()
