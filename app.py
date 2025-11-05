import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- OPTIMIZADOR (opcional) ---
try:
    import pulp
    PULP_OK = True
except Exception:
    PULP_OK = False

st.set_page_config(page_title="Proyecto de Aula - Business Analytics (Matrículas)", layout="wide")

# =========================
# Carga de datos
# =========================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        df = pd.read_csv(path, sep=",", encoding="utf-8")
    return df

def detect_time_columns(df: pd.DataFrame):
    time_like = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            time_like.append(c)
        else:
            try:
                _ = pd.to_datetime(df[c], errors="raise")
                time_like.append(c)
            except Exception:
                pass
    seen, out = set(), []
    for c in time_like:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def simple_forecast(series: pd.Series, periods: int = 3, method: str = "naive"):
    s = series.dropna().astype(float)
    if len(s) == 0:
        return None
    if method == "naive":
        return [s.iloc[-1] for _ in range(periods)]
    elif method == "moving_avg":
        w = min(5, len(s))
        ma = s.rolling(w).mean().dropna()
        if len(ma) == 0:
            return [s.iloc[-1] for _ in range(periods)]
        return [ma.iloc[-1] for _ in range(periods)]
    elif method == "linear_trend":
        x = np.arange(len(s)); y = s.values
        if len(x) < 2:
            return [y[-1] for _ in range(periods)]
        a, b = np.polyfit(x, y, 1)
        return [a*(len(s)+i) + b for i in range(periods)]
    else:
        return None

def build_lp_best_subset(costs, impacts, budget, choose_at_most=None):
    if not PULP_OK:
        return None, None, None
    n = len(costs)
    model = pulp.LpProblem("Prescriptive_Selection", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in range(n)]
    model += pulp.lpSum([impacts[i] * x[i] for i in range(n)]), "TotalImpact"
    model += pulp.lpSum([costs[i] * x[i] for i in range(n)]) <= budget, "Budget"
    if choose_at_most is not None and choose_at_most > 0:
        model += pulp.lpSum(x) <= choose_at_most, "Cardinality"
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[model.status]
    chosen = [i for i in range(n) if x[i].value() is not None and x[i].value() > 0.5]
    objective = pulp.value(model.objective) if model.objective is not None else None
    return status, chosen, objective

# =========================
# App
# =========================
st.title("Proyecto de Aula - Análisis Prescriptivo (Admisiones y Matrículas)")
st.caption("Business Analytics • Ingeniería de Sistemas • Contexto educativo (captación, admisión, matrícula, retención)")

df = load_data("base.xlsx")
st.success("Base cargada desde base.xlsx (carpeta raíz), orientada a admisiones/matrículas.")
st.dataframe(df.head(20), use_container_width=True)

num_cols = numeric_columns(df)
time_cols = detect_time_columns(df)

with st.expander("Resumen descriptivo (automático)", expanded=False):
    if num_cols:
        st.write(df[num_cols].describe().T)
        st.bar_chart(df[num_cols].select_dtypes(include=[np.number]).mean(numeric_only=True))
    else:
        st.info("No se detectaron columnas numéricas para el resumen.")

# =========================
# Paso 1: Definición (solo desplegables)
# =========================
st.header("Paso 1: Definición y Contextualización")

empresa_opts = [
    "Universidad INCCA de Colombia",
    "Facultad de Ingeniería",
    "Facultad de Ciencias Sociales y Humanas",
    "Programa Ingeniería de Sistemas",
    "Programa Ingeniería Electrónica",
    "Otra (N/D)"
]
proceso_opts = ["Captación", "Admisión", "Matrícula", "Retención"]
problema_opts = [
    "Baja conversión de inscritos a matriculados",
    "Abandono antes de pago",
    "Tiempos de respuesta elevados",
    "Fricción en el proceso de admisión",
    "Desalineación entre mercadeo y admisiones",
    "Otro (N/D)"
]

col1, col2, col3 = st.columns(3)
with col1:
    empresa = st.selectbox("Institución/Unidad", options=empresa_opts, index=0)
with col2:
    proceso = st.selectbox("Proceso a optimizar", options=proceso_opts, index=2)
with col3:
    problema = st.selectbox("Problema u oportunidad", options=problema_opts, index=0)

# =========================
# Base Predictiva (solo desplegables)
# =========================
st.markdown("---")
st.subheader("Base Predictiva (rápida)")

kpi = st.selectbox("Selecciona el KPI principal (columna numérica)", options=num_cols if num_cols else [])
forecast_h = st.selectbox("Horizonte de proyección (períodos)", options=list(range(1, 13)), index=2)  # 3 por defecto
method = st.selectbox("Método de proyección", options=["naive", "moving_avg", "linear_trend"], index=2)

kpi_series = df[kpi] if kpi else None
baseline_value = float(kpi_series.dropna().iloc[-1]) if kpi_series is not None and len(kpi_series.dropna()) > 0 else None

if kpi:
    st.write(f"Valor actual (último observado) de **{kpi}**: {baseline_value if baseline_value is not None else 'N/D'}")
    forecast_vals = simple_forecast(kpi_series, periods=forecast_h, method=method)
    if forecast_vals is not None:
        st.write("Proyección si no se hace nada (baseline):")
        proj_df = pd.DataFrame({
            "Periodo": np.arange(1, forecast_h + 1),
            f"Proyección {kpi}": forecast_vals
        })
        st.dataframe(proj_df, use_container_width=True)
        if time_cols:
            try:
                tcol = time_cols[0]
                t = pd.to_datetime(df[tcol], errors="coerce")
                y = kpi_series.astype(float)
                hist = pd.DataFrame({"t": t, "y": y}).dropna()
                hist["tipo"] = "Histórico"
                fut_idx = pd.date_range(hist["t"].max(), periods=forecast_h + 1, freq="MS")[1:]
                fut = pd.DataFrame({"t": fut_idx, "y": forecast_vals})
                fut["tipo"] = "Pronóstico"
                allp = pd.concat([hist, fut], ignore_index=True)
                fig = px.line(allp, x="t", y="y", color="tipo", markers=True, title=f"Histórico y Proyección de {kpi}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=kpi_series.values, mode="lines+markers", name="Histórico"))
                fig.add_trace(go.Scatter(y=forecast_vals, mode="lines+markers", name="Proyección",
                                         x=np.arange(len(kpi_series), len(kpi_series) + len(forecast_vals))))
                fig.update_layout(title=f"Histórico y Proyección de {kpi}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=kpi_series.values, mode="lines+markers", name="Histórico"))
            fig.add_trace(go.Scatter(y=forecast_vals, mode="lines+markers", name="Proyección",
                                     x=np.arange(len(kpi_series), len(kpi_series) + len(forecast_vals))))
            fig.update_layout(title=f"Histórico y Proyección de {kpi}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No se pudo generar proyección con el método seleccionado.")

# =========================
# Paso 2: Alternativas (solo desplegables)
# =========================
st.header("Paso 2: Generación de Alternativas de Decisión")
st.caption("Selecciona 3 alternativas predefinidas y sus parámetros.")

alt_name_opts = [
    "A - Pauta segmentada digital",
    "B - Beca/Descuento por pronto pago",
    "C - Automatizar recordatorios multicanal",
    "D - Simplificar pasos de admisión",
    "E - Refuerzo de call center/asesoría",
    "F - Alianzas con colegios/empresas"
]
alt_desc_opts = {
    "A - Pauta segmentada digital": "Campañas con lookalike/retargeting y creatividades A/B",
    "B - Beca/Descuento por pronto pago": "Incentivo económico a inscritos antes de la fecha corte",
    "C - Automatizar recordatorios multicanal": "Secuencias email/SMS/WhatsApp con reglas y disparadores",
    "D - Simplificar pasos de admisión": "Reducción de trámites, formularios y documentos",
    "E - Refuerzo de call center/asesoría": "Aumento de asesores, horarios extendidos y callbacks",
    "F - Alianzas con colegios/empresas": "Convenios de canal con referidos y ferias"
}
impact_rel_opts = [1.0, 2.0, 5.0, 7.5, 10.0, 12.0, 15.0, 20.0]       # %
impact_abs_opts = [5, 10, 15, 20, 25, 50, 100, 200]                # delta unidades KPI
cost_opts = [0, 1_000_000, 2_500_000, 5_000_000, 8_000_000, 10_000_000, 12_000_000, 15_000_000, 20_000_000, 30_000_000]
extra_opts = ["-15% tiempo de trámite", "+5 p.p. conversión", "-10% abandono", "Mejora NPS", "Mayor alcance orgánico", "Sin cambio adicional"]

def alt_inputs(idx: int):
    st.subheader(f"Alternativa {idx+1}")
    name = st.selectbox(f"Nombre A{idx+1}", options=alt_name_opts, key=f"name_{idx}", index=idx if idx < len(alt_name_opts) else 0)
    desc = alt_desc_opts.get(name, "Acción estándar de mejora")
    st.caption(f"Acción: {desc}")
    kpi_mode = st.selectbox(f"Tipo de impacto en {kpi}", ["% relativo", "Delta absoluto"], key=f"kpi_mode_{idx}", index=0)
    if kpi_mode == "% relativo":
        impact = st.selectbox(f"Impacto esperado (%) A{idx+1}", options=impact_rel_opts, key=f"impact_{idx}")
    else:
        impact = st.selectbox(f"Impacto esperado (Δ unidades) A{idx+1}", options=impact_abs_opts, key=f"impact_{idx}")
    cost = st.selectbox(f"Costo estimado A{idx+1}", options=cost_opts, key=f"cost_{idx}", index=3)
    extra = st.selectbox(f"Métrica adicional A{idx+1}", options=extra_opts, key=f"extra_{idx}", index=0)
    return dict(name=name, desc=desc, kpi_mode=kpi_mode, impact=impact, cost=cost, extra=extra)

alts = []
for i in range(3):
    with st.container(border=True):
        alts.append(alt_inputs(i))

# =========================
# Parámetros del Modelo (solo desplegables)
# =========================
st.markdown("---")
st.subheader("Parámetros del Modelo")

budget_opts = [5_000_000, 10_000_000, 15_000_000, 20_000_000, 25_000_000, 30_000_000, 40_000_000, 50_000_000]
kpi_unit_opts = [1.0, 10.0, 100.0, 1000.0]

colb1, colb2, colb3 = st.columns(3)
with colb1:
    budget = st.selectbox("Presupuesto máximo disponible", options=budget_opts, index=5)
with colb2:
    choose_at_most = st.selectbox("Seleccionar hasta (n) alternativas", options=[1, 2, 3], index=1)
with colb3:
    kpi_unit_value = st.selectbox(f"Valor monetario por unidad de {kpi} (para beneficio neto)", options=kpi_unit_opts, index=0)

# =========================
# Simulación
# =========================
st.subheader("Simulación de resultados por alternativa")
sim_rows = []
if kpi and baseline_value is not None:
    for i, a in enumerate(alts):
        if a["kpi_mode"] == "% relativo":
            new_kpi = baseline_value * (1.0 + float(a["impact"]) / 100.0)
        else:
            new_kpi = baseline_value + float(a["impact"])
        benefit = (new_kpi - baseline_value) * float(kpi_unit_value)
        net_benefit = benefit - float(a["cost"])
        sim_rows.append({
            "Alternativa": a["name"] or f"A{i+1}",
            "Impacto_KPI_proyectado": new_kpi,
            "Beneficio_monetizado": benefit,
            "Costo": a["cost"],
            "Beneficio_neto": net_benefit,
            "Descripción": a["desc"],
            "Extras": a["extra"]
        })
    sim_df = pd.DataFrame(sim_rows)
    st.dataframe(sim_df, use_container_width=True)
    fig2 = px.bar(sim_df, x="Alternativa", y="Beneficio_neto", title="Comparación de beneficio neto por alternativa")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# Optimización
# =========================
st.header("Paso 3: Desarrollo del Modelo Prescriptivo (Optimización)")
if kpi and baseline_value is not None and len(alts) == 3:
    costs, impacts, names = [], [], []
    for i, a in enumerate(alts):
        names.append(a["name"] or f"A{i+1}")
        if a["kpi_mode"] == "% relativo":
            new_kpi = baseline_value * (1.0 + float(a["impact"]) / 100.0)
        else:
            new_kpi = baseline_value + float(a["impact"])
        benefit = (new_kpi - baseline_value) * float(kpi_unit_value)
        impacts.append(benefit); costs.append(float(a["cost"]))

    if st.button("Resolver modelo y elegir la mejor alternativa"):
        if PULP_OK:
            status, chosen, objective = build_lp_best_subset(costs, impacts, budget, choose_at_most=choose_at_most)
            if status is None:
                st.warning("No se pudo ejecutar PuLP. Se usará el método alternativo.")
            else:
                st.success(f"Estado del optimizador: {status}")
                if chosen:
                    chosen_names = [names[i] for i in chosen]
                    st.info(f"Selección óptima: {', '.join(chosen_names)}")
                    st.write(f"Impacto total maximizado: {objective:,.2f}")
                else:
                    st.warning("El modelo no seleccionó ninguna alternativa dadas las restricciones.")
        else:
            candidates = sorted([(i, impacts[i] - costs[i]) for i in range(3)], key=lambda t: t[1], reverse=True)
            chosen = []; remaining_budget = budget
            for i, nb in candidates:
                if len(chosen) >= choose_at_most:
                    break
                if costs[i] <= remaining_budget and nb > 0:
                    chosen.append(i); remaining_budget -= costs[i]
            if chosen:
                chosen_names = [names[i] for i in chosen]
                objective = sum(impacts[i] for i in chosen)
                st.success("Resultado por heurística (sin PuLP):")
                st.info(f"Selección: {', '.join(chosen_names)}")
                st.write(f"Impacto total aproximado: {objective:,.2f}")
            else:
                st.warning("La heurística no encontró alternativas viables con el presupuesto dado.")

# =========================
# Paso 4: Prescripción (solo desplegables)
# =========================
st.header("Paso 4: Prescripción Final y Plan de Acción")
st.caption("Borrador de informe con diagnóstico, recomendación y plan de implementación (sin campos de texto libre).")

tiempo_opts = ["2 semanas", "4 semanas", "6 semanas", "8 semanas"]
equipo_opts = [
    "Admisiones + Mercadeo",
    "Admisiones + Mercadeo + Registro",
    "Mercadeo + Call Center",
    "Equipo interdisciplinar (A+M+R+TI)"
]
tec_opts = [
    "CRM + embudos automatizados",
    "CRM + tableros KPI",
    "Secuencias Email/SMS/WA",
    "CRM + BI (Looker/Power BI)"
]

recurso_tiempo = st.selectbox("Tiempo estimado de ejecución", options=tiempo_opts, index=1)
recurso_personas = st.selectbox("Equipo y roles requeridos", options=equipo_opts, index=1)
recurso_tec = st.selectbox("Tecnología/herramientas", options=tec_opts, index=0)

prescripcion = ""
if 'sim_df' in locals() and kpi and baseline_value is not None and len(sim_df) > 0:
    best_row = sim_df.sort_values("Beneficio_neto", ascending=False).iloc[0]
    promedio_forecast = None
    try:
        promedio_forecast = float(np.mean(sim_df["Impacto_KPI_proyectado"].values))
    except Exception:
        pass
    proyectado_txt = f"{promedio_forecast:.2f}" if promedio_forecast is not None else "N/D"
    prescripcion = f"""
**Diagnóstico conciso**
- Institución/Proceso: {empresa} - {proceso}
- Problema/Oportunidad: {problema}
- KPI principal: {kpi}; valor actual: {baseline_value:.2f}
- Proyección (baseline) sin intervención (h={forecast_h}, método={method}): promedio estimado {proyectado_txt}.

**Recomendación prescriptiva**
- Alternativa recomendada: {best_row['Alternativa']}
- Acción: {best_row['Descripción']}
- Justificación (optimización): maximiza el beneficio neto estimado en {best_row['Beneficio_neto']:.2f}, frente a las demás opciones.
- Efecto proyectado en KPI: {best_row['Impacto_KPI_proyectado']:.2f}

**Recursos necesarios**
- Tiempo: {recurso_tiempo}
- Equipo: {recurso_personas}
- Tecnología: {recurso_tec}
- Presupuesto asociado: {best_row['Costo']:.2f}

**Plan de implementación (primeros 3 pasos)**
1) Alineación con stakeholders y validación de supuestos de impacto/costos.
2) Preparación operativa y despliegue controlado (piloto) de la alternativa seleccionada.
3) Monitoreo de KPI y métricas complementarias; ajustes iterativos y escalamiento.
"""
    st.markdown(prescripcion)


st.caption("Proyecto de Aula - Admisiones y Matrículas • Bussiness Analytics • Ingeniería de Sistemas • Universidad de Cundinamarca • 2025")

st.caption("Fabian Valero - Esteban Fonseca")
