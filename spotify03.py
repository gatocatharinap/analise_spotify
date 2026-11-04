import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np

# Bibliotecas de Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Spotify Streaming Dashboard",
    page_icon="🎵",
    layout="wide"
)

# TÍTULO
st.title("🎵 Spotify Streaming Performance Dashboard")
st.write(
    "Dashboard para análise do desempenho de artistas "
    "e seus streams no Spotify."
)

# CARREGANDO OS DADOS
df = pd.read_csv(
    "Spotify Streaming Performance Dataset(in).csv",
    sep=";"
)

# Remove espaços extras dos nomes das colunas
df.columns = df.columns.str.strip()

# Remove coluna vazia, caso exista
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Converte o ano de estreia para número
df["Debut Year"] = pd.to_numeric(
    df["Debut Year"],
    errors="coerce"
)

# Converte as colunas numéricas
colunas_numericas = [
    "Total Streams (in millions)",
    "Lead Streams (in millions)",
    "Feature Streams (in millions)",
    "Solo Streams (in millions)",
    "Collaborative Streams (in millions)"
]

for coluna in colunas_numericas:
    df[coluna] = pd.to_numeric(
        df[coluna],
        errors="coerce"
    )

# Colunas derivadas (as colunas de % do CSV vêm com separadores de milhar
# errados, por isso são recalculadas a partir dos valores absolutos)
_total = df["Total Streams (in millions)"]
df["% Colaborativo"] = (df["Collaborative Streams (in millions)"] / _total).clip(0, 1)
df["% Feature"] = (df["Feature Streams (in millions)"] / _total).clip(0, 1)
df["Streams Percentil"] = _total.rank(pct=True)

# SIDEBAR - FILTROS
st.sidebar.header("🔎 Filtros")

generos = sorted(df["Primary Genre"].dropna().unique())
generos_selecionados = st.sidebar.multiselect("🎵 Gênero musical", generos, default=generos)

paises = sorted(df["Country of Origin"].dropna().unique())
paises_selecionados = st.sidebar.multiselect("🌎 País de origem", paises, default=paises)

idiomas = sorted(df["Primary Language"].dropna().unique())
idiomas_selecionados = st.sidebar.multiselect("🗣️ Idioma", idiomas, default=idiomas)

sexos = sorted(df["Sex"].dropna().unique())
sexos_selecionados = st.sidebar.multiselect("👤 Sexo", sexos, default=sexos)

tipos = sorted(df["Artist Type"].dropna().unique())
tipos_selecionados = st.sidebar.multiselect("🎤 Tipo de artista", tipos, default=tipos)

# APLICANDO OS FILTROS
df_filtrado = df[
    (df["Primary Genre"].isin(generos_selecionados)) &
    (df["Country of Origin"].isin(paises_selecionados)) &
    (df["Primary Language"].isin(idiomas_selecionados)) &
    (df["Sex"].isin(sexos_selecionados)) &
    (df["Artist Type"].isin(tipos_selecionados))
]

if df_filtrado.empty:
    st.warning("Nenhum artista encontrado com os filtros selecionados.")
    st.stop()

# INDICADORES
total_streams = df_filtrado["Total Streams (in millions)"].sum()
quantidade_artistas = len(df_filtrado)
media_streams = df_filtrado["Total Streams (in millions)"].mean()
quantidade_paises = df_filtrado["Country of Origin"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎧 Total de Streams", f"{total_streams:,.1f} milhões")
col2.metric("🎤 Artistas", quantidade_artistas)
col3.metric("📈 Média de Streams", f"{media_streams:,.1f} milhões")
col4.metric("🌎 Países", quantidade_paises)

# TOP 10 ARTISTAS
st.subheader("🏆 Top 10 artistas por número de streams")
top_artistas = (
    df_filtrado
    .sort_values("Total Streams (in millions)", ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top_artistas["Artist Name"], top_artistas["Total Streams (in millions)"])
ax.set_xlabel("Streams (milhões)")
ax.set_ylabel("Artista")
ax.set_title("Top 10 artistas por streams")
ax.invert_yaxis()
st.pyplot(fig)

# STREAMS POR GÊNERO
st.subheader("🎵 Streams por gênero musical")
streams_genero = (
    df_filtrado
    .groupby("Primary Genre")["Total Streams (in millions)"]
    .sum()
    .sort_values(ascending=False)
)
st.bar_chart(streams_genero)

# STREAMS POR PAÍS
st.subheader("🌎 Streams por país")
streams_pais = (
    df_filtrado
    .groupby("Country of Origin")["Total Streams (in millions)"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)
st.bar_chart(streams_pais)

# ANO DE ESTREIA X STREAMS
st.subheader("📅 Ano de estreia x Streams")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df_filtrado["Debut Year"], df_filtrado["Total Streams (in millions)"])
ax.set_xlabel("Ano de estreia")
ax.set_ylabel("Streams (milhões)")
ax.set_title("Relação entre ano de estreia e streams")
st.pyplot(fig)

# =========================================================
# SVM: SEXO vs. STREAMS vs. GÊNERO MUSICAL
# =========================================================
st.subheader("🤖 SVM: Classificação de Sexo por Gênero Musical e Streams")

df_svm_sex = df_filtrado[
    ["Artist Name", "Primary Genre", "Total Streams (in millions)", "Sex"]
].dropna()

if len(df_svm_sex) > 5 and df_svm_sex["Sex"].nunique() > 1:
    genre_categories = df_svm_sex["Primary Genre"].astype("category")
    df_svm_sex["Genre_Code"] = genre_categories.cat.codes

    X = df_svm_sex[["Genre_Code", "Total Streams (in millions)"]].values
    sex_categories = df_svm_sex["Sex"].astype("category")
    y = sex_categories.cat.codes.values

    model = SVC(kernel="linear", C=1.0)
    model.fit(X, y)

    fig_svm_sex = go.Figure()
    generos = df_svm_sex["Primary Genre"].dropna().unique()

    for genero in generos:
        df_sub = df_svm_sex[df_svm_sex["Primary Genre"] == genero]
        fig_svm_sex.add_trace(
            go.Scatter(
                x=df_sub["Sex"],
                y=df_sub["Total Streams (in millions)"],
                mode="markers",
                name=genero,
                text=df_sub["Artist Name"],
                customdata=np.column_stack((df_sub["Primary Genre"], df_sub["Sex"])),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Sexo: %{customdata[1]}<br>"
                    "Gênero: %{customdata[0]}<br>"
                    "Streams: %{y:,.1f}M"
                    "<extra></extra>"
                ),
                marker=dict(size=9, opacity=0.8)
            )
        )

    fig_svm_sex.update_layout(
        title="Classificação por Sexo com base no Gênero Musical e Total de Streams",
        xaxis=dict(
            title="Sexo do Artista",
            categoryorder="array",
            categoryarray=["Male", "Female", "Mixed"]
        ),
        yaxis=dict(title="Total Streams (milhões)"),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title="Gêneros Musicais"
        ),
        margin=dict(l=60, r=30, t=80, b=150)
    )

    st.plotly_chart(fig_svm_sex, use_container_width=True)
else:
    st.info("Selecione mais opções no filtro de Sexo/Gênero para gerar o gráfico SVM.")

# STREAMS SOLO X COLABORATIVOS
st.subheader("🎤 Streams solo x colaborativos")
total_solo = df_filtrado["Solo Streams (in millions)"].sum()
total_colaborativo = df_filtrado["Collaborative Streams (in millions)"].sum()

dados_streams = pd.Series({"Solo": total_solo, "Colaborativo": total_colaborativo})
st.bar_chart(dados_streams)

# =========================================================
# DBSCAN: AGRUPAMENTO DE ARTISTAS (VERSÃO MELHORADA)
# =========================================================
st.subheader("🔎 DBSCAN: Agrupamento de artistas por perfil de streams")
st.write(
    "O DBSCAN agrupa artistas que estão **densamente próximos** no gráfico e marca como "
    "**ruído** (❌) os que não pertencem a nenhum grupo, isto é, artistas com perfil fora do comum. "
    "Os dados passam por log + padronização, porque os streams são muito assimétricos "
    "(poucos artistas concentram bilhões de streams)."
)

OPCOES_STREAMS = {
    "Streams solo": "Solo Streams (in millions)",
    "Streams colaborativos": "Collaborative Streams (in millions)",
    "Streams como líder": "Lead Streams (in millions)",
    "Streams como feature": "Feature Streams (in millions)",
    "Streams totais": "Total Streams (in millions)",
}

col_fx, col_fy = st.columns(2)
nome_x = col_fx.selectbox("Variável do eixo X", list(OPCOES_STREAMS), index=0, key="db_x")
nome_y = col_fy.selectbox("Variável do eixo Y", list(OPCOES_STREAMS), index=1, key="db_y")

if nome_x == nome_y:
    st.warning("Escolha duas variáveis diferentes para executar o DBSCAN.")
else:
    cx, cy = OPCOES_STREAMS[nome_x], OPCOES_STREAMS[nome_y]
    colunas_db = list(dict.fromkeys([
        "Artist Name", "Primary Genre", "Country of Origin",
        "Total Streams (in millions)", cx, cy
    ]))
    df_dbscan = df_filtrado[colunas_db].dropna(subset=[cx, cy]).copy()
    df_dbscan = df_dbscan[(df_dbscan[[cx, cy]] >= 0).all(axis=1)]

    if len(df_dbscan) < 3:
        st.warning("Não há artistas suficientes com dados válidos para executar o DBSCAN.")
    else:
        col_eps, col_ms = st.columns(2)
        eps = col_eps.slider(
            "Distância eps", 0.05, 1.5, 0.30, 0.05, key="db_eps",
            help="Raio de vizinhança no espaço padronizado. Menor = grupos mais "
                 "estreitos e mais ruído; maior = grupos tendem a se fundir."
        )
        min_samples = col_ms.slider(
            "Mínimo de pontos por vizinhança", 2, 20, 5, 1, key="db_ms",
            help="Quantos artistas (incluindo ele mesmo) precisam estar dentro do raio eps "
                 "para formar uma região densa."
        )

        X_scaled = StandardScaler().fit_transform(np.log1p(df_dbscan[[cx, cy]].to_numpy()))
        df_dbscan["Cluster"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
        df_dbscan["Grupo"] = df_dbscan["Cluster"].map(
            lambda c: "Ruído (-1)" if c == -1 else f"Cluster {c}"
        )

        mascara = df_dbscan["Cluster"].to_numpy() != -1
        n_clusters = len(set(df_dbscan["Cluster"]) - {-1})
        n_ruido = int((~mascara).sum())
        pct_ruido = 100 * n_ruido / len(df_dbscan)

        silhueta = None
        if n_clusters >= 2 and mascara.sum() > n_clusters:
            silhueta = silhouette_score(X_scaled[mascara], df_dbscan["Cluster"].to_numpy()[mascara])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Clusters encontrados", n_clusters)
        m2.metric("⚠️ Artistas como ruído", n_ruido)
        m3.metric("📊 Percentual de ruído", f"{pct_ruido:.1f}%")
        m4.metric(
            "📐 Silhouette",
            f"{silhueta:.2f}" if silhueta is not None else "—",
            help="De -1 a 1 (calculado sem o ruído). Quanto mais perto de 1, mais "
                 "coesos e separados estão os clusters. Exige ao menos 2 clusters."
        )

        with st.expander("📐 Como escolher o eps? (gráfico k-distance, o \"cotovelo\" do DBSCAN)", expanded=True):
            k = min(min_samples, len(df_dbscan))
            dist, _ = NearestNeighbors(n_neighbors=k).fit(X_scaled).kneighbors(X_scaled)
            kdist = np.sort(dist[:, -1])
            fig_k = go.Figure(go.Scatter(y=kdist, mode="lines", name="Distância ao k-ésimo vizinho"))
            fig_k.add_hline(y=eps, line_dash="dash", line_color="#E74C3C",
                            annotation_text=f"eps atual = {eps:.2f}")
            fig_k.update_layout(
                xaxis_title="Artistas ordenados pela distância",
                yaxis_title=f"Distância ao {k}º vizinho mais próximo",
                height=350, margin=dict(t=30)
            )
            st.plotly_chart(fig_k, use_container_width=True)
            st.caption(
                "Procure o 'cotovelo' da curva: antes dele estão os pontos em regiões densas; "
                "depois, os pontos isolados. O eps ideal costuma ficar perto do cotovelo."
            )

        # Coordenadas do gráfico em log10(1 + streams); o hover mostra o valor real
        df_dbscan["_x"] = np.log10(1 + df_dbscan[cx])
        df_dbscan["_y"] = np.log10(1 + df_dbscan[cy])
        total = df_dbscan["Total Streams (in millions)"]
        df_dbscan["_tam"] = np.clip(np.sqrt(total / total.median()) * 8, 6, 28)

        paleta = px.colors.qualitative.Safe
        fig_dbscan = go.Figure()
        grupos = sorted(
            df_dbscan["Grupo"].unique(),
            key=lambda g: (g == "Ruído (-1)", int(g.split()[-1].strip("()")) if g != "Ruído (-1)" else 0)
        )

        for grupo in grupos:
            d = df_dbscan[df_dbscan["Grupo"] == grupo]
            eh_ruido = grupo == "Ruído (-1)"
            cor = "#E74C3C" if eh_ruido else paleta[int(d["Cluster"].iloc[0]) % len(paleta)]
            fig_dbscan.add_trace(go.Scatter(
                x=d["_x"], y=d["_y"], mode="markers",
                name=f"{grupo} ({len(d)})",
                text=d["Artist Name"],
                customdata=np.column_stack([
                    d["Primary Genre"], d["Country of Origin"],
                    d["Total Streams (in millions)"], d[cx], d[cy]
                ]),
                marker=dict(
                    size=d["_tam"], color=cor,
                    symbol="x" if eh_ruido else "circle",
                    opacity=0.85,
                    line=dict(width=0 if eh_ruido else 0.6, color="white")
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Gênero: %{customdata[0]} | País: %{customdata[1]}<br>"
                    f"{nome_x}: " "%{customdata[3]:,.1f} M<br>"
                    f"{nome_y}: " "%{customdata[4]:,.1f} M<br>"
                    "Total: %{customdata[2]:,.1f} M"
                    f"<br>{grupo}<extra></extra>"
                )
            ))

        centros = df_dbscan[mascara].groupby("Cluster")[["_x", "_y"]].mean()
        if not centros.empty:
            fig_dbscan.add_trace(go.Scatter(
                x=centros["_x"], y=centros["_y"], mode="markers+text",
                name="Centro do cluster",
                text=[str(c) for c in centros.index], textposition="top center",
                marker=dict(size=14, symbol="diamond", color="black",
                            line=dict(width=1.5, color="white")),
                hoverinfo="skip"
            ))

        fig_dbscan.update_layout(
            title="Clusters de artistas identificados pelo DBSCAN "
                  "(tamanho do ponto = streams totais)",
            xaxis_title=f"log10(1 + {nome_x.lower()} em milhões)",
            yaxis_title=f"log10(1 + {nome_y.lower()} em milhões)",
            legend_title="Grupo identificado",
            hovermode="closest", height=620
        )
        st.plotly_chart(fig_dbscan, use_container_width=True)

        st.subheader("📋 Resumo dos grupos")
        resumo_dbscan = (
            df_dbscan.groupby("Grupo")
            .agg(**{
                "Artistas": ("Artist Name", "size"),
                f"Média {nome_x.lower()} (M)": (cx, "mean"),
                f"Média {nome_y.lower()} (M)": (cy, "mean"),
                "Média streams totais (M)": ("Total Streams (in millions)", "mean"),
                "Gênero mais comum": ("Primary Genre", lambda s: s.mode().iat[0]),
                "País mais comum": ("Country of Origin", lambda s: s.mode().iat[0]),
            })
            .round(1).reset_index()
        )
        st.dataframe(resumo_dbscan, use_container_width=True, hide_index=True)

        st.subheader("⚠️ Artistas fora dos clusters (ruído)")
        df_ruido = df_dbscan[~mascara].sort_values("Total Streams (in millions)", ascending=False)
        if df_ruido.empty:
            st.info("Nenhum artista foi classificado como ruído com os parâmetros atuais.")
        else:
            st.dataframe(
                df_ruido[["Artist Name", "Primary Genre", "Country of Origin",
                          "Total Streams (in millions)", cx, cy]],
                use_container_width=True, hide_index=True
            )

# =========================================================
# K-MEANS: MÉTODO DO COTOVELO, SILHOUETTE E GAP STATISTIC
# =========================================================
st.subheader("📉 Escolha do número de clusters: Cotovelo, Silhouette e Gap")
st.write(
    "O DBSCAN não pede o número de clusters (ele usa eps e min_samples, vistos no gráfico "
    "k-distance acima). Já o **K-Means** exige o k, e os métodos do **Cotovelo** e do **Gap "
    "Statistic** servem para escolhê-lo. Aqui eles são aplicados aos mesmos dados, em log + padronização."
)

kc1, kc2 = st.columns(2)
km_nome_x = kc1.selectbox("Variável 1", list(OPCOES_STREAMS), index=0, key="km_x")
km_nome_y = kc2.selectbox("Variável 2", list(OPCOES_STREAMS), index=1, key="km_y")


def estatistica_gap(X, k_max=10, B=10, seed=42):
    """Gap statistic (Tibshirani et al.): compara log(Wk) com dados de referência uniformes."""
    rng = np.random.default_rng(seed)
    minimos, maximos = X.min(axis=0), X.max(axis=0)
    gaps, desvios = [], []
    for k in range(1, k_max + 1):
        log_wk = np.log(KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X).inertia_)
        log_ref = []
        for _ in range(B):
            Xr = rng.uniform(minimos, maximos, size=X.shape)
            log_ref.append(np.log(KMeans(n_clusters=k, n_init=3, random_state=seed).fit(Xr).inertia_))
        gaps.append(np.mean(log_ref) - log_wk)
        desvios.append(np.std(log_ref) * np.sqrt(1 + 1 / B))
    k_gap = next((k for k in range(1, k_max)
                  if gaps[k - 1] >= gaps[k] - desvios[k]), k_max)  # critério de Tibshirani
    return np.array(gaps), np.array(desvios), k_gap


if km_nome_x == km_nome_y:
    st.warning("Escolha duas variáveis diferentes.")
else:
    km_cols = [OPCOES_STREAMS[km_nome_x], OPCOES_STREAMS[km_nome_y]]
    df_km = df_filtrado[list(dict.fromkeys(["Artist Name"] + km_cols))].dropna(subset=km_cols)
    df_km = df_km[(df_km[km_cols] >= 0).all(axis=1)].copy()

    if len(df_km) < 12:
        st.info("Poucos artistas para calcular o método do cotovelo com os filtros atuais.")
    else:
        X_km = StandardScaler().fit_transform(np.log1p(df_km[km_cols].to_numpy()))
        k_max = min(10, len(df_km) - 1)
        ks = list(range(1, k_max + 1))

        inercias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_km).inertia_ for k in ks]
        silhuetas = [np.nan] + [
            silhouette_score(X_km, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_km))
            for k in ks[1:]
        ]
        gaps, desvios_gap, k_gap = estatistica_gap(X_km, k_max=k_max)

        # Cotovelo automático: ponto mais distante da reta entre o primeiro e o último ponto
        p = np.column_stack([ks, inercias]).astype(float)
        p = (p - p.min(axis=0)) / (p.max(axis=0) - p.min(axis=0))
        reta = p[-1] - p[0]
        dist_reta = np.abs(reta[0] * (p[:, 1] - p[0, 1]) - reta[1] * (p[:, 0] - p[0, 0])) / np.hypot(*reta)
        k_cotovelo = ks[int(np.argmax(dist_reta))]
        k_silhueta = ks[1 + int(np.nanargmax(silhuetas[1:]))]

        fig_k3 = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Cotovelo (inércia)", "Silhouette médio", "Gap Statistic")
        )
        fig_k3.add_trace(go.Scatter(x=ks, y=inercias, mode="lines+markers", name="Inércia"), row=1, col=1)
        fig_k3.add_vline(x=k_cotovelo, line_dash="dash", line_color="#E74C3C", row=1, col=1)
        fig_k3.add_trace(go.Scatter(x=ks, y=silhuetas, mode="lines+markers", name="Silhouette"), row=1, col=2)
        fig_k3.add_vline(x=k_silhueta, line_dash="dash", line_color="#E74C3C", row=1, col=2)
        fig_k3.add_trace(go.Scatter(
            x=ks, y=gaps, mode="lines+markers", name="Gap",
            error_y=dict(type="data", array=desvios_gap, visible=True)
        ), row=1, col=3)
        fig_k3.add_vline(x=k_gap, line_dash="dash", line_color="#E74C3C", row=1, col=3)
        fig_k3.update_xaxes(title_text="Número de clusters (k)", dtick=1)
        fig_k3.update_yaxes(title_text="Inércia (WCSS)", row=1, col=1)
        fig_k3.update_yaxes(title_text="Silhouette", row=1, col=2)
        fig_k3.update_yaxes(title_text="Gap(k)", row=1, col=3)
        fig_k3.update_layout(height=380, showlegend=False, margin=dict(t=50))
        st.plotly_chart(fig_k3, use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("k sugerido pelo Cotovelo", k_cotovelo)
        r2.metric("k com melhor Silhouette", k_silhueta)
        r3.metric("k sugerido pelo Gap", k_gap)
        st.caption(
            "Linha vermelha tracejada = k sugerido por cada método. Quando os métodos divergem, "
            "o cotovelo é o mais subjetivo; o Gap tem critério estatístico e o Silhouette mede a qualidade da separação."
        )

        k_escolhido = st.slider("k para o K-Means", 2, k_max, int(max(2, k_cotovelo)), key="km_k")
        df_km["Cluster"] = KMeans(n_clusters=k_escolhido, n_init=10, random_state=42).fit_predict(X_km)
        fig_km = go.Figure()
        for c in sorted(df_km["Cluster"].unique()):
            d = df_km[df_km["Cluster"] == c]
            fig_km.add_trace(go.Scatter(
                x=np.log10(1 + d[km_cols[0]]), y=np.log10(1 + d[km_cols[1]]), mode="markers",
                name=f"Cluster {c} ({len(d)})", text=d["Artist Name"],
                marker=dict(size=9, opacity=0.8, color=px.colors.qualitative.Safe[c % 11]),
                hovertemplate="<b>%{text}</b><extra>Cluster " + str(c) + "</extra>"
            ))
        fig_km.update_layout(
            title=f"K-Means com k = {k_escolhido}",
            xaxis_title=f"log10(1 + {km_nome_x.lower()} em milhões)",
            yaxis_title=f"log10(1 + {km_nome_y.lower()} em milhões)",
            height=520
        )
        st.plotly_chart(fig_km, use_container_width=True)

# =========================================================
# SCATTER: RELAÇÃO ENTRE DUAS VARIÁVEIS
# =========================================================
st.subheader("📈 Scatter: relação entre duas variáveis")
st.write(
    "Cada ponto é um artista. Escolha as variáveis dos eixos, a cor e se quer escala "
    "logarítmica. A linha tracejada é uma regressão linear e o coeficiente de correlação "
    "resume a força da relação."
)

OPCOES_SCATTER = {
    "Streams totais": "Total Streams (in millions)",
    "Streams como líder": "Lead Streams (in millions)",
    "Streams como feature": "Feature Streams (in millions)",
    "Streams solo": "Solo Streams (in millions)",
    "Streams colaborativos": "Collaborative Streams (in millions)",
    "% de streams colaborativos": "% Colaborativo",
    "% de streams como feature": "% Feature",
    "Ano de estreia": "Debut Year",
}
OPCOES_COR = {
    "Gênero musical": "Primary Genre",
    "Sexo": "Sex",
    "Tipo de artista": "Artist Type",
    "Idioma": "Primary Language",
    "País de origem": "Country of Origin",
}

s1, s2, s3 = st.columns(3)
sc_x = s1.selectbox("Eixo X", list(OPCOES_SCATTER), index=1, key="sc_x")
sc_y = s2.selectbox("Eixo Y", list(OPCOES_SCATTER), index=2, key="sc_y")
sc_cor = s3.selectbox("Cor dos pontos", list(OPCOES_COR), index=0, key="sc_cor")
o1, o2, o3 = st.columns(3)
log_x = o1.checkbox("Escala log no eixo X", value=True, key="sc_logx")
log_y = o2.checkbox("Escala log no eixo Y", value=True, key="sc_logy")
mostrar_reta = o3.checkbox("Mostrar linha de tendência", value=True, key="sc_reta")

if sc_x == sc_y:
    st.warning("Escolha variáveis diferentes para os eixos X e Y.")
else:
    vx, vy, vcor = OPCOES_SCATTER[sc_x], OPCOES_SCATTER[sc_y], OPCOES_COR[sc_cor]
    TOTAL = "Total Streams (in millions)"
    colunas_sc = list(dict.fromkeys(["Artist Name", vx, vy, vcor, TOTAL]))
    df_sc = df_filtrado[colunas_sc].dropna()
    if log_x:
        df_sc = df_sc[df_sc[vx] > 0]
    if log_y:
        df_sc = df_sc[df_sc[vy] > 0]

    if len(df_sc) < 3:
        st.info("Poucos artistas para montar o scatter com os filtros atuais.")
    else:
        fig_sc = px.scatter(
            df_sc, x=vx, y=vy, color=vcor, size=TOTAL, size_max=28,
            hover_name="Artist Name", opacity=0.75, log_x=log_x, log_y=log_y,
            labels={vx: sc_x, vy: sc_y, vcor: sc_cor, TOTAL: "Streams totais (M)"}
        )

        # Correlações calculadas no mesmo espaço em que o gráfico é lido (log ou linear)
        gx = np.log10(df_sc[vx]) if log_x else df_sc[vx]
        gy = np.log10(df_sc[vy]) if log_y else df_sc[vy]
        pearson = float(np.corrcoef(gx, gy)[0, 1])
        spearman = float(gx.corr(gy, method="spearman"))

        if mostrar_reta and gx.nunique() > 1:
            a, b = np.polyfit(gx, gy, 1)
            xs = np.linspace(gx.min(), gx.max(), 100)
            ys = a * xs + b
            fig_sc.add_trace(go.Scatter(
                x=10 ** xs if log_x else xs,
                y=10 ** ys if log_y else ys,
                mode="lines", name="Tendência linear",
                line=dict(color="black", dash="dash", width=2)
            ))

        fig_sc.update_layout(
            title=f"{sc_y} × {sc_x} (tamanho do ponto = streams totais)",
            height=620, legend_title=sc_cor
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        def forca(r):
            r = abs(r)
            return ("muito fraca" if r < 0.2 else "fraca" if r < 0.4 else
                    "moderada" if r < 0.6 else "forte" if r < 0.8 else "muito forte")

        k1, k2, k3 = st.columns(3)
        k1.metric("Correlação de Pearson (r)", f"{pearson:.2f}",
                  help="Mede a relação LINEAR entre as variáveis (-1 a 1).")
        k2.metric("Correlação de Spearman (ρ)", f"{spearman:.2f}",
                  help="Mede a relação de ORDEM (monotônica). Menos sensível a outliers.")
        k3.metric("Artistas no gráfico", len(df_sc))
        st.caption(
            f"Leitura: relação {'positiva' if pearson >= 0 else 'negativa'} {forca(pearson)} "
            f"entre '{sc_x}' e '{sc_y}'. Correlação não implica causalidade."
        )

# =========================================================
# LÓGICA FUZZY: ÍNDICE DE DEPENDÊNCIA COLABORATIVA
# =========================================================
st.subheader("🌫️ Lógica Fuzzy: índice de dependência de colaborações")
st.write(
    "Em vez de rotular o artista de forma rígida (ex.: 'colaborativo' se > 50%), a lógica fuzzy "
    "permite **graus de pertinência**: um artista pode ser, ao mesmo tempo, 'média' e 'alta' "
    "em colaboração. Um sistema de regras (estilo Mamdani) combina três entradas e gera "
    "um índice de **0 a 100**: quanto maior, mais a carreira do artista depende de colaborações."
)


def trapmf(x, a, b, c, d):
    """Função de pertinência trapezoidal (a == b ou c == d formam 'ombros')."""
    x = np.asarray(x, dtype=float)
    subida = np.ones_like(x) if b == a else (x - a) / (b - a)
    descida = np.ones_like(x) if d == c else (d - x) / (d - c)
    return np.clip(np.minimum(subida, descida), 0, 1)


MF_COLAB = {"Baixa": (0, 0, 0.15, 0.40), "Média": (0.20, 0.50, 0.50, 0.80), "Alta": (0.60, 0.85, 1, 1)}
MF_FEATURE = {"Baixa": (0, 0, 0.10, 0.30), "Média": (0.15, 0.35, 0.35, 0.55), "Alta": (0.40, 0.60, 1, 1)}
MF_VOLUME = {"Baixo": (0, 0, 0.25, 0.50), "Médio": (0.25, 0.50, 0.50, 0.75), "Alto": (0.50, 0.75, 1, 1)}
MF_SAIDA = {"Baixa": (0, 0, 20, 45), "Média": (30, 50, 50, 70), "Alta": (55, 80, 100, 100)}
UNIVERSO = np.linspace(0, 100, 501)


def inferir_fuzzy(colab, feat, vol):
    """Retorna (índice 0-100, curva agregada de saída) para arrays de entrada."""
    colab, feat, vol = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (colab, feat, vol))
    c = {k: trapmf(colab, *p) for k, p in MF_COLAB.items()}
    f = {k: trapmf(feat, *p) for k, p in MF_FEATURE.items()}
    v = {k: trapmf(vol, *p) for k, p in MF_VOLUME.items()}

    regras = [
        (np.minimum(c["Alta"], f["Alta"]), "Alta"),
        (np.minimum(c["Alta"], f["Média"]), "Alta"),
        (np.minimum(c["Alta"], f["Baixa"]), "Média"),
        (np.minimum(c["Média"], f["Alta"]), "Alta"),
        (np.minimum(c["Média"], f["Média"]), "Média"),
        (np.minimum(c["Média"], f["Baixa"]), "Média"),
        (c["Baixa"], "Baixa"),
        (np.minimum(f["Alta"], v["Baixo"]), "Alta"),
        (np.minimum(v["Alto"], f["Baixa"]), "Baixa"),
    ]
    forca = {k: np.zeros_like(colab) for k in MF_SAIDA}
    for peso, saida in regras:                      # operador E = mínimo
        forca[saida] = np.maximum(forca[saida], peso)  # agregação das regras = máximo

    agregada = np.max([
        np.minimum(forca[k][:, None], trapmf(UNIVERSO, *MF_SAIDA[k])[None, :])
        for k in MF_SAIDA
    ], axis=0)                                      # implicação = corte (Mamdani)
    area = agregada.sum(axis=1)
    indice = np.where(area > 0, (agregada * UNIVERSO).sum(axis=1) / np.maximum(area, 1e-9), 50.0)
    return indice, agregada


with st.expander("📖 Como o sistema fuzzy funciona (funções de pertinência e regras)"):
    fig_mf = make_subplots(
        rows=1, cols=4,
        subplot_titles=("% colaborativo", "% como feature", "Volume (percentil)", "Saída: dependência")
    )
    grade = np.linspace(0, 1, 201)
    for col, (mfs, eixo) in enumerate(
        [(MF_COLAB, grade), (MF_FEATURE, grade), (MF_VOLUME, grade), (MF_SAIDA, UNIVERSO)], start=1
    ):
        for nome_mf, params in mfs.items():
            fig_mf.add_trace(
                go.Scatter(x=eixo, y=trapmf(eixo, *params), mode="lines", name=nome_mf,
                           showlegend=(col == 1 or col == 3)),
                row=1, col=col
            )
    fig_mf.update_layout(height=300, margin=dict(t=50), legend_title="Conjunto")
    st.plotly_chart(fig_mf, use_container_width=True)
    st.markdown(
        """
**Entradas:** % colaborativo = streams colaborativos ÷ total; % feature = streams como convidado ÷ total;
volume = posição (percentil) do artista em streams totais.

| # | Regra |
|---|-------|
| 1 | SE colaborativo é Alta E feature é Alta → dependência **Alta** |
| 2 | SE colaborativo é Alta E feature é Média → **Alta** |
| 3 | SE colaborativo é Alta E feature é Baixa → **Média** |
| 4 | SE colaborativo é Média E feature é Alta → **Alta** |
| 5 | SE colaborativo é Média E feature é Média → **Média** |
| 6 | SE colaborativo é Média E feature é Baixa → **Média** |
| 7 | SE colaborativo é Baixa → **Baixa** |
| 8 | SE feature é Alta E volume é Baixo → **Alta** |
| 9 | SE volume é Alto E feature é Baixa → **Baixa** |

Defuzzificação pelo **centroide** da área agregada.
        """
    )

df_fz = df_filtrado[[
    "Artist Name", "Primary Genre", "Country of Origin", "Total Streams (in millions)",
    "% Colaborativo", "% Feature", "Streams Percentil"
]].dropna().copy()

if df_fz.empty:
    st.info("Sem dados válidos para o sistema fuzzy com os filtros atuais.")
else:
    df_fz["Índice fuzzy"], _ = inferir_fuzzy(
        df_fz["% Colaborativo"], df_fz["% Feature"], df_fz["Streams Percentil"]
    )
    df_fz["Perfil"] = pd.cut(
        df_fz["Índice fuzzy"], bins=[-1, 35, 65, 101],
        labels=["Independente", "Equilibrado", "Dependente de colaborações"]
    )

    total_fz = df_fz["Total Streams (in millions)"]
    tam_fz = np.clip(np.sqrt(total_fz / total_fz.median()) * 8, 6, 28)

    fig_fz = go.Figure(go.Scatter(
        x=df_fz["% Colaborativo"] * 100, y=df_fz["% Feature"] * 100, mode="markers",
        text=df_fz["Artist Name"],
        customdata=np.column_stack([
            df_fz["Primary Genre"], df_fz["Índice fuzzy"],
            df_fz["Perfil"].astype(str), total_fz
        ]),
        marker=dict(
            size=tam_fz, color=df_fz["Índice fuzzy"], colorscale="RdYlGn_r",
            cmin=0, cmax=100, opacity=0.85, line=dict(width=0.5, color="white"),
            colorbar=dict(title="Índice fuzzy")
        ),
        hovertemplate=(
            "<b>%{text}</b> (%{customdata[0]})<br>"
            "% colaborativo: %{x:.1f}%<br>% como feature: %{y:.1f}%<br>"
            "Streams totais: %{customdata[3]:,.1f} M<br>"
            "Índice fuzzy: %{customdata[1]:.1f}<br>Perfil: %{customdata[2]}<extra></extra>"
        )
    ))
    fig_fz.update_layout(
        title="Índice fuzzy de dependência colaborativa por artista "
              "(cor = índice, tamanho = streams totais)",
        xaxis_title="% dos streams que vêm de colaborações",
        yaxis_title="% dos streams que vêm como feature (convidado)",
        height=620
    )
    st.plotly_chart(fig_fz, use_container_width=True)

    f1, f2 = st.columns([1, 2])
    contagem = (
        df_fz["Perfil"].value_counts()
        .reindex(["Independente", "Equilibrado", "Dependente de colaborações"]).fillna(0)
    )
    with f1:
        fig_cat = go.Figure(go.Bar(
            x=contagem.index.astype(str), y=contagem.values,
            marker_color=["#2ECC71", "#F1C40F", "#E74C3C"]
        ))
        fig_cat.update_layout(title="Artistas por perfil", height=350,
                              yaxis_title="Artistas", margin=dict(t=50))
        st.plotly_chart(fig_cat, use_container_width=True)
    with f2:
        cols_tab = ["Artist Name", "Primary Genre", "Total Streams (in millions)", "Índice fuzzy"]
        aba1, aba2 = st.tabs(["🔺 Mais dependentes", "🔻 Mais independentes"])
        aba1.dataframe(df_fz.nlargest(10, "Índice fuzzy")[cols_tab].round(1),
                       use_container_width=True, hide_index=True)
        aba2.dataframe(df_fz.nsmallest(10, "Índice fuzzy")[cols_tab].round(1),
                       use_container_width=True, hide_index=True)

    with st.expander("🧪 Simulador: teste um artista hipotético"):
        q1, q2, q3 = st.columns(3)
        sim_c = q1.slider("% colaborativo", 0.0, 1.0, 0.6, 0.01, key="fz_c")
        sim_f = q2.slider("% como feature", 0.0, 1.0, 0.4, 0.01, key="fz_f")
        sim_v = q3.slider("Volume (percentil)", 0.0, 1.0, 0.5, 0.01, key="fz_v")
        idx_sim, agg_sim = inferir_fuzzy(sim_c, sim_f, sim_v)
        st.metric("Índice fuzzy resultante", f"{idx_sim[0]:.1f} / 100")
        fig_sim = go.Figure(go.Scatter(x=UNIVERSO, y=agg_sim[0], fill="tozeroy",
                                       name="Saída agregada"))
        fig_sim.add_vline(x=float(idx_sim[0]), line_dash="dash", line_color="#E74C3C",
                          annotation_text="centroide")
        fig_sim.update_layout(height=300, xaxis_title="Dependência colaborativa",
                              yaxis_title="Pertinência", margin=dict(t=30))
        st.plotly_chart(fig_sim, use_container_width=True)

# TABELA DOS DADOS
st.subheader("📋 Dados dos artistas")
st.dataframe(df_filtrado, use_container_width=True)

# FINAL
st.success("Dashboard carregado com sucesso!")