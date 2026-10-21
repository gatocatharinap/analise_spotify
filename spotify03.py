import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

# Bibliotecas de Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN

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
# DBSCAN: AGRUPAMENTO DE ARTISTAS
# =========================================================
st.subheader("🔎 DBSCAN: Agrupamento de artistas por streams")
st.write("Identificação de grupos de artistas com padrões semelhantes de streams solo e colaborativos.")

colunas_dbscan = [
    "Artist Name",
    "Solo Streams (in millions)",
    "Collaborative Streams (in millions)"
]
df_dbscan = df_filtrado[colunas_dbscan].copy()

colunas_features = [
    "Solo Streams (in millions)",
    "Collaborative Streams (in millions)"
]

for coluna in colunas_features:
    df_dbscan[coluna] = pd.to_numeric(df_dbscan[coluna], errors="coerce")

df_dbscan = df_dbscan.dropna(subset=colunas_features)
df_dbscan = df_dbscan[(df_dbscan[colunas_features] >= 0).all(axis=1)].copy()

if len(df_dbscan) < 2:
    st.warning("Não há artistas suficientes com dados válidos para executar o DBSCAN.")
else:
    col_eps, col_min_samples = st.columns(2)

    with col_eps:
        eps = st.slider(
            "Distância eps",
            min_value=0.1, max_value=2.0, value=0.5, step=0.1,
            help="Controla a distância entre vizinhos no espaço dos dados padronizados."
        )

    with col_min_samples:
        min_samples = st.slider("Mínimo de pontos por vizinhança", min_value=2, max_value=15, value=5, step=1)

    X_log = np.log1p(df_dbscan[colunas_features].to_numpy())
    scaler_dbscan = StandardScaler()
    X_scaled = scaler_dbscan.fit_transform(X_log)

    modelo_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_dbscan["Cluster"] = modelo_dbscan.fit_predict(X_scaled)

    quantidade_ruido = int((df_dbscan["Cluster"] == -1).sum())
    quantidade_clusters = len(set(df_dbscan["Cluster"]) - {-1})

    total_dbscan, ruido_dbscan, percentual_ruido = st.columns(3)
    total_dbscan.metric("🎯 Clusters encontrados", quantidade_clusters)
    ruido_dbscan.metric("⚠️ Artistas classificados como ruído", quantidade_ruido)
    
    percentual = (100 * quantidade_ruido / len(df_dbscan)) if len(df_dbscan) > 0 else 0
    percentual_ruido.metric("📊 Percentual de ruído", f"{percentual:.1f}%")

    df_dbscan["Grupo"] = df_dbscan["Cluster"].apply(
        lambda x: "Ruído (-1)" if x == -1 else f"Cluster {x}"
    )

    fig_dbscan = go.Figure()
    grupos = sorted(
        df_dbscan["Grupo"].unique(),
        key=lambda grupo: (grupo == "Ruído (-1)", grupo)
    )

    for grupo in grupos:
        df_grupo = df_dbscan[df_dbscan["Grupo"] == grupo]
        eh_ruido = grupo == "Ruído (-1)"

        fig_dbscan.add_trace(
            go.Scatter(
                x=df_grupo["Solo Streams (in millions)"],
                y=df_grupo["Collaborative Streams (in millions)"],
                mode="markers",
                name=grupo,
                text=df_grupo["Artist Name"],
                customdata=df_grupo[["Cluster"]].to_numpy(),
                marker=dict(
                    size=10 if eh_ruido else 9,
                    color="#E74C3C" if eh_ruido else None,
                    symbol="x" if eh_ruido else "circle",
                    opacity=0.85
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Streams solo: %{x:,.2f} milhões<br>"
                    "Streams colaborativos: %{y:,.2f} milhões<br>"
                    "Cluster: %{customdata[0]}"
                    "<extra></extra>"
                )
            )
        )

    fig_dbscan.update_layout(
        title="Clusters de artistas identificados pelo DBSCAN",
        xaxis_title="Streams solo (milhões)",
        yaxis_title="Streams colaborativos (milhões)",
        legend_title="Grupo identificado",
        hovermode="closest",
        height=600
    )

    st.plotly_chart(fig_dbscan, use_container_width=True)

    st.subheader("📋 Resumo dos grupos")
    resumo_dbscan = (
        df_dbscan
        .groupby("Grupo")
        .agg(
            Quantidade_Artistas=("Artist Name", "size"),
            Media_Streams_Solo=("Solo Streams (in millions)", "mean"),
            Media_Streams_Colaborativos=("Collaborative Streams (in millions)", "mean")
        )
        .round(2)
        .reset_index()
    )
    st.dataframe(resumo_dbscan, use_container_width=True, hide_index=True)

    st.subheader("⚠️ Artistas fora dos clusters")
    df_ruido = df_dbscan[df_dbscan["Cluster"] == -1].sort_values("Solo Streams (in millions)", ascending=False)

    if not df_ruido.empty:
        st.dataframe(
            df_ruido[["Artist Name", "Solo Streams (in millions)", "Collaborative Streams (in millions)"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Nenhum artista foi classificado como ruído com os parâmetros atuais.")

# TABELA DOS DADOS
st.subheader("📋 Dados dos artistas")
st.dataframe(df_filtrado, use_container_width=True)

# FINAL
st.success("Dashboard carregado com sucesso!")