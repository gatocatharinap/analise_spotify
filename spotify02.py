import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

# Biblioteca para SVR
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

# Filtro de gênero
generos = sorted(
    df["Primary Genre"].dropna().unique()
)

generos_selecionados = st.sidebar.multiselect(
    "🎵 Gênero musical",
    generos,
    default=generos
)

# Filtro de país
paises = sorted(
    df["Country of Origin"].dropna().unique()
)

paises_selecionados = st.sidebar.multiselect(
    "🌎 País de origem",
    paises,
    default=paises
)

# Filtro de idioma
idiomas = sorted(
    df["Primary Language"].dropna().unique()
)

idiomas_selecionados = st.sidebar.multiselect(
    "🗣️ Idioma",
    idiomas,
    default=idiomas
)

# Filtro de sexo
sexos = sorted(
    df["Sex"].dropna().unique()
)

sexos_selecionados = st.sidebar.multiselect(
    "👤 Sexo",
    sexos,
    default=sexos
)

# Filtro de tipo de artista
tipos = sorted(
    df["Artist Type"].dropna().unique()
)

tipos_selecionados = st.sidebar.multiselect(
    "🎤 Tipo de artista",
    tipos,
    default=tipos
)

# APLICANDO OS FILTROS

df_filtrado = df[
    (df["Primary Genre"].isin(generos_selecionados)) &
    (df["Country of Origin"].isin(paises_selecionados)) &
    (df["Primary Language"].isin(idiomas_selecionados)) &
    (df["Sex"].isin(sexos_selecionados)) &
    (df["Artist Type"].isin(tipos_selecionados))
]

# INDICADORES

total_streams = df_filtrado[
    "Total Streams (in millions)"
].sum()

quantidade_artistas = len(df_filtrado)

media_streams = df_filtrado[
    "Total Streams (in millions)"
].mean()

quantidade_paises = df_filtrado[
    "Country of Origin"
].nunique()


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "🎧 Total de Streams",
    f"{total_streams:,.1f} milhões"
)


col2.metric(
    "🎤 Artistas",
    quantidade_artistas
)


col3.metric(
    "📈 Média de Streams",
    f"{media_streams:,.1f} milhões"
)


col4.metric(
    "🌎 Países",
    quantidade_paises
)

# TOP 10 ARTISTAS

st.subheader("🏆 Top 10 artistas por número de streams")


top_artistas = (
    df_filtrado
    .sort_values(
        "Total Streams (in millions)",
        ascending=False
    )
    .head(10)
)


fig, ax = plt.subplots(figsize=(10, 5))


ax.barh(
    top_artistas["Artist Name"],
    top_artistas["Total Streams (in millions)"]
)


ax.set_xlabel("Streams (milhões)")
ax.set_ylabel("Artista")
ax.set_title("Top 10 artistas por streams")


# Deixa o maior artista no topo
ax.invert_yaxis()


st.pyplot(fig)

# STREAMS POR GÊNERO

st.subheader("🎵 Streams por gênero musical")


streams_genero = (
    df_filtrado
    .groupby("Primary Genre")[
        "Total Streams (in millions)"
    ]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(streams_genero)

# STREAMS POR PAÍS

st.subheader("🌎 Streams por país")


streams_pais = (
    df_filtrado
    .groupby("Country of Origin")[
        "Total Streams (in millions)"
    ]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

st.bar_chart(streams_pais)

# ANO DE ESTREIA X STREAMS

st.subheader("📅 Ano de estreia x Streams")

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(
    df_filtrado["Debut Year"],
    df_filtrado["Total Streams (in millions)"]
)

ax.set_xlabel("Ano de estreia")
ax.set_ylabel("Streams (milhões)")
ax.set_title("Relação entre ano de estreia e streams")

st.pyplot(fig)

# =========================================================
# SVM: SEXO vs. STREAMS vs. GÊNERO MUSICAL
# =========================================================

st.subheader("🤖 SVM: Classificação de Sexo por Gênero Musical e Streams")

# Seleciona as colunas necessárias e remove nulos
df_svm_sex = df_filtrado[
    [
        "Artist Name",
        "Primary Genre",
        "Total Streams (in millions)",
        "Sex",
    ]
].dropna()

# O SVM precisa de pelo menos 2 categorias de Sexo
if len(df_svm_sex) > 5 and df_svm_sex["Sex"].nunique() > 1:

    # =====================================================
    # 1. TRANSFORMA O GÊNERO EM CÓDIGO NUMÉRICO
    # =====================================================

    genre_categories = (
        df_svm_sex["Primary Genre"]
        .astype("category")
    )

    df_svm_sex["Genre_Code"] = genre_categories.cat.codes

    # =====================================================
    # 2. CRIA AS VARIÁVEIS PARA O SVM
    # =====================================================

    # O SVM continua utilizando:
    # - Gênero musical
    # - Total de Streams
    #
    # para tentar classificar o Sexo.

    X = df_svm_sex[
        [
            "Genre_Code",
            "Total Streams (in millions)"
        ]
    ].values

    # Sexo é a variável que queremos classificar
    sex_categories = df_svm_sex["Sex"].astype("category")
    y = sex_categories.cat.codes.values

    # =====================================================
    # 3. TREINA O SVM
    # =====================================================

    model = SVC(
        kernel="linear",
        C=1.0
    )

    model.fit(X, y)

    # =====================================================
    # 5. CRIA O GRÁFICO
    # =====================================================

    fig_svm_sex = go.Figure()

    # Lista de gêneros
    generos = (
        df_svm_sex["Primary Genre"]
        .dropna()
        .unique()
    )

    # =====================================================
    # 6. CADA GÊNERO SERÁ UM TIPO DE PONTO
    # =====================================================

    for genero in generos:

        df_sub = df_svm_sex[
            df_svm_sex["Primary Genre"] == genero
        ]

        fig_svm_sex.add_trace(
            go.Scatter(
                x=df_sub["Sex"],
                y=df_sub["Total Streams (in millions)"],
                mode="markers",
                name=genero,

                text=df_sub["Artist Name"],

                customdata=np.column_stack(
                    (
                        df_sub["Primary Genre"],
                        df_sub["Sex"]
                    )
                ),

                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Sexo: %{customdata[1]}<br>"
                    "Gênero: %{customdata[0]}<br>"
                    "Streams: %{y:,.1f}M"
                    "<extra></extra>"
                ),

                marker=dict(
                    size=9,
                    opacity=0.8
                )
            )
        )

    # =====================================================
    # 8. CONFIGURAÇÕES DO GRÁFICO
    # =====================================================

    fig_svm_sex.update_layout(

        title=(
            "Classificação por Sexo com base no "
            "Gênero Musical e Total de Streams"
        ),

        xaxis=dict(
            title="Sexo do Artista",
            categoryorder="array",
            categoryarray=[
                "Male",
                "Female",
                "Mixed"
            ]
        ),

        yaxis=dict(
            title="Total Streams (milhões)"
        ),

        hovermode="closest",

        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title="Gêneros Musicais"
        ),

        margin=dict(
            l=60,
            r=30,
            t=80,
            b=150
        )
    )

    st.plotly_chart(
        fig_svm_sex,
        use_container_width=True
    )

else:

    st.info(
        "Selecione mais opções no filtro de Sexo/Gênero "
        "para gerar o gráfico SVM."
    )

# STREAMS SOLO X COLABORATIVOS

st.subheader("🎤 Streams solo x colaborativos")

total_solo = df_filtrado[
    "Solo Streams (in millions)"
].sum()

total_colaborativo = df_filtrado[
    "Collaborative Streams (in millions)"
].sum()

dados_streams = pd.Series(
    {
        "Solo": total_solo,
        "Colaborativo": total_colaborativo
    }
)

st.bar_chart(dados_streams)

# TABELA DOS DADOS

st.subheader("📋 Dados dos artistas")

st.dataframe(
    df_filtrado,
    use_container_width=True
)

# FINAL

st.success(
    "Dashboard carregado com sucesso! 🎵"
)