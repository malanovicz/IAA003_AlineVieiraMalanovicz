
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter
import statsmodels.api as sm
import statsmodels.stats.weightstats as st_stats
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import norm

st.set_page_config(page_title="Dashboard Netflix - Trabalho Final", layout="wide")

st.title("🎬 Trabalho Final de IAA003 - Probabilidade e Estatística para IA")
st.subheader("Análise Estatística do Catálogo da Netflix - Aline Vieira Malanovicz")

# Cache do carregamento de dados para performance
@st.cache_data
def load_data():
    file_path = "netflix_titles.csv"
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "shivamb/netflix-shows",
        file_path
    )
    # Preparando as populações
    movies_df = df[df['type'] == 'Movie'].copy()
    movies_df['duration_mins'] = movies_df['duration'].str.replace(' min', '', silent=True).str.replace(' min', '').astype(float)
    
    series_df = df[df['type'] == 'TV Show'].copy()
    series_df['release_year'] = series_df['release_year'].astype(int)
    
    return df, movies_df, series_df

try:
    df, movies_df, series_df = load_data()
    st.success("Dados carregados com sucesso diretamente do Kagglehub!")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# Sidebar para navegação
menu = st.sidebar.radio("Navegue pelas Seções:", 
    ["Visão Geral da População", "Amostragem e Estimação", "Testes de Hipóteses"]
)

if menu == "Visão Geral da População":
    st.header("📊 Distribuição da População (Catálogo Completo)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Proporção Geral de Títulos:**")
        fig, ax = plt.subplots(figsize=(6, 4))
        df['type'].value_counts().plot(kind='bar', color='turquoise', ax=ax)
        ax.set_title("Filmes vs Séries")
        st.pyplot(fig)
        
        proporcao = df['type'].value_counts(normalize=True) * 100
        st.write(f"**Filmes:** {proporcao['Movie']:.2f}% | **Séries:** {proporcao['TV Show']:.2f}%")

    with col2:
        st.write("**Duração dos Filmes (Minutos):**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(movies_df['duration_mins'].dropna(), bins=30, color='indianred', edgecolor='black')
        ax.set_title("Distribuição da Duração dos Filmes")
        st.pyplot(fig)
        st.write(f"Média Populacional de Duração: {movies_df['duration_mins'].mean():.2f} minutos")

elif menu == "Amostragem e Estimação":
    st.header("🎲 Amostragem e Intervalos de Confiança (n=369)")
    st.write("Utilizando amostra de tamanho 369 (grau de confiança de 95% e margem de erro de 5%).")
    
    # Gerar amostras a partir do estado fixado para consistência
    np.random.seed(42)
    amostra_duracao = movies_df['duration_mins'].dropna().sample(369)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Duração dos Filmes")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(amostra_duracao, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        ax.set_title("Amostra: Duração de Filmes")
        st.pyplot(fig)
        
        mean_amostra = amostra_duracao.mean()
        se_duracao = np.std(amostra_duracao) / np.sqrt(369)
        lower = mean_amostra - (1.96 * se_duracao)
        upper = mean_amostra + (1.96 * se_duracao)
        
        st.metric("Média da Amostra", f"{mean_amostra:.2f} min")
        st.write(f"**Intervalo de Confiança (95%):** [{lower:.2f} min, {upper:.2f} min]")
        st.write(f"Média Real da População: {movies_df['duration_mins'].mean():.2f} min")

    with col2:
        st.subheader("Proporção do Catálogo")
        amostra_tipo = df['type'].dropna().sample(369)
        counts = amostra_tipo.value_counts()
        st.write(counts)
        
        st.write("**Proporção estimada:**")
        st.write(amostra_tipo.value_counts(normalize=True) * 100)

elif menu == "Testes de Hipóteses":
    st.header("🧪 Testes Estatísticos")
    
    st.subheader("1. Comparação dos Anos de Lançamento (Filmes vs Séries)")
    df_ano_filmes = df[df['type'] == 'Movie']['release_year'].dropna()
    df_ano_series = df[df['type'] == 'TV Show']['release_year'].dropna()
    
    dsw1 = st_stats.DescrStatsW(df_ano_filmes)
    dsw2 = st_stats.DescrStatsW(df_ano_series)
    cm = st_stats.CompareMeans(dsw1, dsw2)
    ht = cm.ztest_ind(alternative='two-sided', usevar='unequal', value=0)
    
    st.write(f"**Estatística Z:** {ht[0]:.4f}")
    st.write(f"**p-valor:** {ht[1]:.4g}")
    if ht[1] < 0.05:
        st.error("Resultado: Rejeitar a hipótese nula (Diferença estatisticamente significativa)")
    else:
        st.success("Resultado: Aceitar a hipótese nula")
        
    st.subheader("2. Proporção de Séries por Século (Século 20 vs 21)")
    pop_df = df.copy()
    pop_df['periodo'] = pop_df['release_year'].apply(lambda x: 'Século 20' if x < 2001 else 'Século 21')
    amostra_df = pop_df.sample(n=369, random_state=42)
    tabela_amostra = pd.crosstab(amostra_df['periodo'], amostra_df['type'])
    
    try:
        s_seculo20 = tabela_amostra.loc['Século 20', 'TV Show']
        s_seculo21 = tabela_amostra.loc['Século 21', 'TV Show']
        n_seculo20 = tabela_amostra.loc['Século 20'].sum()
        n_seculo21 = tabela_amostra.loc['Século 21'].sum()
        
        successes = np.array([s_seculo20, s_seculo21])
        nobs = np.array([n_seculo20, n_seculo21])
        z_stat, p_val = sm.stats.proportions_ztest(successes, nobs, alternative='two-sided')
        
        st.write(f"**Século 20 (Séries):** {s_seculo20}/{n_seculo20} ({s_seculo20/n_seculo20:.2%})")
        st.write(f"**Século 21 (Séries):** {s_seculo21}/{n_seculo21} ({s_seculo21/n_seculo21:.2%})")
        st.write(f"**Estatística Z (Proporções):** {z_stat:.4f} | **p-valor:** {p_val:.4f}")
    except Exception as e:
        st.warning("Amostra insuficiente para calcular as proporções dos dois séculos.")

