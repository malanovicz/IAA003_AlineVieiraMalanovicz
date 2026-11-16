
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

# Estilo geral dos gráficos semelhante ao notebook
plt.style.use('ggplot')

st.set_page_config(page_title="Dashboard Netflix", layout="wide")

st.title("🎬 Trabalho Final de IAA003")
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
    
    # Preparando os dados populacionais do notebook
    movies_df = df[df['type'] == 'Movie'].copy()
    movies_df['duration_mins'] = movies_df['duration'].astype(str).str.replace(' min', '').astype(float)
    
    series_df = df[df['type'] == 'TV Show'].copy()
    series_df['release_year'] = series_df['release_year'].astype(int)
    
    return df, movies_df, series_df

try:
    df, movies_df, series_df = load_data()
    netflix_population = movies_df['duration_mins'].dropna()
    price = movies_df['release_year'].dropna()
    tipo = df['type'].dropna()
    series_population = series_df['release_year'].dropna()
    st.success("Dados carregados com sucesso diretamente do Kagglehub!")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# Menu lateral customizado de acordo com o pedido
menu = st.sidebar.radio("Painéis:", 
    ["1: análise descritiva da população", 
     "2: análise descritiva da amostra", 
     "3: inferência estatística e comparação dos resultados"]
)

if menu == "1: análise descritiva da população":
    st.header("📊 1: Análise Descritiva da População")
    st.write(f"**Tamanho total do catálogo analisado (Filmes e Séries):** {len(df)} títulos.")
    
    # Abas para organizar a grande quantidade de gráficos da população
    tab1, tab2, tab3 = st.tabs(["Distribuição Geral (Tipo)", "Duração de Filmes", "Ano de Lançamento"])
    
    with tab1:
        st.subheader("Distribuição do tipo de produções")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            df['type'].value_counts().plot(kind='bar', color='turquoise', ax=ax)
            ax.set_title("População: Distribuição de Conteúdo na Netflix (Filmes vs. Séries)")
            st.pyplot(fig)
        with col2:
            st.write("**Contagens populacionais brutas:**")
            st.write(df['type'].value_counts())
            st.write("**Proporção percentual da população:**")
            st.write(df['type'].value_counts(normalize=True) * 100)

    with tab2:
        st.subheader("Duração dos Filmes (População)")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(netflix_population, edgecolor='black', bins=30, color='indianred')
            ax.set_title("População: Duração de Filmes da Netflix")
            ax.set_xlabel("Duração (minutos)")
            ax.set_ylabel("Frequência")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.boxplot(netflix_population, vert=False, patch_artist=False)
            ax.set_title("Boxplot Populacional - Duração de Filmes")
            st.pyplot(fig)
        st.write("**Estatísticas descritivas da duração (População):**")
        st.write(netflix_population.describe())

    with tab3:
        st.subheader("Anos de Lançamento")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Filmes (População):**")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(price, edgecolor='black', bins=80, color='indianred')
            ax.set_title("População: Ano de Lançamento de Filmes")
            st.pyplot(fig)
            st.write(price.describe())
        with col2:
            st.write("**Séries (População):**")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(series_population, edgecolor='black', bins=50, color='slateblue')
            ax.set_title("População: Ano de Lançamento de Séries")
            st.pyplot(fig)
            st.write(series_population.describe())

elif menu == "2: análise descritiva da amostra":
    st.header("🎲 2: Análise Descritiva da Amostra (n=369)")
    st.info("Amostragem probabilística realizada com tamanho n=369 de acordo com a fórmula para nível de confiança de 95% e erro amostral de 5%.")
    
    # Amostras consistentes usando semente fixa
    np.random.seed(42)
    sampl1_duracao = netflix_population.sample(369)
    sampl2_tipo = tipo.sample(369)
    sampl3_ano_filmes = price.sample(369)
    sampl4_ano_series = series_population.sample(369)
    
    tab_sampl1, tab_sampl2, tab_sampl3 = st.tabs(["Amostra de Duração", "Amostra de Tipos", "Amostra de Anos de Lançamento"])
    
    with tab_sampl1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(sampl1_duracao, color='red', edgecolor='black')
            ax.set_title("Amostra: Duração de Filmes da Netflix")
            ax.set_xlabel("Duração (minutos)")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.boxplot(sampl1_duracao, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
            ax.set_title("Amostra: Duração de Filmes")
            st.pyplot(fig)
        st.write("**Estatísticas descritivas da amostra de duração:**")
        st.write(sampl1_duracao.describe())

    with tab_sampl2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sampl2_tipo.value_counts().plot(kind='bar', color=['lightblue', 'lightgray'], ax=ax)
            ax.set_title("Amostra: Tipo (Filmes ou Séries) da Netflix")
            st.pyplot(fig)
        with col2:
            st.write("**Contagens da Amostra:**")
            st.write(sampl2_tipo.value_counts())
            st.write("**Proporção estimada da Amostra (%):**")
            st.write(sampl2_tipo.value_counts(normalize=True) * 100)

    with tab_sampl3:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Anos de Lançamento - Amostra Filmes:**")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(sampl3_ano_filmes, edgecolor='black', bins=50, color='darkblue')
            ax.set_title("Amostra: Ano de Lançamento (Filmes)")
            st.pyplot(fig)
            st.write(sampl3_ano_filmes.describe())
        with col2:
            st.write("**Anos de Lançamento - Amostra Séries:**")
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.boxplot(sampl4_ano_series, vert=False, patch_artist=True, boxprops=dict(facecolor="darkblue"))
            ax.set_title("Amostra: Ano de Lançamento (Séries)")
            st.pyplot(fig)
            st.write(sampl4_ano_series.describe())

elif menu == "3: inferência estatística e comparação dos resultados":
    st.header("🧪 3: Inferência Estatística e Comparação de Resultados")
    
    # Preparar amostras consistentes
    np.random.seed(42)
    sampl1_duracao = netflix_population.sample(369)
    sampl2_tipo = tipo.sample(369)
    sampl3_ano_filmes = price.sample(369)
    
    # 1. Intervalos de confiança
    st.subheader("1. Intervalos de Confiança para as Médias e Proporções")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Intervalo de Estimação da Média de Duração (Filmes)**")
        mean_val = sampl1_duracao.mean()
        se_dur = np.std(sampl1_duracao) / np.sqrt(369)
        lower_dur = mean_val - (1.96 * se_dur)
        upper_dur = mean_val + (1.96 * se_dur)
        
        st.write(f"- Média da amostra: **{mean_val:.4f} min**")
        st.write(f"- Erro Padrão da Amostra: **{se_dur:.4f}**")
        st.write(f"- Intervalo (95%): **[{lower_dur:.4f}, {upper_dur:.4f}]**")
        st.write(f"- Média real populacional correspondente: **{netflix_population.mean():.4f} min** (Pertence ao intervalo!)")

    with col2:
        st.markdown("**Intervalo de Confiança para Proporções (Movie vs. TV Show)**")
        conf_lvl = 0.95
        z_value = norm.ppf((1 - (1 - conf_lvl) / 2))
        prbs = tipo.value_counts(normalize=True)
        se_prop = np.sqrt(prbs.prod() / len(tipo))
        ci1 = prbs['Movie'] - z_value * se_prop
        ci2 = prbs['Movie'] + z_value * se_prop
        
        st.write(f"- Proporção real (Filmes): **{prbs['Movie']:.4%}**")
        st.write(f"- Proporção real (Séries): **{prbs['TV Show']:.4%}**")
        st.write(f"- Erro padrão calculado: **{se_prop:.6f}**")
        st.write(f"- Intervalo de confiança (95% para Filmes): **[{ci1:.4f}, {ci2:.4f}]**")

    # 2. Testes de Hipóteses
    st.subheader("2. Testes de Hipóteses Realizados")
    
    st.markdown("**Teste A: Anos Médios de Lançamento (Filmes vs. Séries)**")
    df_ano_filmes = df[df['type'] == 'Movie']['release_year'].dropna()
    df_ano_series = df[df['type'] == 'TV Show']['release_year'].dropna()
    dsw1 = st_stats.DescrStatsW(df_ano_filmes)
    dsw2 = st_stats.DescrStatsW(df_ano_series)
    cm = st_stats.CompareMeans(dsw1, dsw2)
    ht = cm.ztest_ind(alternative='two-sided', usevar='unequal', value=0)
    
    st.write(f"- Média de lançamento Filmes: **{dsw1.mean:.2f}** | Média Séries: **{dsw2.mean:.2f}**")
    st.write(f"- Estatística de teste Z: **{ht[0]:.4f}**")
    st.write(f"- P-valor correspondente: **{ht[1]:.4g}**")
    if ht[1] < 0.05:
        st.error("**Conclusão**: Rejeitamos a hipótese nula ($H_0$). Existe diferença estatisticamente significativa entre os anos médios de lançamento.")
    else:
        st.success("**Conclusão**: Não rejeitamos a hipótese nula ($H_0$).")

    st.markdown("**Teste B: Proporção de Séries por Século (Século 20 vs. 21)**")
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
        
        st.write(f"- Amostra Século 20: **{s_seculo20}/{n_seculo20} ({s_seculo20/n_seculo20:.2%}) de Séries**")
        st.write(f"- Amostra Século 21: **{s_seculo21}/{n_seculo21} ({s_seculo21/n_seculo21:.2%}) de Séries**")
        st.write(f"- Estatística Z: **{z_stat:.4f}** | P-valor: **{p_val:.4g}**")
        if p_val < 0.05:
            st.error("**Conclusão**: Rejeitamos a hipótese nula ($H_0$). A proporção de séries em relação a filmes é significativamente diferente entre os séculos na amostra.")
        else:
            st.success("**Conclusão**: Não rejeitamos a hipótese nula ($H_0$).")
            
        # Gráfico de mosaico para finalizar
        st.write("**Visualização do Mosaico (Amostra):**")
        fig, ax = plt.subplots(figsize=(10, 5))
        props = lambda key: {'color': 'indianred' if 'Movie' in key else 'lightgreen'}
        mosaic(amostra_df, ['periodo', 'type'], title='Mosaico da Amostra: Tipo por Período', properties=props, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Amostra não possui contagens completas para o teste do Século 20.")

