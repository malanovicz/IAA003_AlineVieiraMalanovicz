
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

# Seção fixa de identificação da Fonte e Base de Dados Original no menu lateral
st.sidebar.markdown("--- ")
st.sidebar.subheader("ℹ️ Informações do Projeto")
st.sidebar.markdown("**Fonte Original dos Dados:**")
st.sidebar.markdown("[Kaggle - Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)")
st.sidebar.markdown("**Identificação da Fonte:**")
st.sidebar.caption("Dataset público compilado por Shivam Bansal, contendo todo o catálogo de filmes e séries disponíveis na plataforma da Netflix até 2021.")
st.sidebar.markdown("**Disponibilidade:**")
st.sidebar.caption("Este painel e os dados de origem permanecerão públicos e acessíveis de forma ininterrupta durante todo o período de avaliação do curso.")

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
            ax.boxplot(sampl1_duracao, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            ax.set_title("Amostra: Duração de Filmes")
            st.pyplot(fig)
        st.write("**Estatísticas descritivas da amostra de duração:**")
        st.write(sampl1_duracao.describe())

    with tab_sampl2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            tag = sampl2_tipo.value_counts()
            tag.plot(kind='bar', color=['lightblue', 'lightgray'], ax=ax)
            ax.set_title("Amostra: Tipo (Filmes ou Séries) da Netflix")
            st.pyplot(fig)
        with col2:
            st.write("**Contagens da Amostra:**")
            st.write(sampl2_tipo.value_counts())
            st.write("**Proporção estimada da Amostra (%):**")
            st.write(sampl2_tipo.value_counts(normalize=True) * 100)

    with tab_sampl3:
        st.subheader("Ano de Lançamento (Filmes vs. Séries na Amostra)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Filmes (Amostra):**")
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.hist(sampl3_ano_filmes, edgecolor='black', bins=50, color='darkblue')
            ax1.set_title("Histograma: Ano de Lançamento (Filmes Amostra)")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.boxplot(sampl3_ano_filmes, vert=False, patch_artist=True, boxprops=dict(facecolor='darkblue'))
            ax2.set_title("Boxplot: Ano de Lançamento (Filmes Amostra)")
            st.pyplot(fig2)
            st.write(sampl3_ano_filmes.describe())
        with col2:
            st.markdown("**Séries (Amostra):**")
            fig3, ax3 = plt.subplots(figsize=(8, 4.5))
            ax3.hist(sampl4_ano_series, edgecolor='black', bins=50, color='slateblue')
            ax3.set_title("Histograma: Ano de Lançamento (Séries Amostra)")
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(8, 3))
            ax4.boxplot(sampl4_ano_series, vert=False, patch_artist=True, boxprops=dict(facecolor='darkblue'))
            ax4.set_title("Boxplot: Ano de Lançamento (Séries Amostra)")
            st.pyplot(fig4)
            st.write(sampl4_ano_series.describe())

elif menu == "3: inferência estatística e comparação dos resultados":
    st.header("🧪 3: Inferência Estatística e Comparação de Resultados")

    # Preparar amostras consistentes para cálculos
    np.random.seed(42)
    sampl1_duracao = netflix_population.sample(369)
    sampl2_tipo = tipo.sample(369)
    sampl3_ano_filmes = price.sample(369)
    sampl4_ano_series = series_population.sample(369)

    # Separando os testes em duas abas
    tab_media, tab_proporcao = st.tabs(["Inferência para Média", "Inferência para Proporção"])

    with tab_media:
        st.subheader("Análise de Inferência para Médias")

        st.markdown("#### Intervalo de Confiança: Duração Média de Filmes")
        mean_val = sampl1_duracao.mean()
        se_dur = np.std(sampl1_duracao) / np.sqrt(369)
        lower_dur = mean_val - (1.96 * se_dur)
        upper_dur = mean_val + (1.96 * se_dur)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dados do Cálculo Praticado:**")
            calc_info = (
                "Tamanho da amostra (n) = 369\n"
                "Média amostral = {:.6f} min\n"
                "Desvio Padrão Amostra = {:.6f} min\n"
                "Erro Padrão calculado = {:.6f}\n"
                "Z para 95% = 1.9600\n"
                "Fórmula: Média +/- (Z * Erro_Padrão)"
            ).format(mean_val, np.std(sampl1_duracao), se_dur)
            st.code(calc_info)
            st.write(f"- Intervalo Calculado (95%): **[{lower_dur:.4f}, {upper_dur:.4f}]**")
            st.write(f"- Média real populacional correspondente: **{netflix_population.mean():.4f} min**")
            st.success("Conclusão: A média real populacional pertence ao intervalo de confiança estimado!")

        st.markdown("--- ")
        st.markdown("#### Teste de Hipótese A: Anos Médios de Lançamento (Filmes vs. Séries)")
        st.markdown('**Hipóteses:**\n- H0: Média_Séries = Média_Filmes\n- H1: Média_Séries != Média_Filmes')

        df_ano_filmes = df[df['type'] == 'Movie']['release_year'].dropna()
        df_ano_series = df[df['type'] == 'TV Show']['release_year'].dropna()
        dsw1 = st_stats.DescrStatsW(df_ano_filmes)
        dsw2 = st_stats.DescrStatsW(df_ano_series)
        cm = st_stats.CompareMeans(dsw1, dsw2)
        ht = cm.ztest_ind(alternative='two-sided', usevar='unequal', value=0)
        se_test_a = cm.std_meandiff_separatevar

        st.markdown("**Resultados dos Cálculos:**")
        test_info = (
            "n Filmes = {} | n Séries = {}\n"
            "Média Filmes = {:.4f} | Média Séries = {:.4f}\n"
            "Standard Error de Diferença = {:.6f}\n"
            "Estatística de Teste Z = {:.4f}\n"
            "P-valor correspondente = {:.4g}"
        ).format(dsw1.nobs, dsw2.nobs, dsw1.mean, dsw2.mean, se_test_a, ht[0], ht[1])
        st.code(test_info)

        if ht[1] < 0.05:
            st.error(f"**Decisão**: Rejeitamos a hipótese nula (H0). O p-valor ({ht[1]:.4g}) é menor que alpha (5%). Existe diferença estatisticamente significativa entre os anos médios de lançamento de filmes e séries.")
        else:
            st.success("**Decisão**: Não rejeitamos a hipótese nula (H0).")

    with tab_proporcao:
        st.subheader("Análise de Inferência para Proporções")

        st.markdown("#### Intervalo de Confiança para Proporções (Filmes vs. Séries)")
        conf_lvl = 0.95
        z_value = norm.ppf((1 - (1 - conf_lvl) / 2))
        prbs = tipo.value_counts(normalize=True)
        se_prop = np.sqrt(prbs.prod() / len(tipo))
        ci1 = prbs['Movie'] - z_value * se_prop
        ci2 = prbs['Movie'] + z_value * se_prop

        st.markdown("**Dados do Cálculo Praticado (População total como base de proporção):**")
        prop_info = (
            "Proporção Filmes (p) = {:.6f}\n"
            "Proporção Séries (q) = {:.6f}\n"
            "Erro Padrão Proporção = {:.6f}\n"
            "Escore Z (95%) = {:.4f}"
        ).format(prbs['Movie'], prbs['TV Show'], se_prop, z_value)
        st.code(prop_info)
        st.write(f"- Intervalo de confiança para Filmes (95%): **[{ci1:.4f}, {ci2:.4f}]**")
        st.write(f"- Proporção populacional real de Filmes: **{prbs['Movie']:.4%}**")

        st.markdown("--- ")
        st.markdown("#### Teste de Hipótese B: Proporção de Séries por Século (Século 20 vs. 21)")
        st.markdown('**Hipóteses:**\n- H0: Proporção_Séc20 = Proporção_Séc21\n- H1: Proporção_Séc20 != Proporção_Séc21')

        pop_df = df.copy()
        pop_df['periodo'] = pop_df['release_year'].apply(lambda x: 'Século 20' if x < 2001 else 'Século 21')
        amostra_df = pop_df.sample(n=369, random_state=42)

        tabela_pop = pd.crosstab(pop_df['periodo'], pop_df['type'])
        tabela_pop_prop = pd.crosstab(pop_df['periodo'], pop_df['type'], normalize='index')

        tabela_amostra = pd.crosstab(amostra_df['periodo'], amostra_df['type'])
        tabela_amostra_prop = pd.crosstab(amostra_df['periodo'], amostra_df['type'], normalize='index')

        try:
            s_seculo20 = tabela_amostra.loc['Século 20', 'TV Show']
            s_seculo21 = tabela_amostra.loc['Século 21', 'TV Show']
            n_seculo20 = tabela_amostra.loc['Século 20'].sum()
            n_seculo21 = tabela_amostra.loc['Século 21'].sum()

            successes = np.array([s_seculo20, s_seculo21])
            nobs = np.array([n_seculo20, n_seculo21])
            z_stat, p_val = sm.stats.proportions_ztest(successes, nobs, alternative='two-sided')

            st.markdown("**Tabelas de Contingência e Proporções (Prints Numéricos):**")
            col_tab1, col_tab2 = st.columns(2)
            with col_tab1:
                st.write("**Tabela da População Geral (Quantidades):**")
                st.dataframe(tabela_pop)
                st.write("**Tabela da População Geral (Proporções):**")
                st.dataframe(tabela_pop_prop)
            with col_tab2:
                st.write("**Tabela da Amostra (Quantidades):**")
                st.dataframe(tabela_amostra)
                st.write("**Tabela da Amostra (Proporções):**")
                st.dataframe(tabela_amostra_prop)

            st.markdown("**Resultados dos Cálculos do Teste Z:**")
            z_info = (
                "Sucessos Século 20 = {} de {} ({:.4f})\n"
                "Sucessos Século 21 = {} de {} ({:.4f})\n"
                "Estatística Z do Teste = {:.4f}\n"
                "P-valor do Teste = {:.4g}"
            ).format(s_seculo20, n_seculo20, s_seculo20/n_seculo20, s_seculo21, n_seculo21, s_seculo21/n_seculo21, z_stat, p_val)
            st.code(z_info)

            if p_val < 0.05:
                st.error("**Decisão**: Rejeitamos a hipótese nula (H0). A proporção de séries em relação a filmes é significativamente diferente entre os séculos na amostra.")
            else:
                st.success("**Decisão**: Não rejeitamos a hipótese nula (H0).")

            st.markdown("**Visualização dos Mosaicos Comparativos (População vs. Amostra):**")
            col_mos1, col_mos2 = st.columns(2)
            props = lambda key: {'color': 'indianred' if 'Movie' in key else 'lightgreen'}

            with col_mos1:
                fig_m_pop, ax_m_pop = plt.subplots(figsize=(8, 6))
                mosaic(pop_df, ['periodo', 'type'], title='Mosaico da População Geral', properties=props, ax=ax_m_pop)
                st.pyplot(fig_m_pop)

            with col_mos2:
                fig_m_sample, ax_m_sample = plt.subplots(figsize=(8, 6))
                mosaic(amostra_df, ['periodo', 'type'], title='Mosaico da Amostra (n=369)', properties=props, ax=ax_m_sample)
                st.pyplot(fig_m_sample)

        except Exception as e:
            st.warning(f"Ocorreu um erro no cálculo do Teste B: {e}")

