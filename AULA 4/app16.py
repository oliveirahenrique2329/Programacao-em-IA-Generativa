"""
Analisador de Padrões em Avaliações de Clientes (com banco de dados)
--------------------------------------------------------------------
Cada avaliação digitada é guardada em um banco SQLite. O texto é tokenizado
(NLTK), limpo (sem pontuação e stopwords) e as palavras são gravadas no banco.
Com isso, o app soma as palavras de TODAS as avaliações guardadas e mostra
quais se repetem mais, ajudando a entender padrões de comportamento.

Como rodar:
    pip install streamlit nltk pandas
    python -m streamlit run app_palavras.py
"""

import sqlite3
from contextlib import closing

import nltk
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÕES
# ---------------------------------------------------------------------------

DB_ARQUIVO = "avaliacoes.db"   # o banco é criado na pasta do app
TOP_GRAFICO = 15               # quantas palavras aparecem no gráfico

AVALIACOES_EXEMPLO = [
    "O atendimento foi excelente e a entrega chegou muito rápida.",
    "Adorei o produto, a qualidade é ótima! Recomendo a todos.",
    "A entrega atrasou e o suporte demorou a responder. Fiquei chateado.",
    "Produto bom, mas o preço está caro. O atendimento poderia ser melhor.",
    "Entrega rápida, produto de qualidade e atendimento educado. Comprarei de novo!",
    "O suporte resolveu meu problema rapidamente. Ótimo atendimento.",
]


# ---------------------------------------------------------------------------
# 2. BANCO DE DADOS (SQLite)
# ---------------------------------------------------------------------------

def conectar() -> sqlite3.Connection:
    con = sqlite3.connect(DB_ARQUIVO)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def criar_tabelas():
    """Cria as tabelas na primeira execução:
    - avaliacoes: a frase original;
    - tokens: uma linha para cada palavra (token) de cada frase."""
    with closing(conectar()) as con, con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS avaliacoes (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   texto TEXT NOT NULL,
                   criado_em TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS tokens (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   avaliacao_id INTEGER NOT NULL
                       REFERENCES avaliacoes(id) ON DELETE CASCADE,
                   palavra TEXT NOT NULL)"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_tokens_palavra ON tokens(palavra)")


def salvar_avaliacao(texto: str, palavras: list[str]):
    """Grava a frase e, em seguida, cada uma das suas palavras tokenizadas."""
    with closing(conectar()) as con, con:
        cursor = con.execute("INSERT INTO avaliacoes (texto) VALUES (?)", (texto,))
        con.executemany(
            "INSERT INTO tokens (avaliacao_id, palavra) VALUES (?, ?)",
            [(cursor.lastrowid, palavra) for palavra in palavras],
        )


def frequencia_palavras() -> pd.DataFrame:
    """Soma as palavras de todas as avaliações guardadas.
    'Avaliações' mostra em quantas frases diferentes a palavra aparece:
    quanto maior, mais o comportamento se repete entre clientes."""
    consulta = """
        SELECT palavra AS "Palavra",
               COUNT(*) AS "Frequência",
               COUNT(DISTINCT avaliacao_id) AS "Avaliações"
        FROM tokens
        GROUP BY palavra
        ORDER BY "Frequência" DESC, "Palavra" ASC
    """
    with closing(conectar()) as con:
        return pd.read_sql_query(consulta, con)


def listar_avaliacoes() -> pd.DataFrame:
    """Frases guardadas, com as palavras tokenizadas de cada uma."""
    consulta = """
        SELECT a.id AS "Nº",
               a.texto AS "Avaliação",
               COALESCE(GROUP_CONCAT(t.palavra, ', '), '') AS "Tokens"
        FROM avaliacoes a
        LEFT JOIN tokens t ON t.avaliacao_id = a.id
        GROUP BY a.id
        ORDER BY a.id DESC
    """
    with closing(conectar()) as con:
        return pd.read_sql_query(consulta, con)


def limpar_banco():
    with closing(conectar()) as con, con:
        con.execute("DELETE FROM tokens")
        con.execute("DELETE FROM avaliacoes")


# ---------------------------------------------------------------------------
# 3. PROCESSAMENTO DE TEXTO
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Baixando recursos do NLTK...")
def baixar_recursos():
    """Baixa os pacotes do NLTK (só na primeira execução) e devolve as stopwords."""
    for pacote in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pacote, quiet=True)
    return set(nltk.corpus.stopwords.words("portuguese"))


def tokenizar(texto: str) -> list[str]:
    """TOKENIZAÇÃO: separa o texto em palavras e pontuações individuais."""
    return nltk.word_tokenize(texto.lower(), language="portuguese")


def limpar_tokens(tokens: list[str], stopwords: set[str]) -> list[str]:
    """Mantém só palavras (sem pontuação/números) e remove as stopwords
    (palavras comuns sem significado próprio, como 'de', 'a', 'que')."""
    return [t for t in tokens if t.isalpha() and len(t) > 2 and t not in stopwords]


def processar_e_salvar(texto: str, stopwords: set[str]) -> bool:
    """Tokeniza, limpa e guarda a frase. Devolve False se não houver palavras úteis."""
    palavras = limpar_tokens(tokenizar(texto), stopwords)
    if not palavras:
        return False
    salvar_avaliacao(texto.strip(), palavras)
    return True


# ---------------------------------------------------------------------------
# 4. INTERFACE
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Padrões nas Avaliações", page_icon="🔎")
    st.title("Padrões de comportamento nas avaliações")
    st.write(
        "Digite uma avaliação de cliente e salve. Cada frase é tokenizada e guardada "
        "no banco de dados; o app soma as palavras de todas as avaliações para "
        "mostrar o que mais se repete."
    )

    criar_tabelas()
    stopwords = baixar_recursos()

    # --- Entrada de dados ---
    with st.form("nova_avaliacao", clear_on_submit=True):
        texto = st.text_area("Digite a avaliação do cliente", height=120)
        salvar = st.form_submit_button("Salvar avaliação", type="primary")

    if salvar:
        if not texto.strip():
            st.warning("Digite algum texto antes de salvar.")
        elif processar_e_salvar(texto, stopwords):
            st.success("Avaliação salva e tokenizada!")
        else:
            st.warning("Não encontrei palavras relevantes nessa frase.")

    with st.sidebar:
        st.header("Banco de dados")
        if st.button("Carregar avaliações de exemplo"):
            for frase in AVALIACOES_EXEMPLO:
                processar_e_salvar(frase, stopwords)
            st.rerun()
        if st.button("Limpar banco"):
            limpar_banco()
            st.rerun()

    # --- Análise a partir do banco ---
    frequencia = frequencia_palavras()
    if frequencia.empty:
        st.info("Ainda não há avaliações salvas. Digite uma acima ou carregue os exemplos na barra lateral.")
        return

    avaliacoes = listar_avaliacoes()
    col1, col2, col3 = st.columns(3)
    col1.metric("Avaliações salvas", len(avaliacoes))
    col2.metric("Palavras relevantes", int(frequencia["Frequência"].sum()))
    col3.metric("Palavras únicas", len(frequencia))

    st.subheader("Palavras que mais se repetem")
    st.bar_chart(frequencia.head(TOP_GRAFICO), x="Palavra", y="Frequência", sort="-Frequência")
    st.dataframe(frequencia, use_container_width=True, hide_index=True)

    campea = frequencia.iloc[0]
    st.success(
        f"Palavra mais frequente: **{campea['Palavra']}** "
        f"({campea['Frequência']} vezes, em {campea['Avaliações']} avaliação(ões))."
    )

    with st.expander("Ver frases guardadas e seus tokens"):
        st.dataframe(avaliacoes, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
