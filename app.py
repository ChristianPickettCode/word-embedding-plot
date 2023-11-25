import numpy as np
import re
import streamlit as st
import altair as alt
from sklearn.manifold import TSNE
import pandas as pd
import cohere
import plotly.express as px
from plotly1 import plotly_events
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import openai
import tiktoken
from pydantic import BaseModel
import json
import plotly.io as pio

load_dotenv()

cohere_key: str = os.getenv("COHERE_KEY")
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
openai_key: str = os.getenv("OPENAI_KEY")

co = cohere.Client(cohere_key)  # This is your trial API key
openai.api_key = openai_key

MODELTODB = {
    'embed-english-light-v2.0': 'ten24',
    'text-embedding-ada-002': 'fifteen36'
}


class Textbook_Embedding(BaseModel):
    textbook: str
    chapter: str
    section: str
    body: str
    model: str
    embedding: list
    raw_body: str
    # id: UUID | None
    # created_at: datetime | None
    # updated_at: datetime | None


def get_cohere_embeddings(model: str, str_arr: list):
    response = co.embed(
        model=model,
        texts=str_arr)
    return np.array(response.embeddings)


def get_openai_embeddings(model: str, str_arr: list):
    embeddings = openai.Embedding.create(
        model=model,
        input=str_arr
    )
    embeddings_array = list(map(lambda x: x["embedding"], embeddings["data"]))
    return np.array(embeddings_array)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
# print(num_tokens_from_string("tiktoken is great!", "cl100k_base"))


def extract_sections_from_markdown(file_path):
    sections = []
    first_lines = []
    token_size_arr = []
    raw_arr = []
    with open(file_path, 'r') as file:
        content = file.read()

        raw_sections = re.split(r'^####(?!#)', content, flags=re.MULTILINE)
        for raw_section in raw_sections:
            section = raw_section.strip()
            if section:
                raw_arr.append(section)
                section = re.sub(
                    r'!\[\]\(https://cdn\.mathpix\.com/.*?\n', '', section)
                token_size_arr.append(
                    num_tokens_from_string(section, "cl100k_base"))
                lines = section.split('\n')
                first_line = lines[0]
                section = section[len(first_line):].replace('\n', ' ')
                sections.append(section)
                first_lines.append(first_line)
    return sections, first_lines, token_size_arr, raw_arr


def plot_fct_alt(perp: int, textbook_embedding_arr: list[Textbook_Embedding]):
    array_of_arrays = []
    titles = []
    for emb in textbook_embedding_arr:
        array_of_arrays.append(emb.embedding)
        titles.append(emb.section)

    array_of_arrays_np = np.array(array_of_arrays)

    # Applying t-SNE to compress subarrays into two values (x and y)
    tsne = TSNE(n_components=2, perplexity=perp)
    compressed_array = tsne.fit_transform(array_of_arrays_np)

    # Create a DataFrame for the scatter plot
    data = pd.DataFrame(
        {'x': compressed_array[:, 0], 'y': compressed_array[:, 1], 'label': titles})
    # st.dataframe(data)

    # Create the interactive scatter plot with labels
    scatter_plot = alt.Chart(data).mark_circle().encode(
        x='x',
        y='y',
        color='label',
        tooltip='label'
    ).interactive()

    # Show the scatter plot in Streamlit
    st.altair_chart(scatter_plot, use_container_width=True)


def plot_fct_plotly(perp: int, textbook_embedding_arr: list[Textbook_Embedding]):
    array_of_arrays = []
    titles = []
    for emb in textbook_embedding_arr:
        array_of_arrays.append(emb.embedding)
        titles.append(emb.section)

    array_of_arrays_np = np.array(array_of_arrays)

    # Applying t-SNE to compress subarrays into two values (x and y)
    tsne = TSNE(n_components=2, perplexity=perp)
    compressed_array = tsne.fit_transform(array_of_arrays_np)

    # Create a DataFrame for the scatter plot
    data = pd.DataFrame(
        {'x': compressed_array[:, 0], 'y': compressed_array[:, 1], 'label': titles})
    # st.dataframe(data)

    fig = px.scatter(
        data,
        x="x",
        y="y",
        color="label",
        # hover_name="label",
        hover_data=["label"],
        color_discrete_sequence=[
            "#2D4356",
            "#435B66",
            "#A76F6F",
            "#D21312",
            "#0E2954",
            "#2E8A99",
            "#84A7A1",
            "#5F264A",
            "#643A6B",
            "#957777",
            "#B0A4A4"
        ],
    )

    pio.templates.default = "plotly"
    fig.update(layout_showlegend=False,
               layout_xaxis_title=None, layout_yaxis_title=None, layout_xaxis_showgrid=False, layout_xaxis_showline=False, layout_yaxis_showgrid=False, layout_yaxis_showline=False, layout_xaxis_zeroline=False, layout_yaxis_zeroline=False, layout_xaxis_visible=False, layout_yaxis_visible=False)
    fig.update_traces(mode="markers+lines")
    fig.update_traces(hovertemplate=' ')
    # fig.update_layout(width=700)
    # fig.update_layout(
    #     showlegend=False,
    #     xaxis=dict(visible=False),
    #     yaxis=dict(visible=False),
    # )
    # fig.update_layout(
    #     hoverlabel=dict(
    #         bgcolor="white",
    #         font_size=16,
    #         font_family="Rockwell"
    #     )
    # )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    # fig.update(config_displayModeBar=False)
    return fig, data


def save_to_db(tb_emb: Textbook_Embedding):
    supabase: Client = create_client(url, key)
    db = MODELTODB[tb_emb.model]

    if db:
        return supabase.table(db).insert(tb_emb.dict()).execute()

    st.Warning('MODEL NOT SUPPORTED')
    return None, None


def get_from_db(model):
    supabase: Client = create_client(url, key)
    db = MODELTODB[model]
    response = supabase.table(db).select("*").execute()
    tb_embeddings = []
    for emb in response.data:
        new_emb = Textbook_Embedding(
            textbook=emb['textbook'],
            chapter=emb['chapter'],
            section=emb['section'],
            body=emb['body'],
            model=emb['model'],
            embedding=json.loads(emb['embedding']),
            raw_body=emb['raw_body'],
        )
        tb_embeddings.append(new_emb)
    st.session_state['textbook_embedding_arr'] = tb_embeddings


st.set_page_config(layout="wide")
st.title('Plot word embeddings for Textbook')

with st.expander("Parse & Fetch Embeddings"):
    markdown_file = st.text_input('Markdown file', 'probml-chp2.md')
    parse_md = st.button('Parse MD file')
    if parse_md:
        sections, first_lines, token_sizes, raw_arr = extract_sections_from_markdown(
            markdown_file)
        st.session_state['sections'] = sections[1:]
        st.session_state['titles'] = first_lines[1:]
        st.session_state['raw_arr'] = raw_arr[1:]
        st.write('file parsed')

    get_emb_cohere = st.button('Get cohere embeddings')
    if get_emb_cohere and 'sections' in st.session_state:
        titles = st.session_state.titles
        sections = st.session_state.sections
        raw_arr = st.session_state.raw_arr
        embeddings = get_cohere_embeddings(
            'embed-english-light-v2.0', sections)

        if len(embeddings) != len(titles) != len(sections):
            st.warning('title and and embedding arr not same length')
        else:
            local_textbook_embeddings: list[Textbook_Embedding] = []
            for i in range(len(embeddings)):
                sec = titles[i]
                bd = sections[i]
                emb = embeddings[i]
                raw_bd = raw_arr[i]
                tb_emb = Textbook_Embedding(textbook="probml", chapter="chpter 1", section=sec,
                                            body=bd, model="embed-english-light-v2.0", embedding=emb.tolist(), raw_body=raw_bd)
                local_textbook_embeddings.append(tb_emb)
            st.session_state['local_textbook_embeddings'] = local_textbook_embeddings
            st.write('got embeddings')

    get_emb_openai = st.button('Get openai embeddings')
    if get_emb_openai and 'sections' in st.session_state:
        titles = st.session_state.titles
        sections = st.session_state.sections
        raw_arr = st.session_state.raw_arr
        embeddings = get_openai_embeddings(
            'text-embedding-ada-002', sections)

        if len(embeddings) != len(titles) != len(sections):
            st.warning('title and and embedding arr not same length')
        else:
            local_textbook_embeddings: list[Textbook_Embedding] = []
            for i in range(len(embeddings)):
                sec = titles[i]
                bd = sections[i]
                emb = embeddings[i]
                raw_bd = raw_arr[i]
                tb_emb = Textbook_Embedding(textbook="probml", chapter="chpter 2", section=sec,
                                            body=bd, model="text-embedding-ada-002", embedding=emb.tolist(), raw_body=raw_bd)
                local_textbook_embeddings.append(tb_emb)
            st.session_state['local_textbook_embeddings'] = local_textbook_embeddings
            st.write('got embeddings')

if 'db_expanded' not in st.session_state:
    st.session_state['db_expanded'] = True
with st.expander("DB Options", expanded=st.session_state.db_expanded):
    save_emb = st.button('Save embeddings to db')
    if save_emb and 'local_textbook_embeddings' in st.session_state:
        local_textbook_embeddings: list[Textbook_Embedding] = st.session_state.local_textbook_embeddings
        count = 0
        for local_emb in local_textbook_embeddings:
            save_to_db(local_emb)
            count += 1

        st.write('saved:', count)

    emb_type = st.selectbox(
        "What kind of embeddings?",
        ('cohere', 'openai'))
    load_emb = st.button('Load embeddings from db')
    if load_emb:
        if emb_type == 'cohere':
            get_from_db('embed-english-light-v2.0')
            st.write('loaded from cohere')
        elif emb_type == 'openai':
            get_from_db('text-embedding-ada-002')
            st.write('loaded from openai')
        else:
            st.warning('🚨 nothing chosen')
        st.session_state['db_expanded'] = False
        st.experimental_rerun()


plot_data = st.button('Scatterplot')
if plot_data and ('textbook_embedding_arr' in st.session_state or 'local_textbook_embeddings' in st.session_state):
    if 'textbook_embedding_arr' in st.session_state:
        arr = st.session_state.textbook_embedding_arr
    elif 'local_textbook_embeddings' in st.session_state:
        arr = st.session_state.local_textbook_embeddings

    fig, data = plot_fct_plotly(5, arr)
    st.session_state['fig'] = fig
    st.session_state['data'] = data

if 'fig' in st.session_state:
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    selected_point = plotly_events(st.session_state.fig)
    # st.write("Selected point:")
    # st.write(selected_point)
    # print(selected_point)
    dataf: pd.DataFrame = st.session_state['data']

    if selected_point != None and len(selected_point) > 0:
        selected_x = selected_point[0]['x']
        selected_y = selected_point[0]['y']
        matching_label = dataf.loc[(dataf['x'] == selected_x) & (
            dataf['y'] == selected_y), 'label']
        if not matching_label.empty:
            label = matching_label.iloc[0]  # Extract the first matching label
            # print(label)
            st.write("### {}".format(label))

            if 'textbook_embedding_arr' in st.session_state:
                arr = st.session_state.textbook_embedding_arr
            elif 'local_textbook_embeddings' in st.session_state:
                arr = st.session_state.local_textbook_embeddings
            # print(arr)

            current_section: Textbook_Embedding
            for emb in arr:
                if emb.section == label:
                    current_section = emb

            if current_section:
                st.markdown(current_section.raw_body)

        else:
            print("No matching label found.")
