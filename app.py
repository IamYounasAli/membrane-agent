"""Paper Research Agent - Streamlit app.

Pages: Ask (library / global) · Extract records · Records · Data check
All API keys come from Streamlit secrets. Nothing is stored in the code.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import agent, config, data_loader, extractor, global_search, llm, record_store, schema
from core.embedder import QueryEmbedder
from core.vector_store import VectorStore

st.set_page_config(page_title="Paper Research Agent", page_icon="🧪", layout="wide")
record_store.init_db()

MODE_LIB = "📚 Search my library"
MODE_GLO = "🌍 Search globally"


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading your library...")
def load_library():
    lib = data_loader.load_library()
    store = VectorStore(lib.chunks, lib.embeddings) if lib.ok else None
    return lib, store


@st.cache_resource(show_spinner="Loading the embedding model (the first start downloads it)...")
def load_embedder():
    return QueryEmbedder(config.embedding_model_name(), config.embedding_backend(), config.query_prefix())


def show_library_problems(lib):
    st.error("Your library files are not ready.")
    for e in lib.errors:
        st.markdown(f"- ❌ {e}")
    st.markdown(
        "Add these files to the repo, then reboot the app:\n"
        "- `data/imported/chunks.jsonl`\n- `data/imported/embeddings.npy`\n- `data/imported/embedding_info.json`\n\n"
        "The exact format is explained in the README and on the **Data check** page."
    )


# ----------------------------------------------------------------------------- rendering helpers
def records_frame(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    cols = [c for c in schema.DISPLAY_COLUMNS if c in df.columns]
    return df[cols] if cols else df


def render_chart(records: list, column: str):
    rows = [r for r in records if r.get(column) is not None]
    if len(rows) < 2:
        return
    df = pd.DataFrame({
        "label": [f"{i + 1}. {r.get('membrane') or r.get('solute') or 'record'}" for i, r in enumerate(rows)],
        column: [r[column] for r in rows],
    })
    st.bar_chart(df.set_index("label")[column])


def render_assistant(out: dict, mode: str):
    st.markdown(out["answer"])
    if mode == MODE_LIB:
        if out.get("warnings"):
            st.warning("**Check before comparing**\n\n" + "\n".join(f"- {w}" for w in out["warnings"]))
        if out.get("records"):
            with st.expander(f"📊 Extracted records used ({len(out['records'])})"):
                st.dataframe(records_frame(out["records"]), hide_index=True)
                render_chart(out["records"], out["plan"].get("sort_by") or "rejection_pct")
        if out.get("chunks"):
            with st.expander(f"📄 Evidence excerpts ({len(out['chunks'])})"):
                for c in out["chunks"]:
                    st.markdown(f"**{c['source']} · p.{c.get('page')}** · {c.get('chunk_type')} · score {c['score']}")
                    st.caption(c["text"][:700] + ("…" if len(c["text"]) > 700 else ""))
        st.caption(f"Answered by {out['provider']} · retrieval plan: {out['plan']['intent']}")
    else:
        for p in out.get("problems", []):
            st.warning(p)
        if out.get("results"):
            with st.expander(f"🔗 Sources ({len(out['results'])})"):
                for i, r in enumerate(out["results"], start=1):
                    extra = " · ".join(str(x) for x in (r.get("year"), r.get("journal"),
                                                        f"cited {r['cited_by']}×" if r.get("cited_by") else None) if x)
                    st.markdown(f"**[{i}]** [{r['title']}]({r.get('url')})  \n{extra}")
                    if r.get("oa_url"):
                        st.markdown(f"Open-access PDF/page: {r['oa_url']}")
        st.caption(f"Search query used: “{out.get('query')}” · answered by {out.get('provider')}")


# ----------------------------------------------------------------------------- page: ask
def page_ask():
    st.title("🧪 Paper Research Agent")
    mode = st.sidebar.radio("Search mode", [MODE_LIB, MODE_GLO],
                            help="Library = your own papers. Global = OpenAlex papers + web.")
    lib, store = load_library()

    if not llm.configured_providers():
        st.error("No LLM key found. Add `GEMINI_API_KEY` (free) in Streamlit → App settings → Secrets.")
        return

    if mode == MODE_LIB:
        hint, examples, placeholder = schema.LIBRARY_HINT, schema.LIBRARY_EXAMPLES, schema.LIBRARY_PLACEHOLDER
        if not lib.ok:
            show_library_problems(lib)
            return
    else:
        hint, examples, placeholder = schema.GLOBAL_HINT, schema.GLOBAL_EXAMPLES, schema.GLOBAL_PLACEHOLDER

    st.info(hint)

    source = None
    k = 6
    use_planner = True
    with st.sidebar:
        if mode == MODE_LIB:
            choice = st.selectbox("Limit to one paper (optional)", ["All papers"] + store.sources)
            source = None if choice == "All papers" else choice
            k = st.slider("Excerpts to retrieve", 3, 12, 6)
            use_planner = st.checkbox("Use AI to plan the search", value=True,
                                      help="Uses 1 extra free-tier request per question for better filters.")
        else:
            use_papers = st.checkbox("OpenAlex (scientific papers)", value=True)
            use_web = st.checkbox("Tavily (general web)", value=True)

    state = st.session_state.setdefault("messages", {MODE_LIB: [], MODE_GLO: []})
    history = state[mode]

    for m in history:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            elif "error" in m:
                st.error(m["error"])
            else:
                render_assistant(m["out"], mode)

    if not history:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, q in enumerate(examples):
            if cols[i % 2].button(q, key=f"ex_{mode}_{i}"):
                st.session_state["pending"] = q
                st.rerun()
    else:
        with st.expander("💡 Suggested questions"):
            for i, q in enumerate(examples):
                if st.button(q, key=f"ex2_{mode}_{i}"):
                    st.session_state["pending"] = q
                    st.rerun()

    question = st.chat_input(placeholder)
    if not question and st.session_state.get("pending"):
        question = st.session_state.pop("pending")
    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your library..." if mode == MODE_LIB else "Searching papers and the web..."):
                if mode == MODE_LIB:
                    embedder = load_embedder()
                    if embedder.embed("dimension check").shape[0] != lib.stats["dimension"]:
                        raise RuntimeError(
                            f"Embedding mismatch: your stored vectors have {lib.stats['dimension']} dimensions but the "
                            f"model '{embedder.model_name}' produces a different size. Set the correct model name in "
                            "data/imported/embedding_info.json (see the Data check page).")
                    out = agent.answer_library(question, store, embedder, k=k, use_planner=use_planner, source=source)
                else:
                    out = agent.answer_global(question, use_papers=use_papers, use_web=use_web)
        except (llm.LLMError, RuntimeError, global_search.SearchError) as e:
            st.error(str(e))
            history.append({"role": "assistant", "error": str(e)})
            return
        render_assistant(out, mode)
        history.append({"role": "assistant", "out": out})


# ----------------------------------------------------------------------------- page: extract
def page_extract():
    st.title("🧪 Extract structured records")
    st.write(
        "Reads the most relevant chunks of each paper (tables, results, numbers with units), asks the LLM for "
        "candidate values, then **code** converts units, checks ranges and verifies each number appears in the "
        "source text. Every paper is processed once and stored."
    )
    lib, store = load_library()
    if not lib.ok:
        show_library_problems(lib)
        return
    if not llm.configured_providers():
        st.error("No LLM key found in Streamlit secrets.")
        return

    processed = record_store.processed_sources()
    chosen = st.multiselect("Papers to process", store.sources,
                            format_func=lambda s: f"✅ {s}" if s in processed else s,
                            help="✅ = already processed. Free tiers have daily limits - do a few papers at a time.")
    max_chunks = st.slider("Max chunks per paper", 10, 80, 30,
                           help="Fewer chunks = fewer tokens. 30 is enough for most papers.")
    st.caption("Tip: free LLM tiers have daily request limits. If a limit is hit, extraction stops safely - "
               "run the same paper again later; duplicates are ignored.")

    if st.button("▶ Run extraction", type="primary", disabled=not chosen):
        bar = st.progress(0.0, text="Starting...")
        for pi, src in enumerate(chosen):
            def cb(done, total, msg, pi=pi, src=src):
                frac = (pi + done / max(total, 1)) / len(chosen)
                bar.progress(min(frac, 1.0), text=f"{src[:60]} - {msg}")

            summary = extractor.run_extraction(lib.chunks, src, max_chunks, cb)
            line = (f"**{src}** - {summary['chunks_selected']} chunks → {summary['records_added']} new records"
                    + (f" (via {', '.join(summary['providers'])})" if summary["providers"] else ""))
            if summary["stopped"]:
                st.warning(line + f"\n\nStopped: {summary['stopped']}")
                break
            st.success(line)
        bar.empty()

    st.divider()
    st.subheader("Keep your records permanently")
    st.write(
        f"Records currently stored: **{record_store.count()}**. Streamlit Cloud forgets files when the app restarts, "
        "so download the records and commit them to your repo as `data/records_seed.json` - "
        "the app loads that file automatically."
    )
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download records (records_seed.json)", record_store.export_json(),
                       file_name="records_seed.json", mime="application/json",
                       disabled=record_store.count() == 0)
    up = c2.file_uploader("Restore records from a JSON file", type="json")
    if up is not None and c2.button("Import this file"):
        try:
            n = record_store.import_json(up.getvalue().decode("utf-8"))
            st.success(f"Imported {n} new records.")
        except Exception as e:
            st.error(f"Could not import: {e}")


# ----------------------------------------------------------------------------- page: records
def page_records():
    st.title("📊 Records")
    rows = record_store.all_records()
    if not rows:
        st.info("No records yet. Use the **Extract records** page first.")
        return
    df = records_frame(rows)
    c1, c2 = st.columns(2)
    src = c1.selectbox("Paper", ["All"] + sorted(df["source"].dropna().unique().tolist()))
    flagged_only = c2.checkbox("Only records with flags")
    if src != "All":
        df = df[df["source"] == src]
    if flagged_only:
        df = df[df["flags"].fillna("") != ""]
    st.caption("Flags such as `not_in_text:*` mean a number could not be found in the source chunk - check it before trusting it.")
    st.dataframe(df, hide_index=True)
    st.download_button("⬇ Download as CSV", df.to_csv(index=False), file_name="records.csv", mime="text/csv")
    with st.expander("Danger zone"):
        if st.checkbox("I understand this deletes all records") and st.button("Delete all records"):
            record_store.clear()
            st.rerun()


# ----------------------------------------------------------------------------- page: data check
def page_check():
    st.title("🩺 Data check")
    st.write("Use this page after deploying to confirm everything is wired correctly.")

    st.subheader("1. Your imported files")
    files = [("data/imported/chunks.jsonl", config.CHUNKS_FILE, "required"),
             ("data/imported/embeddings.npy", config.EMBEDDINGS_FILE, "required"),
             ("data/imported/embedding_info.json", config.EMBED_INFO_FILE, "strongly recommended"),
             ("data/imported/metadata.jsonl", config.METADATA_FILE, "optional"),
             ("data/records_seed.json", config.RECORDS_SEED_FILE, "optional")]
    st.table(pd.DataFrame([{"file": n, "status": "✅ found" if p.exists() else "❌ missing", "need": need}
                           for n, p, need in files]))

    lib, store = load_library()
    if lib.ok:
        st.success("Library loaded.")
        st.json(lib.stats)
        for w in lib.warnings:
            st.warning(w)
    else:
        show_library_problems(lib)

    st.subheader("2. Embedding model")
    st.write(f"Model used for questions: `{config.embedding_model_name()}` (backend: `{config.embedding_backend()}`)")
    if st.button("Test embedding model"):
        try:
            dim = load_embedder().embed("membrane rejection test").shape[0]
            if lib.ok and dim != lib.stats["dimension"]:
                st.error(f"Dimension {dim} ≠ stored vectors {lib.stats['dimension']}. The model name is wrong for your embeddings.")
            else:
                st.success(f"Model works - {dim} dimensions" + (" and matches your stored vectors." if lib.ok else "."))
        except Exception as e:
            st.error(f"Could not load the embedding model: {e}")

    st.subheader("3. Secrets and connections")
    keys = ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENALEX_API_KEY", "TAVILY_API_KEY"]
    st.table(pd.DataFrame([{"secret": k, "status": "✅ set" if config.secret(k) else "— not set"} for k in keys]))
    c1, c2, c3 = st.columns(3)
    if c1.button("Test LLM"):
        try:
            r = llm.call_llm("Reply with the single word: OK", max_tokens=50)
            st.success(f"{r.provider} ({r.model}) replied: {r.text.strip()[:40]}")
        except llm.LLMError as e:
            st.error(str(e))
    if c2.button("Test OpenAlex"):
        try:
            st.success(f"OpenAlex works - {len(global_search.openalex_search('membrane filtration', 2))} results")
        except global_search.SearchError as e:
            st.error(str(e))
    if c3.button("Test Tavily"):
        try:
            st.success(f"Tavily works - {len(global_search.tavily_search('membrane filtration', 2))} results")
        except global_search.SearchError as e:
            st.error(str(e))


# ----------------------------------------------------------------------------- navigation
PAGES = {"💬 Ask": page_ask, "🧪 Extract records": page_extract, "📊 Records": page_records, "🩺 Data check": page_check}
with st.sidebar:
    st.header("Paper Research Agent")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    lib_status, _ = load_library()
    if lib_status.ok:
        st.caption(f"Library: {lib_status.stats['chunks']} chunks · {lib_status.stats['papers']} papers")
    else:
        st.caption("Library: not loaded")
    st.caption(f"Records: {record_store.count()} · LLMs: {', '.join(llm.configured_providers()) or 'none'}")
    st.divider()

PAGES[page]()
