import streamlit as st
from pathlib import Path

pages_pth = Path("pages")

pages = [
    st.Page(pages_pth / "01_molecular_set_transformer.py", 
            title="Molecular Set Transformer"),

    st.Page(pages_pth / "02_ethanol_ecfp4_molecular_set_transformer.py",
            title="Ethanol MST")
]

pg = st.navigation(pages)
pg.run()