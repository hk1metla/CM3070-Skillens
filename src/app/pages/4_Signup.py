import streamlit as st

from src.app.ui import inject_css, render_nav
from src.app.views import render_signup


def main() -> None:
    inject_css()
    render_nav()
    render_signup()


if __name__ == "__main__":
    main()
