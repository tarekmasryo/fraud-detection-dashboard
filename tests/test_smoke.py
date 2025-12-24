def test_smoke_imports():
    import sys

    assert sys.version_info >= (3, 10)


def test_app_module_importable():
    # Update module name if needed. This should be safe after deps install.
    # Example: import streamlit_app
    pass
