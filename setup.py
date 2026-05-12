from setuptools import setup, find_packages

setup(
    name="neuro_safety_guard",
    version="0.1.0",
    description="Real-Time Cognitive Overload Detection for Construction Safety via fNIRS–EEG Fusion",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "streamlit>=1.35.0",
        "plotly>=5.21.0",
    ],
)
