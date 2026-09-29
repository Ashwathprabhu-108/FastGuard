from setuptools import setup, find_packages

setup(
    name="fastguard",
    version="0.1.0",
    description="AST-based breaking API contract change detector",
    author="Ashwath Prabhu",
    py_modules=["fastguard", "fastguard_cli"],
    include_package_data=True,
    package_data={
        "": ["fastguard_model.pkl"], 
    },
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib",
        "fastapi",
        "pydantic"
    ],
)
