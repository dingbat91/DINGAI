import beam

app = beam.App(
    name='DingAI',
    cpu=2,
    memory='8Gi',
    python_packages=["langchain","transformers","torch","numpy","python-dotenv"],
    commands=["apt-get update"]
)

