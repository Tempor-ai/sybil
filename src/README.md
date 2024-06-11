# Installing Appserver on localhost

1) Pull repo from Github.com

```bash
git clone git@github.com:Tempor-ai/sybil.git
```

2) Build and run docker image

```bash
cd {repo_location}/src
docker run -p 8000:80 --rm -e MODULE_NAME="app" -e VARIABLE_NAME="app" -e TIMEOUT="1200" -it $(docker build -q .)
# -p 80:80 expose server on port 80
```

# Installing on localhost without Docker

```bash
git clone git@github.com:Tempor-ai/sybil.git
```

2) Create local Virtual environment and install dependencies

```bash
cd {repo_location}/src
python3 -m venv ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt
```

3) Load appserver with uvicorn(Single threaded)

```bash
cd {repo_location}/src
uvicorn appserver:app --reload
```

Do we need to keep the tran testing split in NP or do we want to override it with Sybil testing split.

