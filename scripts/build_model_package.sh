# Set up environment
virtualenv venv
. venv/bin/activate
pip install --upgrade pip
cd packages/football_trading
pip install -r requirements.txt
# Run tox to build the package
tox
