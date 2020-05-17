# Set up environment
cd packages/football_trading
virtualenv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Run tox to build the package
tox
