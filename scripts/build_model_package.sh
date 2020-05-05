# Set up environment
virtualenv venv
. venv/bin/activate
pip install --upgrade pip
cd packages/football_trading
pip install -r requirements.txt
# Create environment variables
export IN_PRODUCTION=false >> $BASH_ENV
export PROJECTSPATH=/home/circleci/project/packages/football_trading >> $BASH_ENV
export DB_DIR=/home/circleci/project/db.sqlite >> $BASH_ENV
export BFEX_USER=$CCI_BFEX_USER >> $BASH_ENV
export BFEX_PASSWORD=$CCI_BFEX_PASS >> $BASH_ENV
export BFEX_APP_KEY=$CCI_BFEX_APP_KEY >> $BASH_ENV
export BFEX_CERTS_PATH=$CCI_BFEX_CERTS_PATH >> $BASH_ENV
export PRODUCTION_MODEL_NAME=match-predict-base >> $BASH_ENV
export RECREATE_DB=false >> $BASH_ENV
export LOCAL=true >> $BASH_ENV
# Run tox to build the package
tox
