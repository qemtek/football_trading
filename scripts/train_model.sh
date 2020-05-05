cd ..
. venv/bin/activate
export IN_PRODUCTION=false
export PROJECTSPATH=/home/circleci/project/packages/football_trading
export DB_DIR=/home/circleci/project/db.sqlite
export BFEX_USER=$CCI_BFEX_USER
export BFEX_PASSWORD=$CCI_BFEX_PASS
export BFEX_APP_KEY=$CCI_BFEX_APP_KEY
export BFEX_CERTS_PATH=$CCI_BFEX_CERTS_PATH
export PRODUCTION_MODEL_NAME=match-predict-base
export RECREATE_DB=false
export LOCAL=false
export PYTHONPATH=./packages/football_trading
export SERVER_ADDRESS=0.0.0.0
export SERVER_PORT=5000

python packages/api/run.py
status=$(curl -v http://${SERVER_ADDRESS}:${SERVER_PORT}/train)
# python ./packages/football_trading/football_trading/train.py