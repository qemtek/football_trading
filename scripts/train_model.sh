export IN_PRODUCTION=false
export PROJECTSPATH=/home/circleci/project/packages/football_trading/football_trading
export DB_DIR=/home/circleci/project/db.sqlite
export BFEX_USER=$CCI_BFEX_USER
export BFEX_PASSWORD=$CCI_BFEX_PASS
export BFEX_APP_KEY=$CCI_BFEX_APP_KEY
export BFEX_CERTS_PATH=$CCI_BFEX_CERTS_PATH
export PRODUCTION_MODEL_NAME=match-predict-base
export RECREATE_DB=false
export LOCAL=false
export S3_BUCKET_NAME=$S3_BUCKET_NAME
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export PYTHONPATH=./packages/football_trading
export SERVER_ADDRESS=0.0.0.0
export SERVER_PORT=5000

cd packages/ft_api
chmod u+x run.sh
./run.sh
status=$(curl -v http://${SERVER_ADDRESS}:${SERVER_PORT}/train)
# python ./packages/football_trading/football_trading/train.py