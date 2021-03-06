cd ..
virtualenv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r packages/football_trading/requirements.txt
export IN_PRODUCTION=false
#export PROJECTSPATH=/home/circleci/project/packages/football_trading/football_trading
export DB_DIR=/home/circleci/project/db.sqlite
export BFEX_USER=$CCI_BFEX_USER
export BFEX_PASSWORD=$CCI_BFEX_PASS
export BFEX_APP_KEY=$CCI_BFEX_APP_KEY
export BFEX_CERTS_PATH=$CCI_BFEX_CERTS_PATH
export PRODUCTION_MODEL_NAME=match-predict-base
export S3_BUCKET_NAME=$S3_BUCKET_NAME
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export RECREATE_DB=false
export LOCAL=true
# Should we download any data here?
py.test -vv packages/football_trading/tests