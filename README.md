# FootballTrading - Predictive Model for Premier League Football

[Currently a work in progress]

This project aims to provide weekly predictions for the English Premier League through a simple API, accompanied by an interactive Dash Dashboard.

The project utilizes publicly available data from various sources to produce odds for Home/Draw/Away for upcoming games. These odds can be used to spot inefficiencies in betting markets, which provide the potential of profiting from them.

An XGBoost model is used, built inside a modular, re-usable project architecture that I have designed and use in my own projects. Feel free to copy what you find useful, and if you ahve any questions please dont hesitate to contact me at qemtek@gmail.com

Thanks for visiting!

Environment variables required for the project to run:

Project Specific Credentials
+ PROJECTSPATH - The path to the project (root/packages/football_trading)
+ DB_DIR - The location of the DB locally (set to {your_project_root}/db.sqlite)
+ RECREATE_DB - Whether to do a full refresh of the database
+ LOCAL - Whether to store data locally or in S3
+ IN_PRODUCTION - Whether the model is being run in production (set to false normally)
+ PRODUCTION_MODEL_NAME - A prefix used to select the right model if multiple models are in production

S3 Credentials
+ S3_BUCKET_NAME - The name of the S3 bucket you want to use for storing data
+ AWS_ACCESS_KEY_ID - Your AWS access key id
+ AWS_SECRET_ACCESS_KEY - Your AWS secret access key

Betfair Exchange API Credentials
+ BFEX_USER - Your betfair username
+ BFEX_PASSWORD - Your Betfair password
+ BFEX_APP_KEY - Your Betfair Exchange API app key (you need to request one from Betfair)
+ BFEX_CERTS_PATH - Path to a folder containing your Betfair Exchange API certs. You can find a limited set that I own in the root directory under /certs.
