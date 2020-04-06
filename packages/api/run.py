from flask_apscheduler import APScheduler

from api.app import get_api
from settings import SERVER_ADDRESS, SERVER_PORT
from football_trading.predict import make_predictions
from football_trading.train import train_new_model
from football_trading.dashboard import get_dashboard_app

application = get_api()
# Add the dashboard
get_dashboard_app(server=application)
# Setup the scheduler
scheduler = APScheduler()
# it is also possible to enable the API directly
# scheduler.api_enabled = True
scheduler.init_app(application)
# Add scheduled jobs
# Run the train method every week (it wont actually train if there are no new games)
application.apscheduler.add_job(
    func=train_new_model, trigger='cron', day_of_week=2, hour=1, id='1', replace_existing=True, max_instances=1)
# Run the predict method every day (it wont actually run if there are no new games)
application.apscheduler.add_job(
    func=make_predictions, trigger='cron', hour=2, id='2', replace_existing=True, max_instances=1)

if __name__ == '__main__':
    scheduler.start()
    application.run(host=SERVER_ADDRESS, port=SERVER_PORT)
