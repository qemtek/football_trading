from api.app import get_api
from api.settings import SERVER_ADDRESS, SERVER_PORT

application = get_api()

if __name__ == '__main__':
    application.run(host=SERVER_ADDRESS, port=SERVER_PORT)
