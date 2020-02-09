

def test_empty_db(client):
    """Start with a blank database."""
    rv = client.get('/')
    assert b'No entries here so far' in rv.data


def test_next_games():
    pass


def test_prerdict():
    pass


def test_get_historic_predictions():
    pass


def test_get_all_historic_predictions():
    pass


def latest_model_id():
    pass


def test_update_tables():
    pass
