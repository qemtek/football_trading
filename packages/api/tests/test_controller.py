def test_health_endpoint_returns_200(flask_test_client):
    response = flask_test_client.get('/health')
    assert response.status_code == 200
