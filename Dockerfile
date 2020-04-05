FROM python:3.7.0

# Create the user that will run the app
RUN adduser —disabled-password —gecos '' ml-api-user

WORKDIR /opt/api

ARG PIP_EXTRA_INDEX_URL
ENV FLASK_APP run.py

# Install requirements
ADD ./packages/api /opt/api
RUN pip install --upgrade pip
RRUN pip install -r /opt/api/requirements.txt

RUN chmod +x /opt/api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 5000

CMD ["bash", "./run.sh"]