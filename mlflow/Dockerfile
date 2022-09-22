FROM python:3.10.5

RUN useradd --create-home wec
USER wec
ENV PATH="/home/wec/.local/bin:${PATH}"

WORKDIR /home/wec/mlflow

COPY --chown=wec . .
RUN pip install --user -r requirements.txt

CMD mlflow server \ 
    --backend-store-uri postgresql://$DB_USERNAME:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME \
    --artifacts-destination $S3_BUCKET_URI \ 
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000
    