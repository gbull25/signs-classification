#!/bin/bash

## Repeat command until port 1221 on address db is not ready.
until nc -z -v -w30 db 1221
do
echo "Waiting for database connection for 5 seconds..."

## Wait for 5 seconds before check again.
sleep 5
done
echo "Database server ready..."

#### run your server afterwards

cd ./app/

alembic upgrade head

cd ../ 

gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker  --bind=0.0.0.0:80 --timeout=5000
