FROM python:3.10.5

RUN useradd --create-home wec
USER wec
ENV PATH="/home/wec/.local/bin:${PATH}"

WORKDIR /home/wec
RUN pip install --upgrade pip wheel

COPY --chown=wec requirements.txt /tmp/requirements.txt
RUN pip install --user -r /tmp/requirements.txt

WORKDIR /app

COPY --chown=wec . .
RUN pip install --user -e .

CMD echo "You need to provide command using compose!"