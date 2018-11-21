FROM python:3.7.1

WORKDIR /app

RUN git clone https://github.com/MultiAgentLearning/playground

RUN pip install -r ./playground/requirements.txt

RUN pip install ./playground

EXPOSE 10080

ADD ./src /app/src

ENV NAME DiscoAgent

ENTRYPOINT ["python"]

CMD ["./src/dockerskin.py", "DiscoAgent"]
