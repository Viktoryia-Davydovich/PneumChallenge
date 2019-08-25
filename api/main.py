from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
from predict_handler import PredictHandler


def make_app():
    urls = [("/", PredictHandler)]
    return Application(urls)


if __name__ == '__main__':
    app = make_app()
    app.listen(8000)
    IOLoop.instance().start()
