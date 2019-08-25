from tornado.web import RequestHandler


class PredictHandler(RequestHandler):
    def get(self):
        self.write({'message': 'hello world'})
