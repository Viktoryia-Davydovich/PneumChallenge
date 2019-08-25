from tornado.web import RequestHandler


class PredictHandler(RequestHandler):

    def post(self, dicom_image):
        self.write("Hello, world!")
