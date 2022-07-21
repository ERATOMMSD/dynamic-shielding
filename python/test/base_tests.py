import unittest

from py4j.java_gateway import JavaGateway, CallbackServerParameters


class Py4JTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())

    def tearDown(self) -> None:
        # close the gateway
        self.gateway.close()
