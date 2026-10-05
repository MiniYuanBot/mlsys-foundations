"""Synthetic unittest outcomes used only to check grader behavior."""
import unittest


class Pass(unittest.TestCase):
    def test_pass(self):
        self.assertTrue(True)


class Environment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise unittest.SkipTest("ENV: synthetic missing dependency")

    def test_unused(self):
        pass


class Unimplemented(unittest.TestCase):
    def test_todo(self):
        raise NotImplementedError("synthetic student TODO")


class Skipped(unittest.TestCase):
    @unittest.skip("synthetic voluntary skip")
    def test_skip(self):
        pass


class Empty(unittest.TestCase):
    pass

