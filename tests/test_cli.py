import importlib.util
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
MODULE_PATH = os.path.join(ROOT, "__main__.py")


spec = importlib.util.spec_from_file_location("stormspyder_main", MODULE_PATH)
stormspyder_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stormspyder_main)


class ParseArgsTests(unittest.TestCase):
    def test_defaults_to_production_browser_setup(self):
        args = stormspyder_main.parse_args([])
        self.assertFalse(args.local_testing)

    def test_accepts_local_testing_flag(self):
        args = stormspyder_main.parse_args(["--local-testing"])
        self.assertTrue(args.local_testing)


if __name__ == "__main__":
    unittest.main()
