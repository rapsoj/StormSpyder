import os
import sys
import unittest

from stormspyder.alerts import send_email_alert


class SendEmailAlertTests(unittest.TestCase):
    def test_skips_when_no_alert_data(self):
        empty_df = None
        with self.assertRaises(AttributeError):
            send_email_alert(empty_df, {'genesis_ts': 'tropical storm'})


if __name__ == '__main__':
    unittest.main()
